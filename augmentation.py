import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from scipy import signal
import pdb

import utils

def augment(sample, rir=None, noise=None, eq_model=None, low_cut_model=None, rate=16000, nsr_range=[-30,-5], normalize=True, eps=1e-6):
    sample = perturb_silence(sample, eps=eps)
    clean_sample = torch.clone(sample)
    if not noise is None:
        nsr_target = ((nsr_range[1] - nsr_range[0])*torch.rand(noise.shape[0]) + nsr_range[0]).to(noise)
        sample = apply_noise(sample, noise, nsr_target)
    if not rir is None:
        sample = apply_reverb(sample, rir, None, rate=rate)
    if not eq_model is None:
        sample = eq_model(sample)
    if not low_cut_model is None:
        sample = low_cut_model(sample)
    if normalize:
        sample = 0.95*sample/sample.abs().max(dim=2, keepdim=True)[0]

    return clean_sample, sample

def normalize(sample):
    return 0.95*sample/sample.abs().max(dim=2, keepdim=True)[0]

def perturb_silence(sample, eps=1e-6):
    """
    Some samples have periods of silence which can cause numerical issues when taking log-spectrograms. Add a little noise
    """
    return sample + eps*torch.randn_like(sample)

def apply_reverb(sample, rir, drr_target, rate=16000):
    """
    Convolve batch of samples with batch of room impulse responses scaled to achieve a target direct-to-reverberation ratio
    """
    if not drr_target is None:
        direct_ir, reverb_ir = decompose_rir(rir, rate=rate)
        drr_db = drr(direct_ir, reverb_ir)
        scale = 10**((drr_db - drr_target)/20)
        reverb_ir_scaled = scale[:, None, None]*reverb_ir
        rir_scaled = torch.cat((direct_ir, reverb_ir_scaled), axis=2)
    else:
        rir_scaled = rir
    return utils.batch_convolution(sample, rir_scaled, pad_both_sides=False)

def apply_noise(sample, noise, nsr_target, peak=False):
    """
    Apply additive noise scaled to achieve target noise-to-signal ratio
    """
    if peak:
        nsr_curr = pnsr(sample, noise)
        noise_flat = noise.view(noise.shape[0], -1)
        peak_noise = noise_flat.max(dim=1)[0] - noise_flat.min(dim=1)[0]
        scale = 10**((nsr_target - nsr_curr)/20)
    else:
        nsr_curr = nsr(sample, noise)
        scale = torch.sqrt(10**((nsr_target - nsr_curr)/10))

    return sample + scale[:, None, None]*noise

def nsr(sample, noise):
    """
    Compute noise-to-signal ratio
    """
    sample, noise = sample.view(sample.shape[0], -1), noise.view(noise.shape[0], -1)
    signal_power = torch.square(sample).mean(dim=1)
    noise_power = torch.square(noise).mean(dim=1)
    return 10*torch.log10(noise_power/signal_power)

def pnsr(sample, noise):
    """
    Compute peak noise-to-signal-ratio
    """
    sample, noise = sample.view(sample.shape[0], -1), noise.view(noise.shape[0], -1)
    peak_noise = noise.max(dim=1)[0] - noise.min(dim=1)[0]
    signal = torch.square(sample).mean(dim=1)
    return 20*torch.log10(peak_noise) - 10*torch.log10(signal)

def drr(direct_ir, reverb_ir):
    """
    Compute direct-to-reverberation ratio
    """
    direct_ir_flat = direct_ir.view(direct_ir.shape[0], -1)
    reverb_ir_flat = reverb_ir.view(reverb_ir.shape[0], -1)
    drr_db = 10*torch.log10(torch.square(direct_ir_flat).sum(dim=1)/torch.square(reverb_ir_flat).sum(dim=1))
    return drr_db

def decompose_rir(rir, rate=16000, window_ms=5):
    direct_window = int(window_ms/1000*rate)
    direct_ir, reverb_ir = rir[:,:,:direct_window], rir[:,:,direct_window:]
    return direct_ir, reverb_ir

class MicrophoneEQ(nn.Module):
    """
    Apply a random EQ on bands demarcated by `bands`
    """
    def __init__(self, low_db=-15, hi_db=15, bands=[200, 1000, 4000], filter_length=8192, rate=16000):
        super(MicrophoneEQ, self).__init__()
        self.low_db = low_db
        self.hi_db = hi_db
        self.rate = rate
        self.filter_length = filter_length
        self.firs = nn.Parameter(self.create_filters(bands))

    def create_filters(self, bands):
        """
        Generate bank of FIR bandpass filters with band cutoffs specified by `bands`
        """
        ir = np.zeros([self.filter_length])
        ir[0] = 1
        bands = [35] + bands
        fir = np.zeros([len(bands) + 1, self.filter_length])
        for j in range(len(bands)):
            freq = bands[j] / (self.rate/2)
            bl, al = signal.butter(4, freq, btype='low')
            bh, ah = signal.butter(4, freq, btype='high')
            fir[j] = signal.lfilter(bl, al, ir)
            ir = signal.lfilter(bh, ah, ir)
        fir[-1] = ir
        pfir = np.square(np.abs(np.fft.fft(fir,axis=1)))
        pfir = np.real(np.fft.ifft(pfir, axis=1))
        fir = np.concatenate((pfir[:,self.filter_length//2:self.filter_length], pfir[:,0:self.filter_length//2]), axis=1)
        return torch.tensor(fir, dtype=torch.float32)

    def get_eq_filter(self, band_gains):
        """
        Apply `band_gains` to bank of FIR bandpass filters to get the final EQ filter
        """
        band_gains = 10**(band_gains/20)
        eq_filter = (band_gains[:,:,None] * self.firs[None,:,:]).sum(dim=1, keepdim=True)
        return eq_filter

    def forward(self, x):
        gains = (self.hi_db - self.low_db)*torch.rand(x.shape[0], self.firs.shape[0]-1, device=self.firs.device) + self.low_db
        gains = torch.cat((torch.zeros((x.shape[0], 1), device=self.firs.device), gains), dim=1)
        eq_filter = self.get_eq_filter(gains)
        eq_x = utils.batch_convolution(x, eq_filter, pad_both_sides=True)
        return eq_x

class LowCut(nn.Module):
    """
    Apply a random EQ on bands demarcated by `bands`
    """
    def __init__(self, cutoff_freq, filter_length=8191, rate=16000):
        super(LowCut, self).__init__()
        self.rate = rate
        self.filter_length = filter_length
        self.fir = nn.Parameter(self.create_filter(cutoff_freq))

    def create_filter(self, cutoff_freq):
        """
        Generate bank of FIR bandpass filters with band cutoffs specified by `bands`
        """
        ir = np.zeros([self.filter_length])
        ir[0] = 1
        freq = cutoff_freq / (self.rate/2)
        bh, ah = signal.butter(4, freq, btype='high')
        fir = signal.lfilter(bh, ah, ir)
        pfir = np.square(np.abs(np.fft.fft(fir,axis=0)))
        pfir = np.real(np.fft.ifft(pfir, axis=0))
        fir = np.concatenate((pfir[self.filter_length//2:self.filter_length], pfir[0:self.filter_length//2]), axis=0)
        fir = torch.tensor(fir, dtype=torch.float32)
        return torch.flip(fir, (0,))[None,None,:]

    def forward(self, x):
        return F.conv1d(x, self.fir, padding=self.filter_length//2)
        
