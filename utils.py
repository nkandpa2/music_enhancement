import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import matplotlib.pyplot as plt

def load(path, sample_rate):
    """
    Load audio sample at `path` and resample to `sample_rate`
    """
    #sample, _ = torchaudio.sox_effects.apply_effects_file(path, [['rate', str(sample_rate)]])
    sample, sr = torchaudio.load(path)
    sample = torchaudio.transforms.Resample(sr, sample_rate)(sample)
    return sample
 
def batch_convolution(x, f, pad_both_sides=True):
    """
    Do batch-elementwise convolution between a batch of signals `x` and batch of filters `f`
    x: (batch_size x channels x signal_length) size tensor
    f: (batch_size x channels x filter_length) size tensor
    pad_both_sides: Whether to zero-pad x on left and right or only on left (Default: True)
    """
    batch_size = x.shape[0]
    f = torch.flip(f, (2,))
    if pad_both_sides:
        x = F.pad(x, (f.shape[2]//2, f.shape[2]-f.shape[2]//2-1))
    else:
        x = F.pad(x, (f.shape[2]-1, 0))
    #TODO: This assumes single-channel audio, fine for now 
    return F.conv1d(x.view(1, batch_size, -1), f, groups=batch_size).view(batch_size, 1, -1)

def plot_waveform(waveform):
    figure = plt.figure(figsize=(8,8))
    plt.plot(waveform.detach().cpu().numpy())
    plt.xlabel('time')
    return figure

def plot_spectrogram(spectrogram):
    figure = plt.figure(figsize=(8,8))
    plt.pcolormesh(spectrogram.log2().detach().cpu().numpy(), shading='gouraud')
    plt.xlabel('time')
    plt.ylabel('frequency')
    return figure

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
