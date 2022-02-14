import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import numpy as np
import math
from tqdm import tqdm

import utils
import augmentation

def train_epoch(global_step, model_params, model, optimizer, train_loaders, summary_writer):
    data_loader, reverb_loader, noise_loader = train_loaders

    eq_model = augmentation.MicrophoneEQ(rate=model_params.sample_rate).cuda()
    melspec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=model_params.sample_rate,
                                                             n_fft=model_params.n_fft, 
                                                             hop_length=model_params.hop_length, 
                                                             n_mels=model_params.n_mels).cuda()

    loss_meter = utils.AverageMeter('loss', fmt=':.4f')

    beta = np.linspace(model_params.noise_min, model_params.noise_max, model_params.noise_scales)
    noise_level = np.cumprod(1 - beta)
    noise_level = torch.tensor(noise_level.astype(np.float32))

    loop = tqdm(data_loader)
    for i, x in enumerate(loop):
        batch_size = x.shape[0]
        x, noise, rir = x.cuda(), next(noise_loader).cuda(), next(reverb_loader).cuda()
        clean_x, aug_x = augmentation.augment(x, rir=rir, noise=noise, eq_model=eq_model)
        aug_spec = melspec_transform(aug_x).log2()
        t = torch.randint(0, model_params.noise_scales, [batch_size], device=clean_x.device)
        noise_scale = noise_level[t].cuda()
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(clean_x)
        noisy_x = noise_scale_sqrt[:,None,None] * clean_x + (1.0 - noise_scale[:,None,None])**0.5 * noise
        predicted = model(noisy_x, aug_spec, t)
        loss = nn.MSELoss()(noise, predicted[:,:,:noise.shape[2]])
        loss.backward()
        optimizer.step()
        model.zero_grad()
        loss_meter.update(loss.item(), n=x.shape[0])
        loop.set_postfix_str(str(loss_meter))
        summary_writer.add_scalar('training/loss', loss.item(), global_step)
        global_step += 1

    return global_step, model, optimizer


def validate(model_params, model, val_loaders):
    data_loader, reverb_loader, noise_loader = val_loaders

    eq_model = augmentation.MicrophoneEQ().cuda()
    melspec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=model_params.sample_rate,
                                                             n_fft=model_params.n_fft, 
                                                             hop_length=model_params.hop_length, 
                                                             n_mels=model_params.n_mels).cuda()

    loss_meter = utils.AverageMeter('loss', fmt=':.4f')

    beta = np.linspace(model_params.noise_min, model_params.noise_max, model_params.noise_scales)
    noise_level = np.cumprod(1 - beta)
    noise_level = torch.tensor(noise_level.astype(np.float32))
    
    with torch.no_grad():
        for x in data_loader: 
            batch_size = x.shape[0]
            x, noise, rir = x.cuda(), next(noise_loader).cuda(), next(reverb_loader).cuda()
            clean_x, aug_x = augmentation.augment(x, rir=rir, noise=noise, eq_model=eq_model)
            aug_spec = melspec_transform(aug_x).log2()
            t = torch.randint(0, model_params.noise_scales, [batch_size], device=clean_x.device)
            noise_scale = noise_level[t].cuda()
            noise_scale_sqrt = noise_scale**0.5
            noise = torch.randn_like(clean_x)
            noisy_x = noise_scale_sqrt[:,None,None] * clean_x + (1.0 - noise_scale[:,None,None])**0.5 * noise
            predicted = model(noisy_x, aug_spec, t)
            loss = nn.MSELoss()(noise, predicted[:,:,:noise.shape[2]])
            loss_meter.update(loss.item(), n=x.shape[0])
    
        samples = generate(model_params, model, aug_spec, torch.randn_like(x))
    
    return clean_x, aug_x, samples, loss_meter.avg


def generate(model_params, model, spectrogram, init_x, t_start=None, t_end=None):
    beta = np.linspace(model_params.noise_min, model_params.noise_max, model_params.noise_scales)
    alpha = 1 - beta
    alpha_t = np.cumprod(alpha)
    
    t_start = model_params.noise_scales - 1 if t_start is None else t_start
    t_end = 0 if t_end is None else t_end
    x = init_x
    assert t_start < model_params.noise_scales and t_end >= 0 and t_start >= t_end, f't_start and t_end must be in range [0, {model_params.noise_scales}) (t_start={t_start}, t_end={t_end})'

    for t in range(t_start, t_end - 1, -1):
        T = torch.tensor([t], dtype=torch.int64).repeat(spectrogram.shape[0]).to(spectrogram.device)
        c1 = 1/math.sqrt(alpha[t])
        c2 = beta[t]/math.sqrt(1 - alpha_t[t])
        x = c1*(x - c2*model(x, spectrogram, T))
        if t > 0:
            sigma = math.sqrt((1 - alpha_t[t-1]) * beta[t] / (1 - alpha_t[t]))
            x += sigma*torch.randn_like(x)
    return x

