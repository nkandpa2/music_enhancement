import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import numpy as np
from tqdm import tqdm
import math
import pdb

import utils
import augmentation

def train_epoch(global_step, model_params, model, optimizer, train_loaders, summary_writer):
    data_loader, reverb_loader, noise_loader = train_loaders

    eq_model = augmentation.MicrophoneEQ(rate=model_params.sample_rate).cuda()
    loss_meter = utils.AverageMeter('loss', fmt=':.4f')

    beta = np.linspace(model_params.noise_min, model_params.noise_max, model_params.noise_scales)
    noise_level = np.cumprod(1 - beta)
    noise_level = torch.tensor(noise_level.astype(np.float32))

    loop = tqdm(data_loader)
    for i, x in enumerate(loop):
        batch_size = x.shape[0]
        x, noise, rir = x.cuda(), next(noise_loader).cuda(), next(reverb_loader).cuda()
        clean_x, aug_x = augmentation.augment(x, rir=rir, noise=noise, eq_model=eq_model)
        t = torch.randint(0, model_params.noise_scales, [batch_size], device=clean_x.device)
        noise_scale = noise_level[t].cuda()
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(clean_x)
        noisy_x = noise_scale_sqrt[:,None,None] * clean_x + (1.0 - noise_scale[:,None,None])**0.5 * noise
        predicted = model(noisy_x, aug_x, t)
        loss = nn.L1Loss()(noise, predicted)
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
    loss_meter = utils.AverageMeter('loss', fmt=':.4f')

    beta = np.linspace(model_params.noise_min, model_params.noise_max, model_params.noise_scales)
    alpha = 1 - beta
    alpha_t = torch.tensor(np.cumprod(1 - beta).astype(np.float32))
    
    with torch.no_grad():
        for x in data_loader: 
            batch_size = x.shape[0]
            x, noise, rir = x.cuda(), next(noise_loader).cuda(), next(reverb_loader).cuda()
            clean_x, aug_x = augmentation.augment(x, rir=rir, noise=noise, eq_model=eq_model)
            t = torch.randint(0, model_params.noise_scales, [batch_size], device=clean_x.device)
            noise_scale = alpha_t[t].cuda()
            noise = torch.randn_like(clean_x)
            noisy_x = (noise_scale[:,None,None])**0.5 * clean_x + (1.0 - noise_scale[:,None,None])**0.5 * noise
            predicted = model(noisy_x, aug_x, t)
            loss = nn.L1Loss()(noise, predicted)
            loss_meter.update(loss.item(), n=x.shape[0])
    
        samples = generate(model_params, model, aug_x)
    
    return clean_x, aug_x, samples, loss_meter.avg


def generate(model_params, model, cond_waveform):
    beta = np.linspace(model_params.noise_min, model_params.noise_max, model_params.noise_scales)
    alpha = 1 - beta
    alpha_t = np.cumprod(alpha)
    x = torch.randn_like(cond_waveform)
    for t in range(model_params.noise_scales - 1, -1, -1):
        T = torch.tensor([t], dtype=torch.int64).repeat(cond_waveform.shape[0]).to(cond_waveform.device)
        c1 = 1/math.sqrt(alpha[t])
        c2 = beta[t]/math.sqrt(1 - alpha_t[t])
        x = c1*(x - c2*model(x, cond_waveform, T))
        if t > 0:
            sigma = math.sqrt((1 - alpha_t[t-1]) * beta[t] / (1 - alpha_t[t]))
            x += sigma*torch.randn_like(x)
    return x
