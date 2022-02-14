import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

import numpy as np
import math
from tqdm import tqdm
import pdb

import utils
import augmentation

def train_epoch(global_step, 
                mel2mel_model_params, 
                mel2mel_models, 
                mel2mel_optimizers, 
                vocoder_model_params,
                vocoder_model,
                vocoder_optimizer,
                train_loaders, 
                summary_writer):

    data_loader, reverb_loader, noise_loader = train_loaders
    eq_model = augmentation.MicrophoneEQ(rate=vocoder_model_params.sample_rate).cuda()
    low_cut_filter = augmentation.LowCut(35, rate=vocoder_model_params.sample_rate).cuda()

    melspec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=vocoder_model_params.sample_rate,
                                                             n_fft=vocoder_model_params.n_fft, 
                                                             hop_length=vocoder_model_params.hop_length, 
                                                             n_mels=vocoder_model_params.n_mels).cuda()
    loss_meter = utils.AverageMeter('loss', fmt=':.4f')

    beta = np.linspace(vocoder_model_params.noise_min, vocoder_model_params.noise_max, vocoder_model_params.noise_scales)
    noise_level = np.cumprod(1 - beta)
    noise_level = torch.tensor(noise_level.astype(np.float32))
    
    generator, discriminator = mel2mel_models
    optim_generator, optim_discriminator = mel2mel_optimizers

    loop = tqdm(data_loader)
    for i, x_waveform in enumerate(loop):
        # Augment sample
        batch_size = x_waveform.shape[0]
        x_waveform, noise, rir = x_waveform.cuda(), next(noise_loader).cuda(), next(reverb_loader).cuda()
        x_waveform, x_aug_waveform = augmentation.augment(x_waveform, rir=rir, noise=noise, eq_model=eq_model, low_cut_model=low_cut_filter)
        x_waveform, x_aug_waveform = x_waveform[:,:,:(x_waveform.shape[2]//256)*256], x_aug_waveform[:,:,:(x_aug_waveform.shape[2]//256)*256]
        
        # Compute mel-spec and generate cleaned mel-spec
        x_aug_spec = melspec_transform(x_aug_waveform).log2()
        x_gen_spec = generator(x_aug_spec)
        
        # Perform diffwave vocoder training step using generated mel-spec
        t = torch.randint(0, vocoder_model_params.noise_scales, [batch_size], device=x_waveform.device)
        noise_scale = noise_level[t].cuda()
        noise_scale_sqrt = noise_scale**0.5
        noise = torch.randn_like(x_waveform)
        x_noisy_waveform = noise_scale_sqrt[:,None,None] * x_waveform + (1.0 - noise_scale[:,None,None])**0.5 * noise
        predicted = vocoder_model(x_noisy_waveform, x_gen_spec, t)
        loss = nn.MSELoss()(noise, predicted)
        loss.backward()
        vocoder_optimizer.step()
        optim_generator.step()
        vocoder_model.zero_grad()
        generator.zero_grad()
        loss_meter.update(loss.item(), n=batch_size)
        loop.set_postfix_str(str(loss_meter))
        summary_writer.add_scalar('training/loss', loss.item(), global_step)
        global_step += 1

    return global_step, (generator, discriminator), (optim_generator, optim_discriminator), vocoder_model, vocoder_optimizer


def validate(mel2mel_model_params, mel2mel_models, vocoder_model_params, vocoder_model, val_loaders):
    data_loader, reverb_loader, noise_loader = val_loaders
    eq_model = augmentation.MicrophoneEQ(rate=vocoder_model_params.sample_rate).cuda()
    low_cut_filter = augmentation.LowCut(35, rate=vocoder_model_params.sample_rate).cuda()

    melspec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=vocoder_model_params.sample_rate,
                                                             n_fft=vocoder_model_params.n_fft, 
                                                             hop_length=vocoder_model_params.hop_length, 
                                                             n_mels=vocoder_model_params.n_mels).cuda()
    loss_meter = utils.AverageMeter('loss', fmt=':.4f')
    generator, discriminator = mel2mel_models

    beta = np.linspace(vocoder_model_params.noise_min, vocoder_model_params.noise_max, vocoder_model_params.noise_scales)
    noise_level = np.cumprod(1 - beta)
    noise_level = torch.tensor(noise_level.astype(np.float32))
 
    with torch.no_grad():
        for x_waveform in data_loader: 

            # Augment sample
            batch_size = x_waveform.shape[0]
            x_waveform, noise, rir = x_waveform.cuda(), next(noise_loader).cuda(), next(reverb_loader).cuda()
            x_waveform, x_aug_waveform = augmentation.augment(x_waveform, rir=rir, noise=noise, eq_model=eq_model, low_cut_model=low_cut_filter)
            x_waveform, x_aug_waveform = x_waveform[:,:,:(x_waveform.shape[2]//256)*256], x_aug_waveform[:,:,:(x_aug_waveform.shape[2]//256)*256]
            
            # Compute mel-spec and generate cleaned mel-spec
            x_aug_spec = melspec_transform(x_aug_waveform).log2()
            x_gen_spec = generator(x_aug_spec)
            
            # Perform diffwave vocoder training step using generated mel-spec
            t = torch.randint(0, vocoder_model_params.noise_scales, [batch_size], device=x_waveform.device)
            noise_scale = noise_level[t].cuda()
            noise_scale_sqrt = noise_scale**0.5
            noise = torch.randn_like(x_waveform)
            x_noisy_waveform = noise_scale_sqrt[:,None,None] * x_waveform + (1.0 - noise_scale[:,None,None])**0.5 * noise
            predicted = vocoder_model(x_noisy_waveform, x_gen_spec, t)
            loss = nn.MSELoss()(noise, predicted)
            loss_meter.update(loss.item(), n=batch_size)
   
        x_gen_waveform = generate(vocoder_model_params, vocoder_model, x_gen_spec)
    
    return x_waveform, x_aug_waveform, 2**x_gen_spec, x_gen_waveform, {'loss': loss_meter.avg}


def generate(model_params, model, spectrogram):
    beta = np.linspace(model_params.noise_min, model_params.noise_max, model_params.noise_scales)
    alpha = 1 - beta
    alpha_t = np.cumprod(alpha)
    x = torch.randn(spectrogram.shape[0], 1, 47360).to(spectrogram.device)
    for t in range(model_params.noise_scales - 1, -1, -1):
        T = torch.tensor([t], dtype=torch.int64).repeat(spectrogram.shape[0]).to(spectrogram.device)
        c1 = 1/math.sqrt(alpha[t])
        c2 = beta[t]/math.sqrt(1 - alpha_t[t])
        x = c1*(x - c2*model(x, spectrogram, T))
        if t > 0:
            sigma = math.sqrt((1 - alpha_t[t-1]) * beta[t] / (1 - alpha_t[t]))
            x += sigma*torch.randn_like(x)
    return x

