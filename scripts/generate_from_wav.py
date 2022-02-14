import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
import argparse
import os
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import yaml

import data
import utils
import augmentation

from models.pix2pix import ResnetGenerator as Mel2MelModel
from models.diffwave_spectrogram import DiffWave as VocoderModel
from training_utils.finetune import generate as vocode_func


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', help='path to input sample')
    parser.add_argument('vocoder_checkpoint', help='path to vocoder model checkpoint')
    parser.add_argument('vocoder_params', help='path to vocoder model params')
    parser.add_argument('pix2pix_checkpoint', help='path to pix2pix model checkpoint')
    parser.add_argument('pix2pix_params', help='path to pix2pix model params')
    parser.add_argument('save_path', help='path to save enhanced sample at')
    parser.add_argument('--sample_rate', type=int, default=16000, help='sample rate (hertz)')
    parser.add_argument('--chunk_length', type=int, default=47360, help='number of samples to process at a time')
    parser.add_argument('--overlap', type=float, default=0.1, help='overlap fraction between adjacent chunks')
    parser.add_argument('--crossfade', default=False, action='store_true', help='cross fade adjacent chunks')
    parser.add_argument('--trimmed_save_path', default=None, help='path to save trimmed sample')
    args = parser.parse_args()
    return args


def load_model_params(params_file):
    with open(params_file, 'r') as f:
        model_params = EasyDict(yaml.load(f, yaml.Loader))
    return model_params


def generate(mel2mel_model, mel2mel_params, vocoder_model, vocoder_params, melspec_transform, aug_x):
    with torch.no_grad():
        aug_x_spec = melspec_transform(aug_x).log2()
        gen_x_spec = mel2mel_model(aug_x_spec)
        gen_x = vocode_func(vocoder_params, vocoder_model, gen_x_spec)
    return gen_x, gen_x_spec


def main(args):
    orig_sample = utils.load(args.sample_path, args.sample_rate)

    vocoder_params = load_model_params(args.vocoder_params)
    mel2mel_params = load_model_params(args.pix2pix_params)

    mel2mel_model = nn.DataParallel(Mel2MelModel(mel2mel_params).cuda())
    vocoder_model = nn.DataParallel(VocoderModel(vocoder_params).cuda())

    mel2mel_checkpoint = torch.load(args.pix2pix_checkpoint)
    vocoder_checkpoint = torch.load(args.vocoder_checkpoint)

    mel2mel_model.load_state_dict(mel2mel_checkpoint['generator_state_dict'])
    vocoder_model.load_state_dict(vocoder_checkpoint['model_state_dict'] if 'model_state_dict' in vocoder_checkpoint.keys() else vocoder_checkpoint['vocoder_state_dict'])

    melspec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=vocoder_params.sample_rate,   
                                                             n_fft=vocoder_params.n_fft,
                                                             hop_length=vocoder_params.hop_length,
                                                             n_mels=vocoder_params.n_mels).cuda()

    hop_length = int(args.chunk_length * (1 - args.overlap))
    overlap = args.chunk_length - hop_length
    cross_fade_in = torch.linspace(0, 1, overlap).cuda()
    cross_fade_out = 1 - cross_fade_in
    
    trim = ((orig_sample.shape[1] - args.chunk_length) % hop_length)
    sample = orig_sample[:,:-trim]
    sample = augmentation.perturb_silence(sample)
    enhanced_sample = torch.zeros_like(sample)
    chunks = [(i, i+args.chunk_length) for i in list(range(0, sample.shape[1], hop_length))]
    chunks = chunks if args.overlap == 0 else chunks[:-1]
    last_sample_gen = None
    for i, (start_idx, end_idx) in enumerate(tqdm(chunks)):
        sample_curr = sample[:, start_idx:end_idx].unsqueeze(0).cuda()
        sample_gen, _ = generate(mel2mel_model, mel2mel_params, vocoder_model, vocoder_params, melspec_transform, sample_curr)
        if not last_sample_gen is None and overlap > 0:
            # Normalize volume so that there isn't significant difference between adjacent chunks
            sample_gen_norm = torch.norm(sample_gen[:,:overlap], keepdim=True)
            last_sample_norm = torch.norm(last_sample_gen[:,-overlap:], keepdim=True)
            sample_gen = sample_gen/sample_gen_norm*last_sample_norm
        last_sample_gen = torch.clone(sample_gen)
        if args.crossfade:
            # Cross fade scaling
            sample_gen[:,:,:overlap] = sample_gen[:,:,:overlap]*cross_fade_in
            sample_gen[:,:,-overlap:] = sample_gen[:,:,-overlap:]*cross_fade_out
        sample_gen = sample_gen.squeeze(0).detach().cpu()
        enhanced_sample[:, start_idx:end_idx] += sample_gen
    
    enhanced_sample = enhanced_sample/enhanced_sample.abs().max()*0.95
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torchaudio.save(args.save_path, enhanced_sample, args.sample_rate)
    if args.trimmed_save_path:
        torchaudio.save(args.trimmed_save_path, sample, args.sample_rate)

if __name__ == '__main__':
    args = parse_args()
    main(args)
