import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
import argparse
import os
import pdb

import torch
import torch.nn as nn
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
from training_utils.finetune import generate as diffwave_vocode_func


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_dataset_dir', help='path to eval dataset directory')
    parser.add_argument('augmentation_names', nargs='+', help='names of augmentation directories')
    parser.add_argument('vocoder_checkpoint', help='path to vocoder model checkpoint')
    parser.add_argument('vocoder_params', help='path to vocoder model params')
    parser.add_argument('pix2pix_checkpoint', help='path to pix2pix model checkpoint')
    parser.add_argument('pix2pix_params', help='path to pix2pix model params')
    parser.add_argument('save_dir', help='directory to save samples to')
    parser.add_argument('--sample_rate', default=16000, help='sample rate (hertz)')
    parser.add_argument('--split', default='validation', choices=['training', 'validation', 'test'], help='dataset split to use')
    parser.add_argument('--batch_size', default=1, help='batch size')
    parser.add_argument('--vocoder_only', default=False, action='store_true', help='only evaluate vocoder')
    parser.add_argument('--griffin_lim_vocoder', default=False, action='store_true', help='use inverse mel scaling + griffin lim as vocoder')
    args = parser.parse_args()
    return args


def load_model_params(params_file):
    with open(params_file, 'r') as f:
        model_params = EasyDict(yaml.load(f, yaml.Loader))
    return model_params


def generate(mel2mel_model, mel2mel_params, vocode_func, melspec_transform, aug_x):
    with torch.no_grad():
        aug_x = aug_x[:,:,:(aug_x.shape[2]//256)*256]
        aug_x_spec = melspec_transform(aug_x).log2()
        gen_x_spec = mel2mel_model(aug_x_spec)
    gen_x = vocode_func(gen_x_spec)
    return gen_x


def vocode_diffwave(vocoder_model, vocoder_params, log_melspec_x):
    with torch.no_grad():
        gen_x = diffwave_vocode_func(vocoder_params, vocoder_model, log_melspec_x)
    return gen_x


def vocode_griffin_lim(inverse_mel, griffin_lim, log_melspec_x):
    spec_x = inverse_mel(2**log_melspec_x)
    gen_x = griffin_lim(spec_x)
    return gen_x


def main(args):
    vocoder_params = load_model_params(args.vocoder_params)
    melspec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=vocoder_params.sample_rate,   
                                                             n_fft=vocoder_params.n_fft,
                                                             hop_length=vocoder_params.hop_length,
                                                             n_mels=vocoder_params.n_mels).cuda()

    if args.griffin_lim_vocoder:
        inverse_mel = torchaudio.transforms.InverseMelScale(n_stft=vocoder_params.n_fft // 2 + 1,
                                                            n_mels=vocoder_params.n_mels,
                                                            sample_rate=vocoder_params.sample_rate,
                                                            max_iter=1000).cuda()
        griffin_lim = torchaudio.transforms.GriffinLim(n_fft=vocoder_params.n_fft,
                                                       hop_length=vocoder_params.hop_length,
                                                       n_iter=1000).cuda()
        vocode = lambda x: vocode_griffin_lim(inverse_mel, griffin_lim, x)
    else:
        vocoder_model = nn.DataParallel(VocoderModel(vocoder_params).cuda())
        vocoder_checkpoint = torch.load(args.vocoder_checkpoint)
        vocoder_model.load_state_dict(vocoder_checkpoint['model_state_dict'] if 'model_state_dict' in vocoder_checkpoint.keys() else vocoder_checkpoint['vocoder_state_dict'])
        vocode = lambda x: vocode_diffwave(vocoder_model, vocoder_params, x)

    if not args.vocoder_only:
        mel2mel_params = load_model_params(args.pix2pix_params)
        mel2mel_model = nn.DataParallel(Mel2MelModel(mel2mel_params).cuda())
        mel2mel_checkpoint = torch.load(args.pix2pix_checkpoint)
        mel2mel_model.load_state_dict(mel2mel_checkpoint['generator_state_dict'])

    for augmentation_name in args.augmentation_names:
        logging.info(f'Evaluating on augmentation {augmentation_name}')
        save_dir = os.path.join(args.save_dir, augmentation_name, args.split)
        os.makedirs(save_dir, exist_ok=True)

        num_samples = len(os.listdir(os.path.join(args.eval_dataset_dir, augmentation_name, args.split)))
        for i in tqdm(list(range(num_samples))):
            sample = utils.load(os.path.join(args.eval_dataset_dir, augmentation_name, args.split, f'sample_{i}.wav'), args.sample_rate).cuda().unsqueeze(0)
            if args.vocoder_only:
                log_melspec_sample = melspec_transform(sample).log2()
                gen_sample = vocode(log_melspec_sample)
            else:
                gen_sample = generate(mel2mel_model, mel2mel_params, vocode, melspec_transform, sample)
            torchaudio.save(os.path.join(save_dir, f'sample_{i}.wav'), gen_sample[0].detach().cpu(), args.sample_rate)


if __name__ == '__main__':
    args = parse_args()
    main(args)
