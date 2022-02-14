import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
import argparse
import os
import pdb

from torch.utils.data import DataLoader
import torchaudio
import numpy as np
from tqdm import tqdm

import data
import utils
import augmentation

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path', help='path to medley db solos')
    parser.add_argument('rir_path', help='path to reverb dataset')
    parser.add_argument('noise_path', help='path to noise dataset')
    parser.add_argument('save_dir', help='directory to save static dataset to')
    parser.add_argument('--sample_rate', type=int, default=16000, help='sample rate (hertz)')
    parser.add_argument('--instruments', nargs='+', default=None, help='instruments to use from medleydb')
    parser.add_argument('--split', default='validation', choices=['training', 'validation', 'test'], help='dataset split to use')
    parser.add_argument('--snr_levels', nargs='+', type=int, default=[5, 15, 25])
    parser.add_argument('--drr_levels', nargs='+', type=int, default=[0,3,6])
    args = parser.parse_args()
    return args


def main(args):
    test_dataset = data.MedleyDBSolosDataset(args.dataset_path, instruments=args.instruments, split=args.split, rate=args.sample_rate) 
    test_loader = DataLoader(test_dataset, batch_size=1, drop_last=True, shuffle=False)

    test_reverb = data.ReverbDataset(args.rir_path, split=args.split)
    test_reverb_loader = iter(DataLoader(test_reverb, batch_size=1))

    test_noise = data.NoiseDataset(args.noise_path, split=args.split)
    test_noise_loader = iter(DataLoader(test_noise, batch_size=1))

    eq_model = augmentation.MicrophoneEQ(rate=args.sample_rate).cuda()
    low_cut_filter = augmentation.LowCut(35, rate=args.sample_rate).cuda()
   
    os.makedirs(os.path.join(args.save_dir, 'clean', args.split), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'random_eq', args.split), exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'random_augmentation', args.split), exist_ok=True)
    for snr in args.snr_levels:
        os.makedirs(os.path.join(args.save_dir, f'snr_{snr}', args.split), exist_ok=True)
    for drr in args.drr_levels:
        os.makedirs(os.path.join(args.save_dir, f'drr_{drr}', args.split), exist_ok=True)

    for i, x in enumerate(tqdm(test_loader)):
        x = x.cuda()
        r = next(test_reverb_loader).cuda()
        n = next(test_noise_loader).cuda()

        x = augmentation.perturb_silence(x)
        torchaudio.save(os.path.join(args.save_dir, 'clean', args.split, f'sample_{i}.wav'), x[0].detach().cpu(), args.sample_rate)
        
        for snr in args.snr_levels:
            sample = augmentation.normalize(low_cut_filter(augmentation.apply_noise(x, n, nsr_target=-snr)))
            torchaudio.save(os.path.join(args.save_dir, f'snr_{snr}', args.split, f'sample_{i}.wav'), sample[0].detach().cpu(), args.sample_rate)

        for drr in args.drr_levels:
            sample = augmentation.normalize(low_cut_filter(augmentation.apply_reverb(x, r, drr, rate=args.sample_rate)))
            torchaudio.save(os.path.join(args.save_dir, f'drr_{drr}', args.split, f'sample_{i}.wav'), sample[0].detach().cpu(), args.sample_rate)
        
        # Generate sample with random EQ
        sample = augmentation.normalize(low_cut_filter(eq_model(x)))
        torchaudio.save(os.path.join(args.save_dir, 'random_eq', args.split, f'sample_{i}.wav'), sample[0].detach().cpu(), args.sample_rate)

        # Generate sample with random augmentation
        _, sample = augmentation.augment(x, rir=r, noise=n, eq_model=eq_model, low_cut_model=low_cut_filter)
        torchaudio.save(os.path.join(args.save_dir, 'random_augmentation', args.split, f'sample_{i}.wav'), sample[0].detach().cpu(), args.sample_rate)

if __name__ == '__main__':
    args = parse_args()
    main(args)
