import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
import argparse
import os
import pdb

import torch
import torch.nn as nn
import torchaudio
import pandas as pd
import numpy as np
from tqdm import tqdm
from pysepm import fwSNRseg

import utils

def trim_samples(clean, estimated):
    min_length = min(clean.shape[2], estimated.shape[2])
    logging.debug(f'Trimming samples to length {min_length}')
    return clean[:,:,:min_length], estimated[:,:,:min_length]

def normalize(clean, estimated):
    clean_shape, estimated_shape = clean.shape, estimated.shape
    clean, estimated = clean.reshape(clean_shape[0], -1), estimated.reshape(estimated_shape[0], -1)
    clean = clean/torch.norm(clean, dim=1, keepdim=True)
    estimated = estimated/torch.norm(estimated, dim=1, keepdim=True)
    clean, estimated = clean.reshape(*clean_shape), estimated.reshape(*estimated_shape)
    return clean, estimated

def fwssnr(clean, estimated, batch_size=100):
    clean, estimated = trim_samples(clean, estimated)
    clean, estimated = normalize(clean, estimated)
    fwSNRseg_vectorized = np.vectorize(fwSNRseg, signature='(n),(n),()->()')
    values = []
    for i in range(0, clean.shape[0], batch_size):
        clean_batch = clean[i:i+batch_size, 0, :].detach().cpu().numpy()
        estimated_batch = estimated[i:i+batch_size, 0, :].detach().cpu().numpy()
        batch_values = fwSNRseg_vectorized(clean_batch, estimated_batch, 16000)
        values.extend(batch_values.tolist())
    return np.mean(values)

def spectrogram_l1(clean, estimated, batch_size=100):
    clean, estimated = trim_samples(clean, estimated)
    clean, estimated = normalize(clean, estimated)
    melspec_transform = torchaudio.transforms.Spectrogram(n_fft=1024, hop_length=256)
    values = []
    for i in range(0, clean.shape[0], batch_size):
        clean_batch = clean[i:i+batch_size]
        estimated_batch = estimated[i:i+batch_size]
        clean_spec = (melspec_transform(clean_batch) + 1e-10).log()
        estimated_spec = (melspec_transform(estimated_batch) + 1e-10).log()
        clean_spec, estimated_spec = clean_spec.reshape(clean_spec.shape[0], -1), estimated_spec.reshape(estimated_spec.shape[0], -1)
        values.extend((clean_spec - estimated_spec).abs().mean(dim=1).detach().cpu().numpy().tolist())
    return np.mean(values)

def multi_resolution_spectrogram(clean, estimated, batch_size=100):
    clean, estimated = trim_samples(clean, estimated)
    clean, estimated = normalize(clean, estimated)
    n_ffts = [512, 1024, 2048]
    hop_lengths = [128, 256, 512]
    spec_transforms = [torchaudio.transforms.Spectrogram(n_fft=n_fft, hop_length=hop_length, power=1) for n_fft, hop_length in zip(n_ffts, hop_lengths)]
    values = []
    for i in range(0, clean.shape[0], batch_size):
        clean_batch = clean[i:i+batch_size]
        estimated_batch = estimated[i:i+batch_size]
        clean_specs = [(spec_transform(clean_batch) + 1e-10).reshape(clean_batch.shape[0], -1)  for spec_transform in spec_transforms]
        estimated_specs = [(spec_transform(estimated_batch) + 1e-10).reshape(estimated_batch.shape[0], -1) for spec_transform in spec_transforms]
    
        losses_sc = [(torch.square(clean_spec - estimated_spec).sum(1) / torch.square(clean_spec).sum(1)) for clean_spec, estimated_spec in zip(clean_specs, estimated_specs)]
        losses_mag = [(clean_spec.log() - estimated_spec.log()).abs().mean(dim=1) for clean_spec, estimated_spec in zip(clean_specs, estimated_specs)]
        losses = [(loss_sc + loss_mag).detach().cpu().numpy() for loss_sc, loss_mag in zip(losses_sc, losses_mag)]
        loss_batch = np.mean(losses, axis=0)
        values.extend(loss_batch.tolist())
    return np.mean(values)

METRICS = {'spec_snr': fwssnr,
           'multi_resolution_spectrogram': multi_resolution_spectrogram,
           'spectrogram_l1': spectrogram_l1}

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('eval_dir', help='path to directory containing samples to evaluate (should contain subdirectories for clean, augmented, and each type of enhancement)')
    parser.add_argument('output_csv_file', help='path to output_file')
    parser.add_argument('--enhancement_names', nargs='+', required=True, help='list names of the enhancement directories eval_dir (should contain subdirectories for each augmentation type)')
    parser.add_argument('--augmentation_names', nargs='+', required=True, help='list of augmentation names to evaluate enhancement for')
    parser.add_argument('--metrics', nargs='+', default=['spec_snr', 'spectrogram_l1', 'multi_resolution_spectrogram'], help='list of evaluation metrics')
    parser.add_argument('--splits', nargs='+', default=['validation', 'test'], choices=['validation', 'test'], help='list of dataset splits to evaluate on')
    parser.add_argument('--batch_size', type=int, default=100, help='batch size to evaluate at a time')
    parser.add_argument('--n_eval', type=int, default=-1, help='number of samples to evaluate on (randomly selected). Default: -1 indicates whole evaluation dataset')
    args = parser.parse_args()
    return args

def main(args):
    results_df = pd.DataFrame(index=pd.MultiIndex.from_product([args.splits, args.augmentation_names, args.enhancement_names + ['no_enhancement']], names=['dataset', 'augmentation', 'model']), columns=args.metrics)

    num_samples = {split: len(os.listdir(os.path.join(args.eval_dir, 'clean', split))) for split in args.splits}
    if args.n_eval == -1:
        sample_idxs = {split: list(range(num_samples[split])) for split in args.splits}
    else:
        sample_idxs = {split: np.random.choice(num_samples[split], args.n_eval, replace=False) for split in args.splits}
    print(sample_idxs)

    clean_samples = {split: [] for split in args.splits}
    for split in args.splits:
        logging.info(f'Loading clean samples from {split} dataset')
        split_dir = os.path.join(args.eval_dir, 'clean', split)
        for i in tqdm(sample_idxs[split]):
            clean_samples[split].append(utils.load(os.path.join(split_dir, f'sample_{i}.wav'), 16000))
        clean_samples[split] = torch.stack(clean_samples[split], dim=0)
    
    for augmentation_name in args.augmentation_names:
        for enhancement_name in args.enhancement_names:
            for split in args.splits:
                logging.info(f'Loading {enhancement_name} enhancement on {split} samples with {augmentation_name} augmentation')
                samples = []
                sample_dir = os.path.join(args.eval_dir, enhancement_name, augmentation_name, split)
                for i in tqdm(sample_idxs[split]):
                    samples.append(utils.load(os.path.join(sample_dir, f'sample_{i}.wav'), 16000)) 
                samples = torch.stack(samples, dim=0)
                
                for metric in args.metrics:
                    logging.info(f'Evaluating metric {metric}')
                    metric_func = METRICS[metric]
                    results_df.loc[(split, augmentation_name, enhancement_name), metric] = metric_func(clean_samples[split], samples, batch_size=args.batch_size)
        
        for split in args.splits:
            logging.info(f'Loading augmented data on {split} samples')
            samples = []
            sample_dir = os.path.join(args.eval_dir, augmentation_name, split)
            num_samples = len(os.listdir(sample_dir))
            for i in tqdm(sample_idxs[split]):
                samples.append(utils.load(os.path.join(sample_dir, f'sample_{i}.wav'), 16000))
            samples = torch.stack(samples, dim=0)

            for metric in args.metrics:
                logging.info(f'Evaluating metric {metric}')
                metric_func = METRICS[metric]
                results_df.loc[(split, augmentation_name, 'no_enhancement'), metric] = metric_func(clean_samples[split], samples)

    logging.info(f'Saving results to {args.output_csv_file}')
    results_df.to_csv(args.output_csv_file)
    logging.info('Eval Results:')
    print(results_df)

if __name__ == '__main__':
    args = parse_args()
    main(args)
