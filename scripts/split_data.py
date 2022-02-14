import torch
import numpy as np

import os
import argparse
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')

import data

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_type', choices=['reverb', 'noise'], help='reverb or noise data')
    parser.add_argument('data_dirs', nargs='+', help='path to data directories')
    parser.add_argument('output_dir', help='path to output directory')
    parser.add_argument('--rate', type=int, default=22050, help='sample rate')
    parser.add_argument('--noise_sample_length', type=int, default=47555, help='length to cut noise samples to')
    parser.add_argument('--validation_fraction', type=float, default=0.1, help='fraction of data to reserve for validation')
    parser.add_argument('--test_fraction', type=float, default=0.1, help='fraction of data to reserve for testing')
    args = parser.parse_args()
    return args

def main(args):
    if args.data_type == 'reverb':
        dataset = data.ReverbDataset(args.data_dirs, rate=args.rate)
        out_file = f'reverb_samples_{args.rate}_hz.npz'
    elif args.data_type == 'noise':
        dataset = data.NoiseDataset(args.data_dirs, rate=args.rate, sample_length=args.noise_sample_length)
        out_file = f'noise_samples_{args.noise_sample_length}_length_{args.rate}_hz.npz'
    else:
        logging.error(f'Unknown data type {args.data_type}')
    
    # Drop samples of all zeros (not sure why these are in the data)
    dataset.data = dataset.data[torch.where(~torch.all(dataset.data == 0, dim=2))].unsqueeze(1)
    
    # Shuffle and split
    num_samples = dataset.data.shape[0]
    dataset.data = dataset.data[torch.randperm(num_samples)]
    splits = [int((1-args.validation_fraction-args.test_fraction)*num_samples), int(args.test_fraction*num_samples)]
    splits.append(num_samples - splits[0] - splits[1])

    train, valid, test = torch.split(dataset.data, splits)
    train, valid, test = train.numpy(), valid.numpy(), test.numpy()
    np.savez(os.path.join(args.output_dir, out_file), training=train, validation=valid, test=test)

if __name__ == '__main__':
    args = parse_args()
    main(args)
