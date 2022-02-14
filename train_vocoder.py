import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchaudio

from tqdm import tqdm
from easydict import EasyDict
import argparse
import yaml
import os
import pdb

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')

import data
import utils
from models.diffwave_spectrogram import DiffWave as DiffWaveSpec
from models.diffwave_waveform import DiffWave as DiffWaveWaveform

from training_utils.diffwave_spectrogram import train_epoch as diffwave_spec_train_epoch, validate as diffwave_spec_validate
from training_utils.diffwave_waveform import train_epoch as diffwave_waveform_train_epoch, validate as diffwave_waveform_validate
from training_utils.diffwave_vocoder import train_epoch as diffwave_vocoder_train_epoch, validate as diffwave_vocoder_validate


MODELS = {'diffwave_spectrogram': DiffWaveSpec, 
          'diffwave_waveform': DiffWaveWaveform,
          'diffwave_vocoder': DiffWaveSpec}
TRAIN_FUNCS = {'diffwave_spectrogram': diffwave_spec_train_epoch,
               'diffwave_waveform': diffwave_waveform_train_epoch,
               'diffwave_vocoder': diffwave_vocoder_train_epoch}
VALIDATE_FUNCS = {'diffwave_spectrogram': diffwave_spec_validate,
                  'diffwave_waveform': diffwave_waveform_validate,
                  'diffwave_vocoder': diffwave_vocoder_validate}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['diffwave_spectrogram', 'diffwave_waveform', 'diffwave_vocoder'])
    parser.add_argument('model_params_file', help='path to model params yaml file')
    parser.add_argument('train_dir', help='path to training directory')
    parser.add_argument('--checkpoint', default=None, help='path to checkpoint')
    parser.add_argument('--epochs', type=int, default=500, help='number of training steps')
    parser.add_argument('--batch_size', type=int, default=8, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--instruments', nargs='+', default=None, help='instruments to train on')
    parser.add_argument('--sample_rate', type=int, default=16000, help='audio sample rate')
    parser.add_argument('--dataset_path', default=None, help='path to medleydb solos data dir')
    parser.add_argument('--rir_path', default=None, help='path to reverb samples .npz file')
    parser.add_argument('--noise_path', default=None, help='path to noise samples .npz file')
    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.train_dir):
        os.makedirs(args.train_dir)
        os.makedirs(os.path.join(args.train_dir, 'tb'))
        os.makedirs(os.path.join(args.train_dir, 'samples'))
        os.makedirs(os.path.join(args.train_dir, 'checkpoints'))

    summary_writer = SummaryWriter(os.path.join(args.train_dir, 'tb'))
    train_loaders, val_loaders, test_loaders = init_dataloaders(args)
    model_params = load_model_params(args)
    model, optimizer = init_model(model_params, args)
    train_epoch = TRAIN_FUNCS[args.model]
    validate = VALIDATE_FUNCS[args.model]
    train(args,
          model_params,
          model,
          optimizer,
          train_epoch,
          validate,
          train_loaders,
          val_loaders,
          summary_writer)


def load_model_params(args):
    with open(args.model_params_file, 'r') as f:
        model_params = EasyDict(yaml.load(f, yaml.Loader))
    return model_params


def init_model(model_params, args):
    model = nn.DataParallel(MODELS[args.model](model_params).cuda())
    optimizer = torch.optim.Adam(model.parameters(), args.lr)
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer


def init_dataloaders(args):
    train_dataset = data.MedleyDBSolosDataset(args.dataset_path, instruments=args.instruments, split='training', rate=args.sample_rate) 
    val_dataset = data.MedleyDBSolosDataset(args.dataset_path, instruments=args.instruments, split='validation', rate=args.sample_rate) 
    test_dataset = data.MedleyDBSolosDataset(args.dataset_path, instruments=args.instruments, split='test', rate=args.sample_rate) 

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, drop_last=True, shuffle=False)
    
    if args.rir_path:
        train_reverb = data.ReverbDataset(args.rir_path, split='training')
        val_reverb = data.ReverbDataset(args.rir_path, split='validation')
        test_reverb = data.ReverbDataset(args.rir_path, split='test')
        train_reverb_loader = iter(DataLoader(train_reverb, batch_size=args.batch_size))
        val_reverb_loader = iter(DataLoader(val_reverb, batch_size=args.batch_size))
        test_reverb_loader = iter(DataLoader(test_reverb, batch_size=args.batch_size))
    else:
        train_reverb_loader = val_reverb_loader = test_reverb_loader = None
    
    if args.noise_path:
        train_noise = data.NoiseDataset(args.noise_path, split='training')
        val_noise = data.NoiseDataset(args.noise_path, split='validation')
        test_noise = data.NoiseDataset(args.noise_path, split='test')
        train_noise_loader = iter(DataLoader(train_noise, batch_size=args.batch_size))
        val_noise_loader = iter(DataLoader(val_noise, batch_size=args.batch_size))
        test_noise_loader = iter(DataLoader(test_noise, batch_size=args.batch_size))
    else:
        train_noise_loader = val_noise_loader = test_noise_loader = None

    return (train_loader, train_reverb_loader, train_noise_loader), \
           (val_loader, val_reverb_loader, val_noise_loader), \
           (test_loader, test_reverb_loader, test_noise_loader)


def save_validation_to_tb(ground_truth, augmented_samples, samples, loss, sample_rate, epoch, summary_writer):
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=128).cuda()
    summary_writer.add_scalar('validation/loss', loss, epoch)
    summary_writer.add_audio('sample', samples[0], epoch, sample_rate=sample_rate)
    summary_writer.add_audio('augmented_sample', augmented_samples[0], epoch, sample_rate=sample_rate)
    summary_writer.add_audio('ground_truth', ground_truth[0], epoch, sample_rate=sample_rate)
    summary_writer.add_figure('waveform_sample', utils.plot_waveform(samples[0,0]), epoch)
    summary_writer.add_figure('waveform_augmented_sample', utils.plot_waveform(augmented_samples[0,0]), epoch)
    summary_writer.add_figure('waveform_ground_truth', utils.plot_waveform(ground_truth[0,0]), epoch)
    summary_writer.add_figure('spectrogram_sample', utils.plot_spectrogram(mel_spec_transform(samples)[0,0]), epoch)
    summary_writer.add_figure('spectrogram_augmented_sample', utils.plot_spectrogram(mel_spec_transform(augmented_samples)[0,0]), epoch)
    summary_writer.add_figure('spectrogram_ground_truth', utils.plot_spectrogram(mel_spec_transform(ground_truth)[0,0]), epoch)
    

def save_samples(sample_dict, sample_rate, epoch, train_dir):
    for key, sample in sample_dict.items():
        for i in range(sample.shape[0]):
            torchaudio.save(os.path.join(train_dir, 'samples', f'epoch_{epoch}_sample_{i}_{key}.wav'), sample[i].detach().cpu(), sample_rate)


def save_checkpoint(model, optimizer, epoch, train_dir):
    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()}, os.path.join(train_dir, 'checkpoints', f'checkpoint_{epoch}.pt'))


def train(args, 
          model_params,
          model, 
          optimizer, 
          train_epoch,
          validate,
          train_loaders,
          val_loaders,
          summary_writer):
    
    global_step = 0 
    for epoch in range(1, args.epochs+1):
        logging.info(f'Training epoch {epoch}')
        global_step, model, optimizer = train_epoch(global_step, model_params, model, optimizer, train_loaders, summary_writer)

        logging.info(f'Validating epoch {epoch}')
        ground_truth, augmented_samples, samples, loss = validate(model_params, model, val_loaders)

        # Save tensorboard data, generated samples, and model checkpoint
        save_validation_to_tb(ground_truth, augmented_samples, samples, loss, args.sample_rate, epoch, summary_writer)
        save_samples({'clean': ground_truth, 'augmented': augmented_samples, 'generated': samples}, args.sample_rate, epoch, args.train_dir)
        save_checkpoint(model, optimizer, epoch, args.train_dir)
  

def test(model_params, model, test_data):
    pass


if __name__ == '__main__':
    args = parse_args()
    main(args)
