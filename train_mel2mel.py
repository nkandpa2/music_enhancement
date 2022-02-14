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
from models.pix2pix import ResnetGenerator as Generator, NLayerDiscriminator as Discriminator
from models.diffwave_spectrogram import DiffWave as DiffWaveSpec

from training_utils.pix2pix import train_epoch as pix2pix_train_epoch, validate as pix2pix_validate
from training_utils.diffwave_spectrogram import generate as vocoder_generate


MODELS = {'pix2pix': (Generator, Discriminator)}
TRAIN_FUNCS = {'pix2pix': pix2pix_train_epoch} 
VALIDATE_FUNCS = {'pix2pix': pix2pix_validate}

VOCODER_MODELS = {'diffwave_vocoder': DiffWaveSpec}
VOCODE_FUNCS= {'diffwave_vocoder': vocoder_generate}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['pix2pix'])
    parser.add_argument('model_params_file', help='path to model params yaml file')
    parser.add_argument('train_dir', help='path to training directory')
    parser.add_argument('--vocoder_model', default='diffwave_vocoder', choices=['diffwave_vocoder'])
    parser.add_argument('--vocoder_model_params_file', default='params/diffwave_vocoder_16k.yml')
    parser.add_argument('--vocoder_model_checkpoint', default='runs/diffwave_vocoder_long2/checkpoints/checkpoint_4000.pt')
    parser.add_argument('--epochs', type=int, default=200, help='number of training steps')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    parser.add_argument('--lr', type=float, default=2e-4, help='learning rate')
    parser.add_argument('--instruments', nargs='+', default=None, help='instruments to train on')
    parser.add_argument('--sample_rate', type=int, default=16000, help='audio sample rate')
    parser.add_argument('--dataset_path', default='/home/code-base/scratch_space/data/medleydb_solos', help='path to medleydb solos data dir')
    parser.add_argument('--rir_path', default='/home/code-base/scratch_space/data/audio-awesomizer-production/batch_data/reverb_samples_22050_hz.npz', 
        help='path to reverb samples .npz file')
    parser.add_argument('--noise_path', default='/home/code-base/scratch_space/data/audio-awesomizer-production/batch_data/noise_samples_65536_length_22050_hz.npz', 
        help='path to noise samples .npz file')
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
    vocoder_model_params = load_model_params(args, vocoder=True)
    models, optimizers = init_model(model_params, args)
    vocoder_model, _ = init_model(vocoder_model_params, args, vocoder=True)
    train_epoch = TRAIN_FUNCS[args.model]
    validate = VALIDATE_FUNCS[args.model]
    vocode = VOCODE_FUNCS[args.vocoder_model]
    train(args,
          model_params,
          models,
          optimizers,
          train_epoch,
          validate,
          train_loaders,
          val_loaders,
          vocoder_model_params,
          vocoder_model,
          vocode,
          summary_writer)


def load_model_params(args, vocoder=False):
    model_params_file = args.vocoder_model_params_file if vocoder else args.model_params_file
    with open(model_params_file, 'r') as f:
        model_params = EasyDict(yaml.load(f, yaml.Loader))
    return model_params


def init_model(model_params, args, vocoder=False):
    if vocoder:
        model = nn.DataParallel(VOCODER_MODELS[args.vocoder_model](model_params).cuda())
        state_dict = torch.load(args.vocoder_model_checkpoint)
        model.load_state_dict(state_dict['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters(), args.lr)
        return model, optimizer
    else:
        generator, discriminator = MODELS[args.model]
        generator, discriminator = nn.DataParallel(generator(model_params).cuda()), nn.DataParallel(discriminator(model_params).cuda())
        optim_generator, optim_discriminator = torch.optim.Adam(generator.parameters(), args.lr, betas=(0.5, 0.999)), \
                                               torch.optim.Adam(discriminator.parameters(), args.lr, betas=(0.5, 0.999))
        models = (generator, discriminator)
        optimizers = (optim_generator, optim_discriminator)
        return models, optimizers


def init_dataloaders(args):
    train_dataset = data.MedleyDBSolosDataset(args.dataset_path, instruments=args.instruments, split='training', rate=args.sample_rate) 
    val_dataset = data.MedleyDBSolosDataset(args.dataset_path, instruments=args.instruments, split='validation', rate=args.sample_rate) 
    test_dataset = data.MedleyDBSolosDataset(args.dataset_path, instruments=args.instruments, split='test', rate=args.sample_rate) 

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, drop_last=True, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, drop_last=True, shuffle=False)
    
    train_reverb = data.ReverbDataset(args.rir_path, split='training')
    val_reverb = data.ReverbDataset(args.rir_path, split='validation')
    test_reverb = data.ReverbDataset(args.rir_path, split='test')
    train_reverb_loader = iter(DataLoader(train_reverb, batch_size=args.batch_size))
    val_reverb_loader = iter(DataLoader(val_reverb, batch_size=8))
    test_reverb_loader = iter(DataLoader(test_reverb, batch_size=8))

    train_noise = data.NoiseDataset(args.noise_path, split='training')
    val_noise = data.NoiseDataset(args.noise_path, split='validation')
    test_noise = data.NoiseDataset(args.noise_path, split='test')
    train_noise_loader = iter(DataLoader(train_noise, batch_size=args.batch_size))
    val_noise_loader = iter(DataLoader(val_noise, batch_size=8))
    test_noise_loader = iter(DataLoader(test_noise, batch_size=8))

    return (train_loader, train_reverb_loader, train_noise_loader), \
           (val_loader, val_reverb_loader, val_noise_loader), \
           (test_loader, test_reverb_loader, test_noise_loader)


def save_validation_to_tb(ground_truth, augmented_samples, generated_spec, vocoded_samples, griffin_lim_samples, loss_g, loss_d, loss_l1, sample_rate, epoch, summary_writer):
    mel_spec_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_fft=1024, hop_length=256, n_mels=128).cuda()
    summary_writer.add_scalar('validation/l1_loss', loss_l1, epoch)
    summary_writer.add_scalar('validation/generator_loss', loss_g, epoch)
    summary_writer.add_scalar('validation/discriminator_loss', loss_d, epoch)
    summary_writer.add_audio('vocoded_sample', vocoded_samples[0], epoch, sample_rate=sample_rate)
    summary_writer.add_audio('griffin_lim_sample', griffin_lim_samples[0], epoch, sample_rate=sample_rate)
    summary_writer.add_audio('augmented_sample', augmented_samples[0], epoch, sample_rate=sample_rate)
    summary_writer.add_audio('ground_truth', ground_truth[0], epoch, sample_rate=sample_rate)
    summary_writer.add_figure('waveform_vocoded_sample', utils.plot_waveform(vocoded_samples[0,0]), epoch)
    summary_writer.add_figure('waveform_griffin_lim_sample', utils.plot_waveform(griffin_lim_samples[0,0]), epoch)
    summary_writer.add_figure('waveform_augmented_sample', utils.plot_waveform(augmented_samples[0,0]), epoch)
    summary_writer.add_figure('waveform_ground_truth', utils.plot_waveform(ground_truth[0,0]), epoch)
    summary_writer.add_figure('spectrogram_generated', utils.plot_spectrogram(generated_spec[0,0]), epoch)
    summary_writer.add_figure('spectrogram_vocoded_sample', utils.plot_spectrogram(mel_spec_transform(vocoded_samples)[0,0]), epoch)
    summary_writer.add_figure('spectrogram_griffin_lim_sample', utils.plot_spectrogram(mel_spec_transform(griffin_lim_samples)[0,0]), epoch)
    summary_writer.add_figure('spectrogram_augmented_sample', utils.plot_spectrogram(mel_spec_transform(augmented_samples)[0,0]), epoch)
    summary_writer.add_figure('spectrogram_ground_truth', utils.plot_spectrogram(mel_spec_transform(ground_truth)[0,0]), epoch)
    

def save_samples(sample_dict, sample_rate, epoch, train_dir):
    for key, sample in sample_dict.items():
        for i in range(sample.shape[0]):
            torchaudio.save(os.path.join(train_dir, 'samples', f'epoch_{epoch}_sample_{i}_{key}.wav'), sample[i].detach().cpu(), sample_rate)


def save_checkpoint(models, optimizers, epoch, train_dir):
    generator, discriminator = models
    optim_generator, optim_discriminator = optimizers
    torch.save({'epoch': epoch,
                'generator_state_dict': generator.state_dict(),
                'discriminator_state_dict': discriminator.state_dict(),
                'generator_optimizer_state_dict': optim_generator.state_dict(),
                'discriminator_optimizer_state_dict': optim_discriminator.state_dict()}, os.path.join(train_dir, 'checkpoints', f'checkpoint_{epoch}.pt'))


def train(args, 
          model_params,
          models, 
          optimizers, 
          train_epoch,
          validate,
          train_loaders,
          val_loaders,
          vocoder_model_params,
          vocoder_model,
          vocode,
          summary_writer):
    
    global_step = 0 
    for epoch in range(1, args.epochs+1):
        logging.info(f'Training epoch {epoch}')
        global_step, models, optimizers = train_epoch(global_step, model_params, models, optimizers, train_loaders, summary_writer)

        logging.info(f'Validating epoch {epoch}')
        ground_truth, augmented_samples, generated_spec, vocoded_samples, griffin_lim_samples, loss_g, loss_d, loss_l1 = validate(model_params, 
                                                                                                                  models, 
                                                                                                                  vocoder_model_params, 
                                                                                                                  vocoder_model, 
                                                                                                                  vocode, 
                                                                                                                  val_loaders)

        # Save tensorboard data, generated samples, and model checkpoint
        save_validation_to_tb(ground_truth, 
                              augmented_samples, 
                              generated_spec, 
                              vocoded_samples, 
                              griffin_lim_samples, 
                              loss_g,
                              loss_d,
                              loss_l1, 
                              args.sample_rate, 
                              epoch, 
                              summary_writer)
        save_samples({'clean': ground_truth, 'augmented': augmented_samples, 'vocoded': vocoded_samples, 'griffin_lim': griffin_lim_samples}, 
                     args.sample_rate, 
                     epoch, 
                     args.train_dir)
        save_checkpoint(models, optimizers, epoch, args.train_dir)
  

def test(model_params, model, test_data):
    pass


if __name__ == '__main__':
    args = parse_args()
    main(args)
