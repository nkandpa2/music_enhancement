import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
import numpy as np
import pandas as pd

import logging
import pdb
from tqdm import tqdm
import itertools
import os

import utils
  
class MedleyDBSolosDataset(Dataset):
    def __init__(self, data_dir, split='training', instruments=None, rate=16000, normalize=True):
        """
        data_dir: path to MedleyDB Solos data directory
        split: training, validation, or test split of dataset
        instruments: list of instruments to use or None for all instruments
        rate: sampling rate
        """
        assert split in ['training', 'test', 'validation']
        self.data_dir = data_dir
        self.rate = rate
        
        self.full_metadata = pd.read_csv(os.path.join(self.data_dir, 'Medley-solos-DB_metadata.csv'))
        self.instruments = instruments if not instruments is None else self.full_metadata.instrument.unique()
        self.metadata = self.full_metadata[(self.full_metadata.subset == split) & (self.full_metadata.instrument.isin(self.instruments))]
        self.file_list = [os.path.join(self.data_dir, f'Medley-solos-DB_{subset}-{instrument_id}_{uuid4}.wav') \
                          for subset, instrument_id, uuid4 in zip(self.metadata.subset, self.metadata.instrument_id, self.metadata.uuid4)]
        
        logging.info(f'Reading MedleyDB Solos Samples from split {split}')
        self.samples = []
        for f in tqdm(self.file_list):
            self.samples.append(utils.load(f, self.rate))
        self.samples = torch.stack(self.samples)
        logging.info(f'Read {self.samples.shape[0]} music samples')   
        
        if normalize:
            self.samples = 0.95*self.samples/self.samples.abs().max(dim=2, keepdim=True)[0]

    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        return self.samples[index]
        
        
class InfiniteDataset(IterableDataset):
    def __init__(self):
        """
        Base class for datasets that repeats forever
        """
        self.index = 0
        self.data = []
        
    def __iter__(self):
        assert len(self.data) > 0, 'Must initialize data before iterating'
        while True:
            yield self.data[self.index]
            self.index = (self.index + 1) % len(self.data)


class NoiseDataset(InfiniteDataset):
    def __init__(self, noise_source, sample_length=47555, rate=16000, split='training'):
        """
        noise_source: either list of directories with .wav files or path to .npy file
        sample_length: length to cut noise samples to
        rate: sampling rate
        """
        assert split in ['training', 'test', 'validation']
        self.sample_length = sample_length
        self.rate = rate
        self.index = 0
        
        if isinstance(noise_source, list):
            self.noise_files = list(itertools.chain.from_iterable([[os.path.join(d, n) for n in filter(lambda f: f.endswith('.wav'), os.listdir(d))] for d in noise_source]))
            logging.info('Loading individual noise sample files')
            self.data = []
            for f in tqdm(self.noise_files):
                noise_sample = utils.load(f, self.rate)
                noise_sample = noise_sample.mean(0, keepdim=True)
                noise_sample = torch.stack(torch.split(noise_sample, sample_length, dim=1)[:-1])
                self.data.append(noise_sample)
            self.data = torch.cat(self.data)
        else:
            logging.info(f'Batch loading from noise sample file for split {split}')
            self.data = torch.from_numpy(np.load(noise_source)[split])
        
        logging.info(f'Read {self.data.shape[0]} noise samples')

    def save(self, path):
        """
        Save data for batch loading later
        path: path to save data at
        """
        logging.info(f'Saving noise samples to {path}')
        np.save(path, self.data.numpy()) 


class ReverbDataset(InfiniteDataset):
    def __init__(self, reverb_source, rate=16000, split='training', trim=True):
        """
        reverb_source: either list of directories with .wav files or path to .npy file
        rate: sampling rate
        """
        assert split in ['training', 'test', 'validation']
        self.rate = rate
        self.index = 0
        
        if isinstance(reverb_source, list):
            self.reverb_files = list(itertools.chain.from_iterable([[os.path.join(d, r) for r in filter(lambda f: f.endswith('.wav'), os.listdir(d))] for d in reverb_source]))
            logging.info('Loading individual room impulse response files')
            self.data = []
            for f in tqdm(self.reverb_files):
                r = utils.load(f, self.rate)
                if trim:
                    direct_impulse_index = r.argmax().item()
                    window_len = int(2.5/1000*self.rate)
                    if direct_impulse_index < window_len:
                        r = torch.cat((torch.zeros(1, window_len - direct_impulse_index), r), dim=1)
                    r = r[:, direct_impulse_index - window_len:]
                self.data.append(r)
            max_ir_length = max([d.shape[1] for d in self.data])
            self.data = [torch.cat((d, torch.zeros(1, max_ir_length-d.shape[1])), dim=1) for d in self.data]
            self.data = torch.stack(self.data)
        else:
            logging.info(f'Batch loading from room impulse response file for split {split}')
            self.data = torch.from_numpy(np.load(reverb_source)[split])

        logging.info(f'Read {self.data.shape[0]} room impulse response samples')

    def save(self, path):
        """
        Save data for batch loading later
        path: path to save data at
        """
        logging.info(f'Saving reverb samples to {path}')
        np.save(path, self.data.numpy()) 


