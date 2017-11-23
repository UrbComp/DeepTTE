import ujson as json
import h5py
import time
import utils
import cPickle
import linecache

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

from ipdb import set_trace

class MySet(Dataset):
    '''
    Main class to load the HDF5 file
    '''
    def __init__(self, input_file):
        self.content = open('./data/' + input_file, 'r').readlines()
        self.length = open('./data/' + input_file.split('.')[0] + '.lens', 'r').readlines()
        self.length = map(lambda x: int(x.rstrip()), self.length)

    def __getitem__(self, idx):
        return self.content[idx]

    def __len__(self):
        return len(self.length)

def collate_fn(data, conf, resample = False):
    data = map(lambda x: json.loads(x), data)

    attr_keys = ['driverID', 'dateID', 'weekID', 'timeID', 'weather', '_dist', '_time']
    traj_keys = ['lngs', 'lats', 'roads', 'time_gap', 'dist_gap']

    attr, traj = {}, {}

    lens = np.asarray([len(item['lngs']) for item in data])

    # fetch attributes
    for key in attr_keys:
        if key in ['_dist', '_time']:
            attr_t = [item[key] for item in data]
            attr_t = torch.FloatTensor(attr_t)
        else:
            attr_t = [item[key] for item in data]
            attr_t = torch.LongTensor(attr_t)

        if key == '_dist':
            attr_t = utils.normalize(attr_t, conf, 'dist')
        if key == '_time':
            attr_t = utils.normalize(attr_t, conf, 'time')

        attr[key] = attr_t

    start_lng = [item['lngs'][0] for item in data]
    start_lat = [item['lats'][0] for item in data]

    end_lng = [item['lngs'][-1] for item in data]
    end_lat = [item['lats'][-1] for item in data]

    attr['start_lng'] = utils.normalize(torch.FloatTensor(start_lng), conf, 'lngs')
    attr['start_lat'] = utils.normalize(torch.FloatTensor(start_lat), conf, 'lats')

    attr['end_lng'] = utils.normalize(torch.FloatTensor(end_lng), conf, 'lngs')
    attr['end_lat'] = utils.normalize(torch.FloatTensor(end_lat), conf, 'lats')


    # fetcht sequences
    if resample == 1:
        indices = []
        for i, item in enumerate(data):
            indices.append(np.random.choice(np.arange(lens[i]), 128))
        for key in traj_keys:
            seqs = np.asarray([np.asarray(item[key])[indices[i]] for i, item in enumerate(data)])
            if key in ['lngs', 'lats', 'time_gap']:
                padded = utils.normalize(seqs, conf, key)
            else:
                padded = seqs

            padded = torch.from_numpy(padded).float()
            traj[key] = padded
    else:
        for key in traj_keys:
            # pad to the max length
            seqs = np.asarray([item[key] for item in data])
            mask = np.arange(lens.max()) < lens[:, None]
            padded = np.zeros(mask.shape, dtype = np.float32)
            padded[mask] = np.concatenate(seqs)

            if key in ['lngs', 'lats', 'time_gap']:
                padded = utils.normalize(padded, conf, key)

            padded = torch.from_numpy(padded).float()
            traj[key] = padded

    lens = lens.tolist()
    traj['lens'] = lens
    return attr, traj

class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.length
        self.indices = range(self.count)

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]


    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size

def get_loader(input_file, batch_size, resample = 0):
    dataset = MySet(input_file = input_file)
    conf = json.load(open('./config.json', 'r'))

    batch_sampler = BatchSampler(dataset, batch_size)

    data_loader = DataLoader(dataset = dataset, \
                             batch_size = 1, \
                             collate_fn = lambda x: collate_fn(x, conf, resample), \
                             num_workers = 4,
                             batch_sampler = batch_sampler,
                             pin_memory = True
    )

    return data_loader
