import torch
from torch.utils.data import Dataset

import numpy as np
from utils import *

class RawDataset(Dataset):
    def __init__(self, datadir, set_name, uniform, delta, get_idxs, max_len=128):
        self.uniform = uniform
        self.delta = delta
        self.get_idxs = get_idxs
        self.datadir = datadir
        self.max_len = max_len

        alignment = open(datadir + '/alignment_' + set_name + '.txt', 'r').readlines()

        self.data = []
        for i, line in enumerate(alignment):
            name, string = line.split('\t')

            y = encode_string(string.strip())
            if len(y) > self.max_len: continue
            self.data.append((name + '.npy', y))

        #print(max(len(self.data[i][1]) for i in range(len(self.data))))
        #print(min(len(self.data[i][1]) for i in range(len(self.data))))
        #print(len(self.data))
        #asdf

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        name, y = self.data[idx]

        X_raw = np.load(self.datadir + '/NPY/' + name).T

        #min_y, max_y = X_raw[:, 1].min().item(), X_raw[:, 1].max().item()
        #min_x = X_raw[:, 0].min().item()

        #X_raw[:, 1] = (X_raw[:, 1] - min_y) / (max_y - min_y)
        #X_raw[:, 0] = (X_raw[:, 0] - min_x) / (max_y - min_y)

        #if self.delta:
        #    X_prev = X_raw.copy()
        #    X_prev[1:] = X_raw[:-1]
        #    X_raw[:, :2] -= X_prev[:, :2]

        X_len = min(625, len(X_raw))
        X = np.zeros((625, X_raw.shape[1]))
        X[:X_raw.shape[0]] = X_raw[:625]

        y_len = len(y)
        y = np.array(y[:self.max_len] + [0]*max(0, self.max_len-len(y)))

        return torch.from_numpy(X).float(), torch.from_numpy(y).int(), X_len, y_len
