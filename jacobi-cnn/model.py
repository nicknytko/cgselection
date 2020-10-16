import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.utils.data as td

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 2, 5)
        self.conv2 = nn.Conv1d(2, 8, 5)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(184, 92)
        self.fc2 = nn.Linear(92, 1)

    def forward(self, x):
        c = x
        c = nnF.relu(self.conv1(c))
        c = nnF.relu(self.conv2(c))
        c = c.view([-1,1,184])
        c = nnF.relu(self.fc1(c))
        c = nnF.relu(self.fc2(c))
        return c

class GridDataset(td.Dataset):
    def load_pickle(self, fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def __init__(self, grid_file, metric_file):
        grids = torch.Tensor(self.load_pickle(grid_file))
        self.grids = grids.reshape((grids.shape[0], 1, grids.shape[1]))

        self.orig_metric = self.load_pickle(metric_file)
        self.m_min = np.min(self.orig_metric)
        self.m_max = np.max(self.orig_metric)

        scaled_metric = (((self.orig_metric - self.m_min) / (self.m_max - self.m_min)) * 2) - 1
        self.metric = torch.Tensor(scaled_metric)

    def scale_output(self, output):
        # scale to 0 - 1
        output = (output + 1) / 2
        # scale to 0 - (max-min)
        output = output * (self.m_max - self.m_min)
        # scale to min - max
        output = output + self.m_min
        return output

    def __len__(self):
        return self.grids.shape[0]

    def __getitem__(self, idx):
        return [self.grids[idx], self.metric[idx]]
