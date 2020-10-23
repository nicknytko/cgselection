import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.utils.data as td

if __name__ == '__main__':
    debug = print
else:
    def noop(*args, **kwargs):
        pass
    debug = noop

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 20, 7)
        self.conv2 = nn.Conv1d(20, 20, 5)
        self.conv3 = nn.Conv1d(20, 20, 3)
        self.dropout = nn.Dropout(p=0.2)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(20, 15)
        self.fc2 = nn.Linear(15, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        c = x
        debug('Input', c.shape)
        c = nnF.relu(self.conv1(c))
        debug('First convolution', c.shape)
        c = self.pool(c)
        debug('First pool', c.shape)
        c = nnF.relu(self.conv2(c))
        debug('Second convolution', c.shape)
        c = self.pool(c)
        debug('Second pool', c.shape)
        c = nnF.relu(self.conv3(c))
        debug('Third convolution', c.shape)
        c = self.pool(c)
        debug('Third pool', c.shape)
        c = self.dropout(c)
        debug('Dropout', c.shape)
        c = c.view([-1,1,self.fc1.in_features])
        debug('Flattening', c.shape)
        c = nnF.relu(self.fc1(c))
        debug('Fully Connected Layer 1', c.shape)
        c = nnF.relu(self.fc2(c))
        debug('Fully Connected Layer 2', c.shape)
        c = nnF.relu(self.fc3(c))
        debug('Fully Connected Layer 3', c.shape)
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

        scaled_metric = (self.orig_metric - self.m_min) / (self.m_max - self.m_min)
        self.metric = torch.Tensor(scaled_metric)

    def scale_output(self, output):
        # scale to 0 - (max-min)
        output = output * (self.m_max - self.m_min)
        # scale to min - max
        output = output + self.m_min
        return output

    def __len__(self):
        return self.grids.shape[0]

    def __getitem__(self, idx):
        return [self.grids[idx], self.metric[idx]]

if __name__ == '__main__':
    # Run this file to print verbose debug output on the forward propagation
    # Use it to make sure the layer sizes match up
    cnn = CNN()
    x = torch.zeros((1,1,31))
    cnn.forward(x)
