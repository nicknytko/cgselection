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
        cl = 5
        self.conv7_1 = nn.Conv1d(1,  cl, 7, padding=3)
        self.conv7_2 = nn.Conv1d(cl, cl, 7, padding=3)
        self.conv7_3 = nn.Conv1d(cl, cl, 7, padding=3)
        self.conv7_4 = nn.Conv1d(cl, cl, 7, padding=3)
        #self.adamax1 = nn.AdaptiveMaxPool1d(19)
        #self.adamax2 = nn.AdaptiveMaxPool1d(13)
        self.conv5_1 = nn.Conv1d(cl, cl, 5, padding=2)
        self.conv5_2 = nn.Conv1d(cl, cl, 5, padding=2)
        self.conv5_3 = nn.Conv1d(cl, cl, 5, padding=2)
        self.conv5_4 = nn.Conv1d(cl, cl, 5, padding=2)
        self.conv3_1 = nn.Conv1d(cl, cl, 3, padding=1)
        self.conv3_2 = nn.Conv1d(cl, cl, 3, padding=1)
        self.conv3_3 = nn.Conv1d(cl, cl, 3, padding=1)
        self.conv3_4 = nn.Conv1d(cl, cl, 3, padding=1)
        self.pool = nn.MaxPool1d(2, 2)
        self.fc1 = nn.Linear(75, 40)
        self.fc2 = nn.Linear(40, 10)
        self.fc3 = nn.Linear(10, 1)

    def forward(self, x):
        c1 = x; debug('input', c1.shape)

        # First convolution with kernel size 7
        c2_1 = nnF.relu(self.conv7_1(c1)); debug('conv7_1', c2_1.shape)
        c2_2 = nnF.relu(self.conv7_2(c2_1 + c1)); debug('conv7_2', c2_2.shape)
        c2_3 = nnF.relu(self.conv7_2(c2_2)); debug('conv7_3', c2_3.shape)
        c2_4 = nnF.relu(self.conv7_2(c2_3 + c2_2)); debug('conv7_4', c2_4.shape)

        # Second convolution with kernel size 5
        c3_1 = nnF.relu(self.conv5_1(c2_4)); debug('conv5_1', c3_1.shape)
        c3_2 = nnF.relu(self.conv5_2(c3_1 + c2_4)); debug('conv5_2', c3_2.shape)
        c3_3 = nnF.relu(self.conv5_3(c3_2)); debug('conv5_3', c3_3.shape)
        c3_4 = nnF.relu(self.conv5_4(c3_3 + c3_2)); debug('conv5_4', c3_4.shape)

        # Third convolution with kernel size
        #c4_1 = nnF.relu(self.conv3_1(c3 + self.adamax1(c2))); debug('conv3_1', c4.shape)
        c4_1 = nnF.relu(self.conv3_1(c3_4)); debug('conv3_1', c4_1.shape)
        c4_2 = nnF.relu(self.conv3_2(c4_1 + c3_4)); debug('conv3_2', c4_2.shape)
        c4_3 = nnF.relu(self.conv3_3(c4_2)); debug('conv3_3', c4_3.shape)
        c4_4 = nnF.relu(self.conv3_4(c4_3 + c4_2)); debug('conv3_4', c4_4.shape)

        # Max pooling layer
        c_max = self.pool(c4_4 + c4_2 + c3_4); debug('max pool', c_max.shape)

        # Flatten
        c_flat = c_max.view([-1,1,self.fc1.in_features]); debug('flatten', c_flat.shape)

        # Fully connected layers
        c_fc = nnF.relu(self.fc1(c_flat)); debug('fc1', c_fc.shape)
        c_fc = nnF.relu(self.fc2(c_fc));   debug('fc2', c_fc.shape)
        c_fc = nnF.relu(self.fc3(c_fc));   debug('fc3', c_fc.shape)

        return c_fc

class GridDataset(td.Dataset):
    def load_pickle(self, fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def __init__(self, grid_file, metric_file):
        grid_ary, grid_unique_idx = np.unique(self.load_pickle(grid_file), axis=0, return_index=True)
        grids = torch.Tensor(grid_ary)
        self.grids = grids.reshape((grids.shape[0], 1, grids.shape[1]))

        self.orig_metric = np.array(self.load_pickle(metric_file))[grid_unique_idx]
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

def load_model():
    cnn = CNN()
    cnn.load_state_dict(torch.load('cnn_jacobi_model'))
    cnn.eval()
    return cnn

if __name__ == '__main__':
    # Run this file to print verbose debug output on the forward propagation
    # Use it to make sure the layer sizes match up
    cnn = CNN()
    x = torch.zeros((1,1,31))
    cnn.forward(x)
