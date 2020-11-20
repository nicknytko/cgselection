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


# Sequential Residual block
# every odd module takes outputs n-1 and n-2. even module takes output n-1.
# important that all modules have the same output shape
class SequentialRes(nn.Module):
    def __init__(self, modules, activation):
        super(SequentialRes, self).__init__()
        self.modules = modules
        for i, m in enumerate(modules):
            self.add_module(str(i), m)
        self.activation = activation

    def forward(self, x):
        prev = x*0
        cur = x
        for i, m in enumerate(self.modules):
            if i % 2 == 0:
                cur_next = self.activation(m(prev + cur))
            else:
                cur_next = self.activation(m(cur))
            prev = cur
            cur = cur_next
        return cur


# sequential block
class Sequential(nn.Module):
    def __init__(self, modules, activation):
        super(Sequential, self).__init__()
        self.modules = modules
        for i, m in enumerate(modules):
            self.add_module(str(i), m)
        self.activation = activation

    def forward(self, x):
        cur = x
        for m in self.modules:
            cur = self.activation(m(cur))
        return cur


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # convolutional layer
        cl = 10
        self.res_conv = SequentialRes([
            nn.Conv1d(2, cl, 7, padding=3),
            nn.Conv1d(cl, cl, 7, padding=3),
            nn.Conv1d(cl, cl, 7, padding=3),
            nn.Conv1d(cl, cl, 7, padding=3),
            nn.Conv1d(cl, cl, 7, padding=3),
            nn.Conv1d(cl, cl, 7, padding=3),

            nn.Conv1d(cl, cl, 5, padding=2),
            nn.Conv1d(cl, cl, 5, padding=2),
            nn.Conv1d(cl, cl, 5, padding=2),
            nn.Conv1d(cl, cl, 5, padding=2),
            nn.Conv1d(cl, cl, 5, padding=2),
            nn.Conv1d(cl, cl, 5, padding=2),

            nn.Conv1d(cl, cl, 3, padding=1),
            nn.Conv1d(cl, cl, 3, padding=1),
            nn.Conv1d(cl, cl, 3, padding=1),
            nn.Conv1d(cl, cl, 3, padding=1),
            nn.Conv1d(cl, cl, 3, padding=1),
            nn.Conv1d(cl, cl, 3, padding=1),

            nn.MaxPool1d(2, 2)], nnF.relu)

        # fully connected layers
        self.fc_input = 15 * cl
        self.fc_layers = 8
        layer_sizes = np.ceil(np.linspace(self.fc_input, 1, self.fc_layers+1)).astype(int)
        layers = []
        for i in range(self.fc_layers):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.fully_connected = Sequential(layers, nnF.relu)
        debug(layers)


    def forward(self, x):
        N = x.shape[2]
        c1 = x; debug('input', c1.shape)

        # Convolve the grid and coefficients
        c_grid = self.res_conv(c1)
        debug('c_grid', c_grid.shape)

        # Flatten and concatenate
        #c_flat = torch.cat((c_grid, c_coeff)).view([-1,1,self.fc_input]); debug('flatten', c_flat.shape)
        c_flat = c_grid.view([-1,1,self.fc_input]); debug('flatten', c_flat.shape)

        # Fully connected layers
        c_fc = self.fully_connected(c_flat)

        return c_fc

class GridDataset(td.Dataset):
    def load_pickle(self, fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def __init__(self, grid_file, coefficients_file, metric_file):
        grids = torch.Tensor(self.load_pickle(grid_file))
        grid_coeffs_tensor = torch.Tensor(self.load_pickle(coefficients_file))
        grid_coeffs_tensor = grid_coeffs_tensor.reshape((-1, 1, grid_coeffs_tensor.shape[1]))
        self.coeffs_radius = torch.max(torch.abs(grid_coeffs_tensor))

        grid_tensor = grids.reshape((grids.shape[0], 1, grids.shape[1]))
        self.grids = torch.cat((grid_tensor, grid_coeffs_tensor), dim=1)

        self.orig_metric = np.array(self.load_pickle(metric_file))
        self.m_min = np.min(self.orig_metric)
        self.m_max = np.max(self.orig_metric)

        scaled_metric = (self.orig_metric - self.m_min) / (self.m_max - self.m_min)
        self.metric = torch.Tensor(scaled_metric)

    def scale_coeff_input(self, c):
        return c / self.coeffs_radius

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
    ds = GridDataset('../grids/grids.pkl', '../grids/var.pkl', '../grids/conv.pkl')
    cnn = CNN()
    x = torch.zeros((1,2,31))
    cnn.forward(x)
