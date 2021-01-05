import numpy as np
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.utils.data as td
import scipy.sparse as sp
import torch_geometric as tg

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
        cl = 1
        self.res_conv = SequentialRes([
            nn.Conv2d(1,  cl, 7, padding=3),
            nn.Conv2d(cl, cl, 7, padding=3),
            nn.Conv2d(cl, cl, 7, padding=3),

            nn.Conv2d(cl, cl, 5, padding=2),
            nn.Conv2d(cl, cl, 5, padding=2),
            nn.Conv2d(cl, cl, 5, padding=2),

            nn.Conv2d(cl, cl, 3, padding=1),
            nn.Conv2d(cl, cl, 3, padding=1),
            nn.Conv2d(cl, cl, 3, padding=1),

            nn.MaxPool2d(2, 2)], nnF.relu)

        # fully connected layers
        self.fc_input = 13**2 * cl
        self.fc_layers = 1
        layer_sizes = np.ceil(np.linspace(self.fc_input, 1, self.fc_layers+1)).astype(int)
        layers = []
        for i in range(self.fc_layers):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        self.fully_connected = Sequential(layers, nnF.relu)
        debug(layers)


    def forward(self, x):
        N = x.shape[-1]
        c1 = x.unsqueeze(1); debug('input', c1.shape)

        # Convolve the grid and coefficients
        c_grid = self.res_conv(c1)
        debug('c_grid', c_grid.shape)

        # Flatten
        c_flat = c_grid.view([-1,1,self.fc_input]); debug('flatten', c_flat.shape)

        # Fully connected layers
        c_fc = self.fully_connected(c_flat)

        return c_fc


def elem_vector(n, i):
    with torch.no_grad():
        e = torch.zeros((n,1))
        e[i] = 1.0
    return e

def elem_mat(m, n, col):
    with torch.no_grad():
        E = torch.zeros((m,n))
        E[:,col] = 1.0
    return e

class GNN(nn.Module):
    def __init__(self, A):
        super(GNN, self).__init__()

        self.A = A
        self.edge_index, self.edge_weight = tg.utils.from_scipy_sparse_matrix(A)

        self.conv1 = tg.nn.GCNConv(1, 1)
        self.conv2 = tg.nn.GCNConv(1, 1)
        self.lin1 = nn.Linear(676, 1)

    # def forward_one(self, x):
    #     A = self.A
    #     n = A.shape[0]
    #     h = x.reshape((n,1))
    #     for i in range(self.hidden_layers):
    #         for v in range(n):
    #             mt = torch.zeros(1)
    #             N = sp.find(A[v])[1]
    #             for vN in N:
    #                 if vN == v:
    #                     continue
    #                 evw =  A[v,vN]
    #                 Mt = self.tanh(self.Wfc * (self.Wcf * h[v] + self.b1) * (self.Wdf * evw + self.b2))
    #                 mt = mt + Mt
    #             ev = elem_vector(n, v)
    #             h = h + ev * mt
    #         h = nnF.relu(h)
    #     R = 0
    #     for i in range(n):
    #         R += nnF.relu(self.lin2(nnF.relu(self.lin1(h[i]))))
    #     return R

    def forward(self, x):
        x = x.reshape((-1, 26*26, 1))
        x = nnF.relu(self.conv1(x, self.edge_index, self.edge_weight)).float()
        #print(x[0])
        #print(x)
        x = nnF.relu(self.conv2(x, self.edge_index, self.edge_weight)).float()
        x = x.reshape(-1, 26*26)
        x = nnF.relu(self.lin1(x))
        return x

        # N = x.shape[0]
        # y = torch.ones((N,1))

        # h = x.reshape(N,-1)
        # n = h.shape[1]

        # for i in range(N):
        #    y = y + (elem_vector(N, i) * self.forward_one(x[i]))
        # return y

        # for i in range(self.hidden_layers):
        #     for v in range(n):
        #         mt = torch.zeros(N)
        #         N = sp.find(A[v])[1]
        #         for vN in N:
        #             if vN == v:
        #                 continue
        #             evw = A[v, vN]
        #             Mt = self.tanh(self.Wfc * (self.Wcf * h[:,v] + self.b1) * (self.Wdf * evw + self.b2))
        #             mt = mt + Mt
        #         em = elem_mat(N,n)
        #         h = h + em * mt

class GridDataset(td.Dataset):
    def load_pickle(self, fname):
        with open(fname, 'rb') as f:
            return pickle.load(f)

    def __init__(self, grid_file, metric_file):
        grids = torch.Tensor(self.load_pickle(grid_file))
        self.grids = grids

        self.orig_metric = np.array(self.load_pickle(metric_file))
        self.orig_metric[np.where(np.isnan(self.orig_metric))] = 0
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
    ds = GridDataset('grids.pkl', 'omegas.pkl')
    cnn = CNN()
    x = torch.zeros((1,26,26))
    cnn.forward(x)
