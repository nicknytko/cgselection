import numpy as np
import numpy.linalg as la
import pickle
import torch
import torch.utils.data as td
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import sys
import argparse
import scipy.sparse as sp

sys.path.append('../lib')
sys.path.append('.')
import helpers

from model import *

parser = argparse.ArgumentParser(description='Trains a GNN to predict some metric on an input grid.')
parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations to perform', required=False)
parser.add_argument('--batchsize', type=int, default=500, help='Number of entries in each minibatch', required=False)
parser.add_argument('--testsplit', type=float, default=0.85, help='Percent of entries to keep in the training set (as opposed to the testing set).  Should be a value between 0 and 1.', required=False)
parser.add_argument('--metric', type=str, default='conv', help='Metric to train on', required=False)

args = vars(parser.parse_args())

metric = args['metric']
ds = GridDataset('grids.pkl', f'{metric}.pkl')

p = args['testsplit']
#ls = 1000
#ds, _ = td.random_split(dso, [ls, len(dso) - ls])
ltr = int(len(ds)*p)
lte = len(ds) - ltr
train, test = td.random_split(ds, [ltr, lte])

iterations = args['iterations']
mod = int(np.sqrt(iterations))

A, b = helpers.load_recirc_flow(f'recirc-flow-25.mat')

mse_loss_train = []
mse_loss_test = []
l1_loss_train = []
l1_loss_test = []

def dataset_to_tensor(ds):
    grids = torch.cat(list(map(lambda b: b[0].unsqueeze(0), ds)))
    metrics = torch.Tensor(list(map(lambda b: b[1], ds)))
    return grids, metrics

def batched_forward(gnn, grids, batch_size=500):
    n = grids.shape[0]
    xhat = np.zeros(n)
    for i in range(0, n, batch_size):
        ie = min(i + batch_size, n)
        out = gnn.forward(grids[i:ie])
        xx = out.detach().numpy().flatten()
        xhat[i:ie] = xx
    return xhat

def compute_grid_loss(gnn, ds):
    grids, metrics = dataset_to_tensor(ds)

    xhat = batched_forward(gnn, grids)
    x = metrics.detach().numpy().flatten()

    mse = np.average((x-xhat)**2)
    l1 = la.norm(x-xhat, 1) / len(x)

    return mse, l1

test_grids, test_metrics = dataset_to_tensor(test)
# torch.autograd.set_detect_anomaly(True)

# compute initial loss

And = A.data.copy()
And -= np.min(And)
And /= np.max(And)
A_normalized = sp.csr_matrix((And, A.indices, A.indptr), shape=A.shape)
#print(A_normalized.data)
#print(np.min(A_normalized))
#print(np.max(A_normalized))

gnn = GNN(A_normalized)
mse_loss = nn.MSELoss()
l1_loss = nn.L1Loss()
sgd = optim.Adam(gnn.parameters(), lr=0.01)
#sgd = optim.LBFGS(gnn.parameters(), lr=0.1)
mse_initial_train, l1_initial_train = compute_grid_loss(gnn, train)
mse_initial_test, l1_initial_test = compute_grid_loss(gnn, test)
mse_loss_train.append(mse_initial_train); l1_loss_train.append(l1_initial_train)
mse_loss_test.append(mse_initial_test); l1_loss_test.append(l1_initial_test)
print(f'\t e=0 \t MSE Loss: {mse_initial_train:.8f}, L1 Loss: {l1_initial_train:.8f}')

e = 0
while True:
    batches = td.BatchSampler(td.SubsetRandomSampler(train), args['batchsize'], False)

    # print(' -- parameters --')
    # for param in gnn.parameters():
    #     print(torch.linalg.norm(param.data))
    # print(' end parameters')

    for i, batch in enumerate(batches):
        #batch_grids, batch_metrics = dataset_to_tensor(batch)
        def closure():
            sgd.zero_grad()

            cur_batch_mse = 0
            cur_batch_l1 = 0

            for (grid, metric) in batch:
                output = gnn(grid).reshape(1)
                cur_batch_mse += mse_loss(output, metric.reshape(1))
                cur_batch_l1 += l1_loss(output, metric.reshape(1))
            cur_batch_mse.backward()
            return cur_batch_mse

        sgd.step(closure)

    mse_epoch_train, l1_epoch_train = compute_grid_loss(gnn, train)
    mse_epoch_test, l1_epoch_test = compute_grid_loss(gnn, test)

    mse_loss_train.append(mse_epoch_train); l1_loss_train.append(l1_epoch_train)
    mse_loss_test.append(mse_epoch_test); l1_loss_test.append(l1_epoch_test)

    e += 1

    print(f'\t e={e+1} \t MSE Loss: {mse_epoch_train:.8f}, L1 Loss: {l1_epoch_train:.8f}')
    s = input(' (c)ontinue, (s)top [c]:')
    if len(s) > 0 and s.lower()[0] == 's':
        break

mse_loss_train = np.array(mse_loss_train); l1_loss_train = np.array(l1_loss_train)
mse_loss_test = np.array(mse_loss_test); l1_loss_test = np.array(l1_loss_test)

torch.save(gnn.state_dict(), f'gnn_{metric}_model'); print(f'saved model of {metric}')
helpers.pickle_save(f'gnn_{metric}_train_mse_loss.pkl', mse_loss_train); print('saved training mse loss')
helpers.pickle_save(f'gnn_{metric}_test_mse_loss.pkl', mse_loss_test); print('saved testing mse loss')
helpers.pickle_save(f'gnn_{metric}_train_l1_loss.pkl', l1_loss_train); print('saved training l1 loss')
helpers.pickle_save(f'gnn_{metric}_test_l1_loss.pkl', l1_loss_test); print('saved testing l1 loss')

train_grids, train_metric = dataset_to_tensor(train)
metric_vals = ds.scale_output(train_metric)
pred_metric = ds.scale_output(batched_forward(gnn, train_grids))
metric_samples = np.array([metric_vals.numpy().flatten(), pred_metric]).T
helpers.pickle_save(f'gnn_{metric}_pred_train.pkl', metric_samples)

print('saved training samples and predictions')

test_grids, test_metric = dataset_to_tensor(test)
metric_vals = ds.scale_output(test_metric)
pred_metric = ds.scale_output(batched_forward(gnn, test_grids))
metric_samples = np.array([metric_vals.numpy().flatten(), pred_metric]).T
helpers.pickle_save(f'gnn_{metric}_pred_test.pkl', metric_samples)

print('saved testing samples and predictions')
