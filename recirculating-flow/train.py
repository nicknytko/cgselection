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

sys.path.append('../lib')
sys.path.append('.')
import helpers

from model import *

parser = argparse.ArgumentParser(description='Trains a CNN to predict some metric on an input grid.')
parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations to perform', required=False)
parser.add_argument('--batchsize', type=int, default=500, help='Number of entries in each minibatch', required=False)
parser.add_argument('--testsplit', type=float, default=0.85, help='Percent of entries to keep in the training set (as opposed to the testing set).  Should be a value between 0 and 1.', required=False)
parser.add_argument('--metric', type=str, default='conv', help='Metric to train on', required=False)

args = vars(parser.parse_args())

metric = args['metric']
ds = GridDataset('grids.pkl', f'{metric}.pkl')
# sgd = optim.LBFGS(cnn.parameters(), lr=0.1)

p = args['testsplit']
ltr = int(len(ds)*p)
lte = len(ds) - ltr
train, test = td.random_split(ds, [ltr, lte])

iterations = args['iterations']
mod = int(np.sqrt(iterations))

mse_loss_train = []
mse_loss_test = []
l1_loss_train = []
l1_loss_test = []

def dataset_to_tensor(ds):
    grids = torch.cat(list(map(lambda b: b[0].unsqueeze(0), ds)))
    metrics = torch.Tensor(list(map(lambda b: b[1], ds)))
    return grids, metrics

def batched_forward(cnn, grids, batch_size=500):
    n = grids.shape[0]
    xhat = np.zeros(n)
    for i in range(0, n, batch_size):
        ie = min(i + batch_size, n)
        out = cnn.forward(grids[i:ie])
        xx = out.detach().numpy().flatten()
        xhat[i:ie] = xx
    return xhat

def compute_grid_loss(cnn, ds):
    grids, metrics = dataset_to_tensor(ds)

    xhat = batched_forward(cnn, grids)
    x = metrics.detach().numpy().flatten()

    mse = np.average((x-xhat)**2)
    l1 = la.norm(x-xhat, 1) / len(x)
    return mse, l1

test_grids, test_metrics = dataset_to_tensor(test)

# compute initial loss
while True:
    cnn = CNN()
    loss = nn.MSELoss()
    sgd = optim.Adam(cnn.parameters(), lr=0.01)
    mse_initial_train, l1_initial_train = compute_grid_loss(cnn, train)
    mse_initial_test, l1_initial_test = compute_grid_loss(cnn, test)
    if mse_initial_train < 0.4:
        break
    else:
        print(mse_initial_train)
mse_loss_train.append(mse_initial_train); l1_loss_train.append(l1_initial_train)
mse_loss_test.append(mse_initial_test); l1_loss_test.append(l1_initial_test)
print(f'\t e=0 \t MSE Loss: {mse_initial_train:.8f}, L1 Loss: {l1_initial_train:.8f}')

e = 0
while True:
    batches = td.BatchSampler(td.SubsetRandomSampler(train), args['batchsize'], False)

    # print(' -- parameters --')
    # for param in cnn.parameters():
    #     print(torch.linalg.norm(param.data))
    # print(' end parameters')

    for i, batch in enumerate(batches):
        batch_grids, batch_metrics = dataset_to_tensor(batch)

        def closure():
            sgd.zero_grad()
            outputs = cnn.forward(batch_grids)
            cur_batch_loss = loss(outputs.reshape([1,1,-1]), batch_metrics.reshape([1,1,-1]))
            cur_batch_loss.backward()
            return cur_batch_loss

        sgd.step(closure)

        mse_epoch_train, l1_epoch_train = compute_grid_loss(cnn, train)
        mse_epoch_test, l1_epoch_test = compute_grid_loss(cnn, test)

        mse_loss_train.append(mse_epoch_train); l1_loss_train.append(l1_epoch_train)
        mse_loss_test.append(mse_epoch_test); l1_loss_test.append(l1_epoch_test)

        e += 1

    mse_epoch_train, l1_epoch_train = compute_grid_loss(cnn, train)
    mse_epoch_test, l1_epoch_test = compute_grid_loss(cnn, test)
    print(f'\t e={e+1} \t MSE Loss: {mse_epoch_train:.8f}, L1 Loss: {l1_epoch_train:.8f}')
    s = input(' (c)ontinue, (s)top [c]:')
    if len(s) > 0 and s.lower()[0] == 's':
        break

mse_loss_train = np.array(mse_loss_train); l1_loss_train = np.array(l1_loss_train)
mse_loss_test = np.array(mse_loss_test); l1_loss_test = np.array(l1_loss_test)

torch.save(cnn.state_dict(), f'cnn_{metric}_model'); print(f'saved model of {metric}')
helpers.pickle_save(f'cnn_{metric}_train_mse_loss.pkl', mse_loss_train); print('saved training mse loss')
helpers.pickle_save(f'cnn_{metric}_test_mse_loss.pkl', mse_loss_test); print('saved testing mse loss')
helpers.pickle_save(f'cnn_{metric}_train_l1_loss.pkl', l1_loss_train); print('saved training l1 loss')
helpers.pickle_save(f'cnn_{metric}_test_l1_loss.pkl', l1_loss_test); print('saved testing l1 loss')

train_grids, train_metric = dataset_to_tensor(train)
metric_vals = ds.scale_output(train_metric)
pred_metric = ds.scale_output(batched_forward(cnn, train_grids))
metric_samples = np.array([metric_vals.numpy().flatten(), pred_metric]).T
helpers.pickle_save(f'cnn_{metric}_pred_train.pkl', metric_samples)

print('saved training samples and predictions')

test_grids, test_metric = dataset_to_tensor(test)
metric_vals = ds.scale_output(test_metric)
pred_metric = ds.scale_output(batched_forward(cnn, test_grids))
metric_samples = np.array([metric_vals.numpy().flatten(), pred_metric]).T
helpers.pickle_save(f'cnn_{metric}_pred_test.pkl', metric_samples)

print('saved testing samples and predictions')
