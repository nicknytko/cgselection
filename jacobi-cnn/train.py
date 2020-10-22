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

parser = argparse.ArgumentParser(description='Trains a CNN to predict optimal smoothing weight from a random grid .')
parser.add_argument('--iterations', type=int, default=10, help='Number of training iterations to perform', required=False)
parser.add_argument('--batchsize', type=int, default=500, help='Number of entries in each minibatch', required=False)
parser.add_argument('--testsplit', type=float, default=0.85, help='Percent of entries to keep in the training set (as opposed to the testing set).  Should be a value between 0 and 1.', required=False)

args = vars(parser.parse_args())

ds = GridDataset('../grids/grids.pkl', '../grids/omegas.pkl')
cnn = CNN()
loss = nn.MSELoss()
sgd = optim.SGD(cnn.parameters(), lr=0.001, momentum=0.98)

p = args['testsplit']
ltr = int(len(ds)*p)
lte = len(ds) - ltr
train, test = td.random_split(ds, [ltr, lte])

iterations = args['iterations']
mod = int(np.sqrt(iterations))

mse_loss_train = np.zeros(iterations)
mse_loss_test = np.zeros(iterations)
l1_loss_train = np.zeros(iterations)
l1_loss_test = np.zeros(iterations)

def dataset_to_tensor(ds):
    grids = torch.cat(list(map(lambda b: b[0], ds)))
    grids = grids.reshape((grids.shape[0], 1, grids.shape[1]))
    metrics = torch.Tensor(list(map(lambda b: b[1], ds)))
    return grids, metrics

def compute_grid_loss(cnn, ds):
    grids, metrics = dataset_to_tensor(ds)
    output = cnn.forward(grids)

    xhat = output.detach().numpy().flatten()
    x = metrics.detach().numpy().flatten()

    mse = np.average((x-xhat)**2)
    l1 = la.norm(x-xhat, 1) / len(x)
    return mse, l1

test_grids, test_metrics = dataset_to_tensor(test)

for e in range(iterations):
    batches = td.BatchSampler(td.SubsetRandomSampler(train), args['batchsize'], False)

    for i, batch in enumerate(batches):
        sgd.zero_grad()
        batch_grids, batch_metrics = dataset_to_tensor(batch)

        outputs = cnn.forward(batch_grids)
        cur_batch_loss = loss(outputs.reshape([1,1,-1]), batch_metrics.reshape([1,1,-1]))
        cur_batch_loss.backward()
        sgd.step()

    mse_epoch_train, l1_epoch_train = compute_grid_loss(cnn, train)
    mse_epoch_test, l1_epoch_test = compute_grid_loss(cnn, test)

    mse_loss_train[e] = mse_epoch_train; l1_loss_train[e] = l1_epoch_train
    mse_loss_test[e] = mse_epoch_test; l1_loss_test[e] = l1_epoch_test

    if e % mod == 0:
        print(f'({e/iterations*100:.2f}%) \t MSE Loss: {mse_epoch_train:.8f}, L1 Loss: {l1_epoch_train:.8f}')

torch.save(cnn.state_dict(), 'cnn_jacobi_model')
helpers.pickle_save('cnn_train_mse_loss.pkl', mse_loss_train)
helpers.pickle_save('cnn_test_mse_loss.pkl', mse_loss_test)
helpers.pickle_save('cnn_train_l1_loss.pkl', l1_loss_train)
helpers.pickle_save('cnn_test_l1_loss.pkl', l1_loss_test)

omega_samples = []
for sample in train:
    grid, omega = sample
    omega = ds.scale_output(omega.item())
    pred_omega = ds.scale_output(cnn.forward(grid.reshape([1,1,-1]))).item()
    omega_samples.append([omega, pred_omega])
helpers.pickle_save('cnn_omega_pred_train.pkl', omega_samples)

omega_samples = []
for sample in test:
    grid, omega = sample
    omega = ds.scale_output(omega.item())
    pred_omega = ds.scale_output(cnn.forward(grid.reshape([1,1,-1]))).item()
    omega_samples.append([omega, pred_omega])
helpers.pickle_save('cnn_omega_pred_test.pkl', omega_samples)
