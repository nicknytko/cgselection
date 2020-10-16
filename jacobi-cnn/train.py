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
sgd = optim.SGD(cnn.parameters(), lr=0.001, momentum=1)

p = args['testsplit']
ltr = int(len(ds)*p)
lte = len(ds) - ltr
train, test = td.random_split(ds, [ltr, lte])

iterations = args['iterations']
mod = int(np.sqrt(iterations))
overall_loss = np.zeros(iterations)
l1_loss = np.zeros(iterations)

for e in range(iterations):
    batches = td.BatchSampler(td.SubsetRandomSampler(train), args['batchsize'], True)

    batch_loss = 0
    batch_l1_loss = 0
    for i, batch in enumerate(batches):
        sgd.zero_grad()
        batch_grids = torch.cat(list(map(lambda b: b[0], batch)))
        batch_grids = batch_grids.reshape((batch_grids.shape[0], 1, batch_grids.shape[1]))
        batch_metrics = torch.Tensor(list(map(lambda b: b[1], batch)))

        outputs = cnn.forward(batch_grids)
        cur_batch_loss = loss(outputs.reshape([1,1,-1]), batch_metrics.reshape([1,1,-1]))
        cur_batch_loss.backward()
        batch_loss += cur_batch_loss.item()
        batch_l1_loss += la.norm((outputs.detach().numpy().flatten()) - batch_metrics.detach().numpy().flatten(), 1)
        sgd.step()

    overall_loss[e] = batch_loss
    l1_loss[e] = batch_l1_loss

    if e % mod == 0:
        print(f'({e/iterations*100:.2f}%) \t Loss: {batch_loss:.4f}')

torch.save(cnn.state_dict(), 'cnn_jacobi_model')
helpers.pickle_save('cnn_train_loss.pkl', overall_loss)
helpers.pickle_save('cnn_train_l1_loss.pkl', l1_loss)

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
