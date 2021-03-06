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

ds = GridDataset('../grids/grids.pkl', '../grids/var.pkl', '../grids/omegas.pkl')
cnn = CNN()
loss = nn.MSELoss()
sgd = optim.Adam(cnn.parameters(), lr=0.01)

p = args['testsplit']
ltr = int(len(ds)*p)
lte = len(ds) - ltr
train, test = td.random_split(ds, [ltr, lte])

iterations = args['iterations']
mod = int(np.sqrt(iterations))

mse_loss_train = np.zeros(iterations+1)
mse_loss_test = np.zeros(iterations+1)
l1_loss_train = np.zeros(iterations+1)
l1_loss_test = np.zeros(iterations+1)

def dataset_to_tensor(ds):
    grids = torch.cat(list(map(lambda b: b[0], ds)))
    grids = grids.reshape((grids.shape[0]//2, 2, grids.shape[1]))
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

# compute initial loss
mse_initial_train, l1_initial_train = compute_grid_loss(cnn, train)
mse_initial_test, l1_initial_test = compute_grid_loss(cnn, test)
mse_loss_train[0] = mse_initial_train; l1_loss_train[0] = l1_initial_train
mse_loss_test[0] = mse_initial_test; l1_loss_test[0] = l1_initial_test
print(f'(0%) \t\t e=0 \t MSE Loss: {mse_initial_train:.8f}, L1 Loss: {l1_initial_train:.8f}')

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

    mse_loss_train[e+1] = mse_epoch_train; l1_loss_train[e+1] = l1_epoch_train
    mse_loss_test[e+1] = mse_epoch_test; l1_loss_test[e+1] = l1_epoch_test

    print(f'({(e+1)/iterations*100:.2f}%) \t e={e+1} \t MSE Loss: {mse_epoch_train:.8f}, L1 Loss: {l1_epoch_train:.8f}')

torch.save(cnn.state_dict(), 'cnn_jacobi_model'); print('saved model')
helpers.pickle_save('cnn_train_mse_loss.pkl', mse_loss_train); print('saved training mse loss')
helpers.pickle_save('cnn_test_mse_loss.pkl', mse_loss_test); print('saved testing mse loss')
helpers.pickle_save('cnn_train_l1_loss.pkl', l1_loss_train); print('saved training l1 loss')
helpers.pickle_save('cnn_test_l1_loss.pkl', l1_loss_test); print('saved testing l1 loss')

train_grids, train_omegas = dataset_to_tensor(train)
omegas = ds.scale_output(train_omegas)
pred_omegas = ds.scale_output(cnn.forward(train_grids))
omega_samples = np.array([omegas.numpy().flatten(), pred_omegas.detach().numpy().flatten()]).T
helpers.pickle_save('cnn_omega_pred_train.pkl', omega_samples)

print('saved training samples and predictions')

test_grids, test_omegas = dataset_to_tensor(train)
omegas = ds.scale_output(test_omegas)
pred_omegas = ds.scale_output(cnn.forward(test_grids))
omega_samples = np.array([omegas.numpy().flatten(), pred_omegas.detach().numpy().flatten()]).T
helpers.pickle_save('cnn_omega_pred_test.pkl', omega_samples)

print('saved testing samples and predictions')
