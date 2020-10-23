import numpy as np
import numpy.linalg as la
import pickle
import torch
import torch.utils.data as td
import torch.nn as nn
import torch.nn.functional as nnF
import torch.optim as optim
import sys
import matplotlib.pyplot as plt

sys.path.append('../lib')
sys.path.append('.')
import helpers

omegas = np.array(helpers.pickle_load('../grids/omegas.pkl'))
trivial_predictor = np.average(omegas)
print(trivial_predictor)

trivial_mse_loss = np.sum((omegas - trivial_predictor)**2) / len(omegas)
trivial_l1_loss = la.norm(omegas - trivial_predictor, 1) / len(omegas)

loss_mse_train = helpers.pickle_load('cnn_train_mse_loss.pkl')
loss_mse_test = helpers.pickle_load('cnn_test_mse_loss.pkl')
plt.semilogy(loss_mse_train, '-', label='Training set')
plt.semilogy(loss_mse_test, '-', label='Testing set')
plt.semilogy([0, len(loss_mse_train)], [trivial_mse_loss, trivial_mse_loss], '-', label='Trivial Predictor')
plt.title('MSE loss per epoch')
plt.xlabel('Epoch number')
plt.ylabel('MSE Loss')
plt.legend()
plt.show(block=True)

loss_l1_train = helpers.pickle_load('cnn_train_l1_loss.pkl')
loss_l1_test = helpers.pickle_load('cnn_test_l1_loss.pkl')
plt.semilogy(loss_l1_train, '-', label='Training set')
plt.semilogy(loss_l1_test, '-', label='Testing set')
plt.semilogy([0, len(loss_l1_train)], [trivial_l1_loss, trivial_l1_loss], '-', label='Trivial Predictor')
plt.title('L1 loss per epoch')
plt.xlabel('Epoch number')
plt.ylabel('L1 Loss')
plt.legend()
plt.show(block=True)

pred_omegas = np.array(helpers.pickle_load('cnn_omega_pred_test.pkl'))
plt.plot(pred_omegas[:,0], pred_omegas[:,1], 'o', label='Predicted Values')
plt.plot([0,1],[0,1], label='A diagonal line')
plt.plot([0,1], [trivial_predictor,trivial_predictor], '-', label='Trivial Predictor')
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('Actual Jacobi Weight')
plt.ylabel('Predicted Jacobi Weight')
plt.title('Actual vs Predicted Jacobi Weight (Testing dataset)')
plt.legend()
plt.show(block=True)

pred_omegas = np.array(helpers.pickle_load('cnn_omega_pred_train.pkl'))
plt.plot(pred_omegas[:,0], pred_omegas[:,1], 'o', label='Predicted Values')
plt.plot([0,1],[0,1], label='A diagonal line')
plt.plot([0,1], [trivial_predictor,trivial_predictor], '-', label='Trivial Predictor')
plt.xlim(0,1)
plt.ylim(0,1)
plt.xlabel('Actual Jacobi Weight')
plt.ylabel('Predicted Jacobi Weight')
plt.title('Actual vs Predicted Jacobi Weight (Training dataset)')
plt.legend()
plt.show(block=True)
