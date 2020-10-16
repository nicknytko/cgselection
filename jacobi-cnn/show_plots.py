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

loss = helpers.pickle_load('cnn_train_loss.pkl')
plt.semilogy(loss, '-')
plt.title('MSE loss per iteration')
plt.xlabel('Iteration number')
plt.ylabel('MSE Loss')
plt.show(block=True)

l1_loss = helpers.pickle_load('cnn_train_l1_loss.pkl')
plt.semilogy(l1_loss, '-')
plt.title('L1 loss per iteration')
plt.xlabel('Iteration number')
plt.ylabel('L1 Loss')
plt.show(block=True)

pred_omegas = np.array(helpers.pickle_load('cnn_omega_pred_test.pkl'))
plt.plot(pred_omegas[:,0], pred_omegas[:,1], 'o')
minimum = 0.4
plt.plot([minimum,1],[minimum,1])
plt.xlim(minimum,1)
plt.ylim(minimum,1)
plt.xlabel('Actual Jacobi Weight')
plt.ylabel('Predicted Jacobi Weight')
plt.title('Actual vs Predicted Jacobi Weight (Testing dataset)')
plt.show(block=True)

pred_omegas = np.array(helpers.pickle_load('cnn_omega_pred_train.pkl'))
plt.plot(pred_omegas[:,0], pred_omegas[:,1], 'o')
minimum = 0.3
plt.plot([minimum,1],[minimum,1])
plt.xlim(minimum,1)
plt.ylim(minimum,1)
plt.xlabel('Actual Jacobi Weight')
plt.ylabel('Predicted Jacobi Weight')
plt.title('Actual vs Predicted Jacobi Weight (Training dataset)')
plt.show(block=True)
