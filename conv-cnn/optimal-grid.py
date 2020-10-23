import numpy as np
import numpy.linalg as la
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as nnF
import torch.utils.data as td
import torch.optim as optim
import sys
import matplotlib.pyplot as plt

sys.path.append('../lib')
sys.path.append('.')
import helpers
from model import *

N = 31

cnn = load_model()
ds = GridDataset('../grids/grids.pkl', '../grids/conv.pkl')

input_T = helpers.grid_to_tensor(helpers.random_grid(N)).reshape([1,1,-1])
input_T.requires_grad = True
zero_T = torch.zeros([1,1,1])

def tensor_to_grid(T):
    t = T.detach().numpy().flatten()
    return np.array(list(map(lambda x: x > 0, t)))

loss = nn.MSELoss()

sgd = optim.ASGD((input_T,), lr=0.01)
I = 50000
for e in range(I):
    sgd.zero_grad()
    output = cnn.forward(input_T)
    cur_loss = loss(output, zero_T)
    cur_loss.backward()
    sgd.step()

    if e % int(I**0.5) == 0:
        print(f'{(e/I)*100:.2f}% \t {cur_loss.item()}')

print(input_T)

grid = tensor_to_grid(input_T)
helpers.display_grid(grid)
plt.show()

A = helpers.gen_1d_poisson_fd(N)
x = np.ones(N)
u = np.zeros(N)
u_ref = la.solve(A,x)

print('True convergence factor:', helpers.det_conv_factor_optimal_omega(A, grid, x, u, u_ref)[0])

grid_T = helpers.grid_to_tensor(grid)
print('Predicted convergence factor:', ds.scale_output(cnn.forward(grid_T).detach().numpy().flatten()))
