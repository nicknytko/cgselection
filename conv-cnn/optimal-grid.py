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
import scipy.optimize as sopt

sys.path.append('../lib')
sys.path.append('.')
import helpers
from model import *

N = 31

cnn = load_model()
ds = GridDataset('../grids/grids.pkl', '../grids/conv.pkl')

input_vec = helpers.grid_to_pytorch(helpers.random_grid(N))

def vec_to_grid(T):
    t = T.flatten()
    return np.array(list(map(lambda x: x > 0, t)))

def obj(grid):
    return cnn.forward(torch.Tensor(grid).reshape((1,1,-1))).detach().item()

def take_step(grid):
    grid = grid.copy()
    n = np.random.randint(0, len(grid))
    grid[n] = 1.0 - grid[n]
    return grid

res = sopt.basinhopping(obj, input_vec, take_step=take_step)

grid = vec_to_grid(res.x)
helpers.display_grid(grid)
plt.show()

A = helpers.gen_1d_poisson_fd(N)
x = np.ones(N)
u = np.zeros(N)
u_ref = la.solve(A,x)

print('True convergence factor:', helpers.det_conv_factor_optimal_omega(A, grid, x, u, u_ref)[0])

grid_T = helpers.grid_to_tensor(grid)
print('Predicted convergence factor:', ds.scale_output(cnn.forward(grid_T).detach().numpy().flatten()))
