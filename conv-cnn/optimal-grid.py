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
import time

sys.path.append('../lib')
sys.path.append('.')
import helpers
from model import *

N = 31

cnn = load_model()
ds = GridDataset('../grids/grids.pkl', '../grids/var.pkl', '../grids/conv.pkl')

input_grid = torch.Tensor(helpers.grid_to_pytorch(helpers.random_grid(N))).reshape(1,-1)
x = np.linspace(-1,1,N+1)
#coeffs_orig = np.sqrt(1-x**2)*0.9 + 0.1
#coeffs_orig = np.ones(N+1)
coeffs_orig = 1 + 0.95 * np.sin(400*np.pi*x)
coeffs_mid = helpers.midpt(coeffs_orig)
input_vec = torch.cat((input_grid, torch.Tensor(coeffs_mid).reshape(1,-1)), 0)

plt.plot(x, coeffs_orig)
plt.show()

def vec_to_grid(T):
    T = T.reshape(2,-1)
    t = T[0,:].flatten()
    return np.array(list(map(lambda x: x > 0, t)))

def obj(grid):
    return cnn.forward(torch.Tensor(grid).reshape((1,2,-1))).detach().item()

grid_steps = []

def take_step(grid):
    stepsize = take_step.stepsize
    grid = grid.reshape(2,-1)
    grid_steps.append(vec_to_grid(grid))

    # with probability 1-stepsize we will pick the locally optimal grid
    # otherwise we will permute a random point and call it a day

    if np.random.rand() < stepsize:
        grid = grid.copy()
        n = np.random.randint(0, grid.shape[1])
        grid[0,n] *= -1
        return grid
    else:
        # make N copies of the grids
        grids = torch.Tensor(np.kron(np.ones((N, 1, 1)), grid))

        # grid i has point i flipped
        for i in range(N):
            grids[i,0,i] *= -1

            # pick the optimal grid
            outputs = cnn.forward(grids).detach().numpy().flatten()
            output_min = np.argmin(outputs)
            return grids[output_min]
take_step.stepsize = 0.5

t = time.time()
res = sopt.basinhopping(obj, input_vec, take_step=take_step, stepsize=1, T=0.15)
t = time.time()-t
print('Elapsed', t)

grid = vec_to_grid(res.x)
#helpers.display_grid(grid)
#plt.show()

grid_steps.append(grid)

helpers.display_many_grids(np.array(grid_steps))
plt.xlabel('Grid Points')
plt.ylabel('Iteration Number')
plt.show()

A = helpers.gen_1d_poisson_fd_vc(N, coeffs_orig)
x = np.ones(N)
u = np.zeros(N)
u_ref = la.solve(A,x)

#helpers.disp_grid_convergence(A, x, grid, u)
helpers.display_grid(grid)
plt.show()

print('True convergence factor:', helpers.det_conv_factor_optimal_omega(A, grid, x, u, u_ref)[0])

grid_T = torch.cat((helpers.grid_to_tensor(grid).reshape(1,-1), torch.Tensor(coeffs_mid).reshape(1,-1)), 0).reshape(1,2,-1)
print('Predicted convergence factor:', ds.scale_output(cnn.forward(grid_T).detach().numpy().flatten()))
