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

conv = np.array(helpers.pickle_load('../grids/conv.pkl'))
grids = np.array(helpers.pickle_load('../grids/grids.pkl'))

opt_conv = np.argmin(conv)
grid = grids[opt_conv]

helpers.display_grid(grid)
plt.show()

print('True convergence factor:', conv[opt_conv])

grid_T = helpers.grid_to_tensor(grid)
print('Predicted convergence factor:', ds.scale_output(cnn.forward(grid_T).detach().numpy().flatten()))
