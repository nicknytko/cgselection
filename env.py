import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.linalg as sla
import torch
import matplotlib.pyplot as plt
import sys

sys.path.append('lib')
sys.path.append('jacobi-cnn')

import helpers
from model import *

try:
    omegas = np.array(helpers.pickle_load('grids/omegas.pkl'))
    conv = np.array(helpers.pickle_load('grids/conv.pkl'))
    conv_rate = -np.log10(conv)
    grids = np.array(helpers.pickle_load('grids/grids.pkl'))
    var = np.array(helpers.pickle_load('grids/var.pkl'))
    print('loaded weights, convergence factors, and grids to `omegas`, `conv`, `grids`, `var`')
except e:
    print('No generated grid data')

# run this with python3 -i env.py
