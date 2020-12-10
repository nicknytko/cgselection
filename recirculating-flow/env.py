import numpy as np
import numpy.linalg as la
import scipy as sp
import scipy.linalg as sla
import torch
import matplotlib.pyplot as plt
import sys

sys.path.append('../lib')

import helpers
#from model import *

try:
    omegas = np.array(helpers.pickle_load('omegas.pkl'))
    conv = np.array(helpers.pickle_load('conv.pkl'))
    grids = np.array(helpers.pickle_load('grids.pkl'))
    print('loaded weights, convergence rates, and grids to `omegas`, `conv`, `grids`')
except e:
    print('No generated grid data')

# run this with python3 -i env.py
