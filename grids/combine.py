import argparse
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import pickle
import sys
import time

sys.path.append('../lib')
import helpers

existing_outputs = ['grids.pkl', 'omegas.pkl', 'conv.pkl', 'var.pkl']
new_outputs = ['new_grids.pkl', 'new_omegas.pkl', 'new_conv.pkl', 'new_var.pkl']

for i in range(len(existing_outputs)):
    ex = existing_outputs[i]
    new = new_outputs[i]

    ex_data = np.array(helpers.pickle_load(ex))
    new_data = np.array(helpers.pickle_load(new))

    combined = np.concatenate((ex_data, new_data))

    helpers.pickle_save(ex, combined)
