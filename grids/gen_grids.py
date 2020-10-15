import argparse
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import pickle
import sys
import time

sys.path.append('../lib')
import helpers

parser = argparse.ArgumentParser(description='Randomly generates 1D C/F split grids.  Takes a "reference" grid and generates random permutations by flipping points at various probabilities.')
parser.add_argument('--gridsize', metavar='N', type=int, nargs=1, default=31, help='Number of nodes on the grid', required=False)
parser.add_argument('--coarsening', type=float, default=2.0, help='Coarsening factor for reference grid', required=False)
parser.add_argument('--iterations', type=int, default=1000, help='Number of permutations for each probability ', required=False)
parser.add_argument('--outgrids', type=str, default='grids.pkl', help='File to output grids to (pickled list of numpy arrays)', required=False)
parser.add_argument('--outweights', type=str, default='omegas.pkl', help='File to output Jacobi weights to (pickled list of floats)', required=False)
parser.add_argument('--outconv', type=str, default='conv.pkl', help='File to output convergence factors to (pickled list of floats)', required=False)

args = vars(parser.parse_args())
I = args['iterations']
N = args['gridsize']

# Set up 1D poisson system
A = helpers.gen_1d_poisson_fd(N)
x = np.linspace(-1, 1, N)
u = helpers.random_u(N, 0.0055)

# Create "reference" grid which will get randomly permuted
reference_C, reference_F = helpers.grid_from_coarsening_factor(N, args['coarsening'])
reference_conv, reference_omega = helpers.det_conv_factor_optimal_omega(A, reference_C, x, u)

def run_trials(p_switch):
    print(f' -- running trials for p={p_switch:.2f} --')

    rates = np.zeros(I)
    omegas = np.zeros(I)
    grids = np.zeros((I, N))

    # Optimal jacobi weight trial space, which we will sweep over
    omega_trials = np.linspace(0.01, 0.99, 100)

    for i in range(I):
        # Create the randomly permuted "perm_C"
        perm_C = reference_C.copy()
        for j in range(N):
            if np.random.rand(1) < p_switch:
                perm_C[j] = not perm_C[j]

        best_conv, best_omega = helpers.det_conv_factor_optimal_omega(A, perm_C, x, u)

        grids[i] = helpers.grid_to_pytorch(perm_C)
        rates[i] = best_conv
        omegas[i] = best_omega

    return grids, rates, omegas

t_start = time.time()

print(' -- randomly generating permuted grids -- ')

p_trials = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75]
trials = [run_trials(p) for p in p_trials]
trials = []

print(' -- generating grids from coarsening sweep -- ')

coarsenings = np.array([1/9, 1/8, 1/7, 1/6, 1/5, 1/4, 1/3, 1/2, 2, 3, 4, 5, 6, 7, 8, 9])
C = len(coarsenings)
c_omegas = np.zeros(C)
c_rates = np.zeros(C)
c_grids = np.zeros((C, N))

for i, c in enumerate(coarsenings):
    grid_C, grid_F = helpers.grid_from_coarsening_factor(N, c)
    conv, omega = helpers.det_conv_factor_optimal_omega(A, grid_C, x, u)
    c_omegas[i] = omega
    c_rates[i] = conv
    c_grids[i] = helpers.grid_to_pytorch(grid_C)

trials.append((c_grids, c_rates, c_omegas))

print(' -- finished trials -- ')

grids = np.concatenate([ t[0] for t in trials ])
rates = np.concatenate([ t[1] for t in trials ])
omegas = np.concatenate([ t[2] for t in trials ])

with open(args['outgrids'], 'wb') as f:
    pickle.dump(grids, f)

with open(args['outconv'], 'wb') as f:
    pickle.dump(rates, f)

with open(args['outweights'], 'wb') as f:
    pickle.dump(omegas, f)

t_end = time.time()
print(f'finished in {int(t_end-t_start)} seconds')
