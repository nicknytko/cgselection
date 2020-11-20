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
parser.add_argument('--iterations', type=int, default=2000, help='Number of permutations for each probability ', required=False)
parser.add_argument('--outgrids', type=str, default='grids.pkl', help='File to output grids to (pickled list of numpy arrays)', required=False)
parser.add_argument('--outweights', type=str, default='omegas.pkl', help='File to output Jacobi weights to (pickled list of floats)', required=False)
parser.add_argument('--outconv', type=str, default='conv.pkl', help='File to output convergence factors to (pickled list of floats)', required=False)
parser.add_argument('--outvar', type=str, default='var.pkl', help='File to output variable coefficients k(x) to (pickled list of numpy arrays)', required=False)

args = vars(parser.parse_args())
I = args['iterations']
N = args['gridsize']

# Create our domain omega.  We will also use this as the RHS: f(x) = x.
x = np.linspace(-1, 1, N)

t_start = time.time()

print('randomly generating permuted grids')

coarsenings = np.array([9, 8, 7, 6, 5, 4, 3, 2])
p_trials = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75]
trials = []

def random_value(N, radius=1, mean=0):
    return (np.random.rand(N) * 2 - 1) * radius + mean

# random c functions.  all return >0 values

def c_constant(x):
    return random_value(len(x), 5, 5)

def c_random(x):
    return random_value(len(x), 0.95, 1) * (np.random.rand(1)*10 + 1)

def c_cos(x):
    lmbda = random_value(1, 5, 5)
    omega = random_value(1, 5, 5)
    alpha = random_value(1, 5, 5)
    return omega*np.cos(x * np.pi * lmbda) + alpha

def c_poly(x):
    n = np.random.randint(2, 5)
    coeffs = random_value(n, 10)
    polyval = np.polyval(coeffs, x)
    return np.abs(polyval) + 0.01

c_functions = [c_constant, c_random, c_cos, c_poly]
def rand_coeffs():
    return np.random.choice(c_functions)(np.linspace(0,1,N+1))

def midpt(x):
    return np.average(np.column_stack([x[1:], x[:-1]]), axis=1)

for c in coarsenings:
    print(f'  generating grids from coarsening by {c}')
    # Create "reference" grid which will get randomly permuted
    reference_C, reference_F = helpers.grid_from_coarsening_factor(N, c)

    for p in p_trials:
        print(f'    running trials for p={p:.2f}')

        rates = np.zeros(I)
        omegas = np.zeros(I)
        grids = np.zeros((I, N))
        coeffs = np.zeros((I, N))

        for i in range(I):
            cs = rand_coeffs()
            A = helpers.gen_1d_poisson_fd_vc(N, cs)
            u = np.zeros(N)
            u_ref = la.solve(A, x)

            # Create the randomly permuted "perm_C"
            perm_C = reference_C.copy()
            for j in range(N):
                if np.random.rand(1) < p:
                    perm_C[j] = not perm_C[j]
            best_conv, best_omega = helpers.det_conv_factor_optimal_omega_numopt(A, perm_C, x, u, u_ref)

            grids[i] = helpers.grid_to_pytorch(perm_C)
            rates[i] = best_conv
            omegas[i] = best_omega
            coeffs[i] = midpt(cs)

        trials.append((grids, rates, omegas, coeffs))

print('finished trials')

grids = np.concatenate([ t[0] for t in trials ])
rates = np.concatenate([ t[1] for t in trials ])
omegas = np.concatenate([ t[2] for t in trials ])
var = np.concatenate([ t[3] for t in trials ])

# append files to existing files

outputs = [
    {
        'var': grids,
        'fname': args['outgrids']
    },
    {
        'var': rates,
        'fname': args['outweights']
    },
    {
        'var': omegas,
        'fname': args['outconv']
    },
    {
        'var': var,
        'fname': args['outvar']
    },
]

for o in outputs:
    var = o['var']
    fname = o['fname']
    try:
        existing = helpers.pickle_load(fname)
        existing = np.concatenate([ np.array(existing), var ])
    except Exception as e:
        print(fname, e)
        existing = var
    helpers.pickle_save(fname, existing)

t_end = time.time()
print(f'finished in {int(t_end-t_start)} seconds')
