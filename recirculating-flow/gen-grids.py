import argparse
import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import scipy.optimize
import pickle
import sys
import time
import pyamg

sys.path.append('../lib')
import helpers

parser = argparse.ArgumentParser(description='Randomly generates 2D C/F split grids.  Takes various reference grids and generates random permutations by flipping points at various probabilities')
parser.add_argument('--gridsize', metavar='N', type=int, nargs=1, default=25, help='Number of nodes on the grid', required=False)
parser.add_argument('--iterations', type=int, default=500, help='Number of permutations for each probability ', required=False)
parser.add_argument('--outgrids', type=str, default='grids.pkl', help='File to output grids to (pickled list of numpy arrays)', required=False)
parser.add_argument('--outweights', type=str, default='omegas.pkl', help='File to output Jacobi weights to (pickled list of floats)', required=False)
parser.add_argument('--outconv', type=str, default='conv.pkl', help='File to output convergence factors to (pickled list of floats)', required=False)

args = vars(parser.parse_args())
I = args['iterations']
N = args['gridsize']+1
A, b = helpers.load_recirc_flow(f'recirc-flow-{N-1}.mat')
S = pyamg.strength.classical_strength_of_connection(A, theta=0.25)
x_ref = spla.spsolve(A,b)

t_start = time.time()

print('randomly generating permuted grids')

# reference grid generators

def ref_all_fine(A):
    N = int(A.shape[0]**0.5)
    z = np.zeros((N,N))
    return 'All fine', z

def ref_all_coarse(A):
    N = int(A.shape[0]**0.5)
    z = np.ones((N,N))
    return 'All coarse', z

def ref_coarsen_by(c):
    def f(A):
        N = int(A.shape[0]**0.5)
        zz = np.zeros((N,N))
        for x in range(0,N,c):
            for y in range(0,N,c):
                zz[x,y] = 1
        return f'Coarsening by {c}', zz
    return f

def ref_amg(A):
    N = int(A.shape[0]**0.5)
    spl = pyamg.classical.RS(S)
    return 'AMG - RS', spl.reshape((N,N))

# multigrid

Dinv = sp.diags(1.0/A.diagonal())
def jacobi(A, b, x, omega=0.666, nu=2):
    # Sorry, 'A' doesn't actually matter here.
    # Precompute D^{-1} ahead of time
    for i in range(nu):
        x += omega * Dinv @ b - omega * Dinv @ A @ x
    return x

def mg(P, A, b, x, omega=0.666):
    x = jacobi(A, b, x, omega)
    AH = P.T@A@P
    rH = P.T@(b-A@x)
    x += P@spla.spsolve(AH, rH)
    x = jacobi(A, b, x, omega)
    return x

def mgv(P, A, b, x_ref, omega=0.666, tol=1e-10):
    err = []
    n = A.shape[0]
    N = int(n ** 0.5)
    x = np.zeros(n)

    for i in range(50):
        x += mg(P, A, b - A@x, np.zeros(n), omega)
        e = la.norm(x_ref - x, np.inf)
        err.append(e)
        if e < tol:
            break

    err = np.array(err)
    conv_factor = np.mean(err[1:] / err[:-1])
    return conv_factor

def create_interp(grid):
    N = grid.shape[0]
    G = grid.reshape(N**2)
    return pyamg.classical.direct_interpolation(A,S,G.astype('intc'))

def det_conv_factor_optimal_omega(P, A, b, x_ref):
    def obj(omega):
        return mgv(P, A, b, x_ref, omega)
    opt = scipy.optimize.minimize_scalar(obj, (0, 1), bounds=(0, 1), method='bounded', options={'maxiter': 50})
    return opt.fun, opt.x

# grid generation

coarsenings = [ref_all_fine, ref_all_coarse, ref_amg, ref_coarsen_by(2), ref_coarsen_by(3), ref_coarsen_by(4), ref_coarsen_by(5)]
p_trials = np.array([0.01, 0.05, 0.10, 0.25, 0.50, 0.75])
trials = []

for ci, c in enumerate(coarsenings):
    name, G = c(A)
    print(f'  Generating grids from: {name}')

    for p in p_trials:
        print(f'    Running trials for p={p:.2f}')

        rates = np.zeros(I)
        omegas = np.zeros(I)
        grids = np.zeros((I, N, N))

        for i in range(I):
            def permute_grid():
                # Automatically try again for degenerate cases
                while True:
                    try:
                        perm_G = G.copy()
                        for j in range(N):
                            for k in range(N):
                                if np.random.rand(1) < p:
                                    perm_G[j,k] = not perm_G[j,k]
                        P = create_interp(perm_G)
                        break
                    except Exception as e:
                        pass # restart

                return perm_G, P

            # Create the randomly permuted "perm_C"
            perm_G, P = permute_grid()
            best_conv, best_omega = det_conv_factor_optimal_omega(P, A, b, x_ref)

            # run multigrid iterations
            grids[i] = (perm_G*2)-1
            rates[i] = best_conv
            omegas[i] = best_omega

        trials.append((grids, rates, omegas))

print('finished trials')

grids = np.concatenate([ t[0] for t in trials ])
rates = np.concatenate([ t[1] for t in trials ])
omegas = np.concatenate([ t[2] for t in trials ])

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
]

for o in outputs:
    var = o['var']
    fname = o['fname']
    try:
        existing = helpers.pickle_load(fname)
        existing = np.concatenate([ np.array(existing), var ])
    except Exception as e:
        existing = var
    helpers.pickle_save(fname, existing)

t_end = time.time()
print(f'finished in {int(t_end-t_start)} seconds')
