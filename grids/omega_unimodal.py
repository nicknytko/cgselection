import numpy as np
import numpy.linalg as la
import scipy.linalg as sla
import scipy.optimize as sopt
import scipy.interpolate as sinterp
import pickle
import sys
import matplotlib.pyplot as plt

sys.path.append('../lib')
import helpers

N = 31

# Set up 1D poisson system
A = helpers.gen_1d_poisson_fd(N)
x = np.linspace(-1, 1, N)
u = helpers.random_u(N, 0.0055)

#grid_C, grid_F = helpers.grid_from_coarsening_factor(N, 3)
grid_C = helpers.random_grid(N)

omegas = np.linspace(0.001, 0.999, 1000)
convs = np.zeros_like(omegas)

for i, omega in enumerate(omegas):
    convs[i] = helpers.det_conv_fact(A, grid_C, x, u, omega)

def obj_func(omega):
    return helpers.det_conv_fact(A, grid_C, x, u, omega)

conv_opt = sopt.minimize_scalar(obj_func, (0, 1), options={'maxiter': 12})

plt.plot(omegas, convs)
plt.plot(conv_opt.x, conv_opt.fun, 'o')
plt.title('Unimodality of Jacobi weight vs convergence factor')
plt.xlabel('Jacobi weight')
plt.ylabel('Convergence factor (smaller is better)')
plt.xlim(0,1)
plt.ylim(0,1)
plt.show(block=True)
