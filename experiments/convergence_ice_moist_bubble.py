from matplotlib import pyplot as plt
from moist_euler_dg.fortran_three_phase_euler_2D import FortranThreePhaseEuler2D as ThreePhaseEuler2D
import numpy as np
import os
from moist_euler_dg import utils
import scipy


exp_name_short = 'smooth-ice-bubble'
order = 3
nproc = 4

xlim = 10_000
zlim = 10_000
cfl = 0.5
g = 9.81
a = 0.5
upwind = True
zmap = lambda x, z: z * zlim
xmap = lambda x, z: xlim * (x - 0.5)


def get_solver(n, order, nproc):
    nx = nz = n

    experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{order}'
    data_dir = os.path.join('data', experiment_name)

    solver = ThreePhaseEuler2D(xmap, zmap, order, nx, g=g, cfl=0.5, a=a, nz=nz, upwind=upwind, nprocx=1)

    tend = 600
    filepaths = [solver.get_filepath(data_dir, exp_name_short, proc=i, nprocx=nproc, time=tend)
                 for i in range(nproc)]
    solver.load(filepaths)

    return solver


def to_multigrid_points_2D(order):

    x_in, _ = utils.gll(order, iterative=True)
    y_in, _ = utils.gll(order, iterative=True)

    x_out_1 = (x_in - 1) * 0.5
    x_out_2 = (x_in + 1) * 0.5
    y_out_1 = (y_in - 1) * 0.5
    y_out_2 = (y_in + 1) * 0.5

    interp_mats = [np.zeros((order+1, order+1, order+1, order+1)) for _ in range(4)]

    # interp_mat[i, j, k, l] = kl basis at ij point

    for i in range(len(y_out_1)):
        for j in range(len(x_out_1)):
            for k in range(len(y_in)):
                for l in range(len(x_in)):

                    #

                    y_data = np.zeros_like(y_in)
                    y_data[k] = 1.0
                    y_poly = scipy.interpolate.lagrange(y_in, y_data)

                    x_data = np.zeros_like(x_in)
                    x_data[l] = 1.0
                    x_poly = scipy.interpolate.lagrange(x_in, x_data)

                    interp_mats[0][i, j, k, l] = y_poly(y_out_1[i]) * x_poly(x_out_1[j])
                    interp_mats[1][i, j, k, l] = y_poly(y_out_1[i]) * x_poly(x_out_2[j])

                    interp_mats[2][i, j, k, l] = y_poly(y_out_2[i]) * x_poly(x_out_1[j])
                    interp_mats[3][i, j, k, l] = y_poly(y_out_2[i]) * x_poly(x_out_2[j])

    return interp_mats

interp_mats = to_multigrid_points_2D(order)


def refine(arr_in, interp_mats):
    order = arr_in.shape[2] - 1
    n = order + 1
    cellsx, cellsz = arr_in.shape[:2]
    out_shape = (2 * cellsx, 2 * cellsz, n, n)
    arr_out = np.zeros(out_shape)

    arr_out[::2, ::2] = np.einsum('abcd,kjcd->kjab', interp_mats[0], arr_in)
    arr_out[::2, 1::2] = np.einsum('abcd,kjcd->kjab', interp_mats[1], arr_in)
    arr_out[1::2, ::2] = np.einsum('abcd,kjcd->kjab', interp_mats[2], arr_in)
    arr_out[1::2, 1::2] = np.einsum('abcd,kjcd->kjab', interp_mats[3], arr_in)

    return arr_out


ns = np.array([8, 16, 32, 64])
solvers = [get_solver(n, order=order, nproc=n//2) for n in ns]
n = 128
ref_solver = get_solver(n, order=order, nproc=n//2)

var_funcs = [lambda s: s.u, lambda s: s.w, lambda s: s.h, lambda s: s.s, lambda s: s.q]
labels = ['u', 'w', 'density', 'entropy', 'water']

max_val = -np.inf
min_val = np.inf

for var_func, label in zip(var_funcs, labels):
    errors = []
    norm = np.sqrt(ref_solver.integrate(var_func(ref_solver) ** 2))
    for solver in solvers:

        arr = var_func(solver)

        while (arr.shape[0] < ref_solver.xs.shape[0]):
            arr = refine(arr, interp_mats)

        assert arr.shape == ref_solver.xs.shape

        error = np.sqrt(ref_solver.integrate((arr - var_func(ref_solver)) ** 2))
        error /= norm
        errors.append(error)

        if errors[0] > max_val:
            max_val = errors[0]

        if errors[0] < min_val:
            min_val = errors[0]


    plt.loglog(ns, errors, label=label)

start_val = np.exp(0.5 * (np.log(min_val) + np.log(max_val)))
plt.loglog(ns, start_val * ns[0] ** 3 * (ns * 1.0) ** (-3), '--', label='3rd order')
plt.loglog(ns, start_val * ns[0] ** 2 * (ns * 1.0) ** (-2), '--', label='2nd order')
plt.legend()
plt.savefig('plots/convergence-ice-moist-bubble.png')