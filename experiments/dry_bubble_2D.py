from matplotlib import pyplot as plt
import matplotlib.animation as animation
from moist_euler_dg.euler_2D import Euler2D
import numpy as np
import time
import os
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

run_model = True # if false just plot most recent run
xlim = 20_000
zlim = 10_000

nz = 10
nx = 2 * nz
eps = 1.4
g = 9.81
poly_order = 3

# make data and plotting directories
experiment_name = f'dry-bubble-nx-{nx}-nz-{nz}-p{poly_order}'
data_dir = os.path.join('data', experiment_name)
plot_dir = os.path.join('plots', experiment_name)

if rank == 0:
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    if not os.path.exists(data_dir): os.makedirs(data_dir)

comm.barrier()

#
zmap = lambda x, z: z * zlim
xmap = lambda x, z: xlim * (x - 0.5)

solver = Euler2D(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=0.5, nz=nz, upwind=True, nprocx=size)

def initial_condition(xs, ys, R, cp, cv, p0, g, gamma):

    u = 0 * xs
    v = 0 * xs
    b = 300

    dexdy = -g / 300

    # p_ground = (b * R * density_ground) ** gamma * (1 / p0) ** (R / cv)
    p_ground = 100_000.0
    density_ground = (p_ground / ((1 / p0) ** (R / cv))) ** (1 / gamma) / (300 * R)

    const = cp * (R * 300 / p0) ** (R / cv)
    ex0 = const * density_ground ** (R / cv) # surface density is 1.2 kg/m^3
    ex = ex0 + ys * dexdy

    density = (ex / const) ** (cv / R)
    p = (b * R * density) ** gamma * (1 / p0) ** (R / cv)

    R_max = 2_000.0
    rad = np.sqrt(xs ** 2 + (ys - R_max) ** 2)
    mask = rad < R_max
    pert = -(2.0 / 300.0) * density
    density += pert * mask * 0.5 * (1 + np.cos(np.pi * rad / R_max))
    # b += mask * 2 * np.cos(0.5 * np.pi * rad / R_max)**2

    # p = (b * R * density)**gamma * (1 / p0)**(R / cv)
    s = cv * np.log(p / density**gamma)

    # print('Pressure min:', p.min(), p.max())
    # print('b min-max:', b.min(), b.max())
    return u, v, density, s * density

u, v, density, hs = initial_condition(
    solver.xs,
    solver.zs,
    solver.R,
    solver.cp,
    solver.cv,
    solver.p0,
    solver.g,
    solver.gamma
)
solver.set_initial_condition(u, v, density, hs / density)

tends = np.array([0.0, 400, 800, 1000])

if run_model:
    for i, tend in enumerate(tends):
        t0 = time.time()
        while solver.time < tend:
            dt = min(solver.get_dt(), tend - solver.time)
            solver.time_step(dt=dt)
        t1 = time.time()

        if rank == 0:
            print("Simulation time (unit less):", solver.time)
            print("Wall time:", time.time() - t0, '\n')

        solver.save(solver.get_filepath(data_dir, experiment_name))

if rank == 0:
    plot_func = lambda s: s.project_H1(s.hs / s.h)
    plt.rcParams['font.size'] = '12'
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

    solver_plot = Euler2D(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=0.5, nz=nz, upwind=True, nprocx=1)

    energy = []
    for i, tend in enumerate(tends):

        filepaths = [solver_plot.get_filepath(data_dir, experiment_name, proc=i, nprocx=size, time=tend) for i in range(size)]
        solver_plot.load(filepaths)
        energy.append(solver_plot.integrate(solver_plot.energy()))

        ax = axs[i // 2][i % 2]
        ax.set_xlim(-0.25 * xlim, 0.25 * xlim)

        im = solver_plot.plot_solution(ax, dim=2, plot_func=plot_func)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)

    print("Relative energy change:", (energy[-1] - energy[0]) / energy[0])
    plt.savefig(solver_plot.get_filepath(plot_dir, experiment_name, ext='png'))
    plt.show()

