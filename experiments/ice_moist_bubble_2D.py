from matplotlib import pyplot as plt
from moist_euler_dg.three_phase_euler_2D import ThreePhaseEuler2D
import numpy as np
import time
import os
import argparse
from mpi4py import MPI
import matplotlib.ticker as ticker


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, help='Number of cells')
args = parser.parse_args()

run_model = True # whether to run model - set false to just plot previous run
xlim = 20_000
zlim = 10_000

nz = args.n
nx = 2 * nz
eps = 0.8
g = 9.81
poly_order = 3

exp_name_short = 'ice-bubble'
experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{poly_order}'
data_dir = os.path.join('data', experiment_name)
plot_dir = os.path.join('plots', experiment_name)

if rank == 0:
    print(f"---------- Ice bubble with nx={nx}, nz={nz}, cfl={eps}")
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    if not os.path.exists(data_dir): os.makedirs(data_dir)

comm.barrier()
#

zmap = lambda x, z: z * zlim
xmap = lambda x, z: xlim * (x - 0.5)

solver = ThreePhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=1.0, a=0.5, nz=nz, upwind=True, nprocx=size)



def initial_condition(xs, ys, solver, pert):

    u = 0 * ys
    v = 0 * ys

    dry_theta = 300
    dexdy = -g / (solver.cpd * dry_theta)
    ex = 1 + dexdy * ys
    p = 1_00_000.0 * ex ** (solver.cpd / solver.Rd)
    density = p / (solver.Rd * ex * dry_theta)

    qw = solver.rh_to_qw(0.95, p, density)
    qd = 1 - qw

    R = solver.Rd * qd + solver.Rv * qw
    T = p / (R * density)

    assert (qw <= solver.saturation_fraction(T, density)).all()

    rad_max = 2_000
    rad = np.sqrt(xs ** 2 + (ys - 1.0 * rad_max) ** 2)
    mask = rad < rad_max
    density -= mask * (pert * density / 300) * (np.cos(np.pi * (rad / rad_max) / 2) ** 2)

    T = p / (R * density)
    assert (qw <= solver.saturation_fraction(T, density)).all()

    s = qd * solver.entropy_air(T, qd, density)
    s += qw * solver.entropy_vapour(T, qw, density)

    qv, ql, qi = solver.solve_fractions_from_entropy(density, qw, s, verbose=True)
    #  0.3410208713540216 0.10594892674155956 0.6589791286459784
    # print('qw min-max:', qw.min(), qw.max())
    # print('T min-max:', T.min() - 273, T.max() - 273)
    # print('Density min-max:', density.min(), density.max())
    # print('Pressure min-max:', p.min(), p.max())
    # print('qv/qw min-max:', (qv/qw).min(), (qv/qw).max())
    # print('all vapour mean:', (qv == qw).mean())
    # print('ql/qw min-max:', (ql/qw).min(), (ql/qw).max())
    # print('qi/qw min-max:', (qi/qw).min(), (qi/qw).max(), '\n')

    return u, v, density, s, qw, qv, ql, qi


run_time = 600

u, v, density, s, qw, qv, ql, qi = initial_condition(
    solver.xs,
    solver.zs,
    solver,
    pert=2.0
)
solver.set_initial_condition(u, v, density, s, qw)

tends = np.array([0.0, 200.0, 400.0, 600.0])

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

        solver.save(solver.get_filepath(data_dir, exp_name_short))

# plotting
if rank == 0:
    plt.rcParams['font.size'] = '12'

    solver_plot = ThreePhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=0.5, nz=nz, upwind=True, nprocx=1)
    _, _, _, s0, qw0, qv0, ql0, qi0 = initial_condition(solver_plot.xs, solver_plot.zs, solver_plot, pert=0.0)

    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    plot_func_entropy = lambda s: s.project_H1(s.s - s0)
    plot_func_density = lambda s: s.project_H1(s.h)
    plot_func_water = lambda s: s.project_H1(s.q - qw0)
    plot_func_vapour = lambda s: s.project_H1(s.solve_fractions_from_entropy(s.h, s.q, s.s)[0] - qv0)
    plot_func_liquid = lambda s: s.project_H1(s.solve_fractions_from_entropy(s.h, s.q, s.s)[1] - ql0)
    plot_func_ice = lambda s: s.project_H1(s.solve_fractions_from_entropy(s.h, s.q, s.s)[2] - qi0)

    fig_list = [plt.subplots(2, 2, sharex=True, sharey=True) for _ in range(6)]

    pfunc_list = [
        plot_func_entropy, plot_func_density,
        plot_func_water, plot_func_vapour, plot_func_liquid, plot_func_ice
    ]

    labels = ["entropy", "density", "water", "vapour", "liquid", "ice"]

    energy = []
    for i, tend in enumerate(tends):
        filepaths = [solver_plot.get_filepath(data_dir, exp_name_short, proc=i, nprocx=size, time=tend) for i in range(size)]
        solver_plot.load(filepaths)
        energy.append(solver_plot.integrate(solver_plot.energy()))

        for (fig, axs), plot_fun in zip(fig_list, pfunc_list):
            ax = axs[i // 2][i % 2]
            ax.tick_params(labelsize=8)
            ax.set_xlim(-0.25 * xlim, 0.25 * xlim)
            im = solver_plot.plot_solution(ax, dim=2, plot_func=plot_fun)
            cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt))
            cbar.ax.tick_params(labelsize=8)


    for (fig, ax), label in zip(fig_list, labels):
        plot_name = f'{label}_{exp_name_short}'
        fp = solver_plot.get_filepath(plot_dir, plot_name, ext='png')
        fig.savefig(fp, bbox_inches="tight")

