from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
#from moist_euler_dg.two_phase_euler_2D import TwoPhaseEuler2D
from moist_euler_dg.fortran_two_phase_euler_2D import FortranTwoPhaseEuler2D as TwoPhaseEuler2D
import numpy as np
import time
import os
import argparse
from mpi4py import MPI


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, help='Number of cells')
parser.add_argument('--nproc', type=int, help='Number of procs', default=1)
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

xlim = 10_000
zlim = 10_000

nz = args.n
nproc = args.nproc
run_model = (not args.plot) # whether to run model - set false to just plot previous run
nx = nz
cfl = 0.5
g = 9.81
poly_order = 3


exp_name_short = 'bryan-fritsch-bubble'
experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{poly_order}'
data_dir = os.path.join('data', experiment_name)
plot_dir = os.path.join('plots', experiment_name)

if rank == 0:
    print(f"---------- Moist bubble with nx={nx}, nz={nz}, cfl={cfl}")
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    if not os.path.exists(data_dir): os.makedirs(data_dir)

comm.barrier()
#

zmap = lambda x, z: z * zlim
xmap = lambda x, z: xlim * (x - 0.5)


def initial_condition(xs, ys, solver, pert):

    u = 0 * ys
    v = 0 * ys

    # compute ground values
    density_ground = 1.2
    p_ground = 1_00_000.0
    qw_ground = 0.0196 # 0.02

    qv_ground = solver.solve_qv_from_p(density_ground, qw_ground, p_ground)
    R_ground = (1 - qw_ground) * solver.Rd + qv_ground * solver.Rv
    cp_ground = (1 - qw_ground) * solver.cpd + qv_ground * solver.cpv + (qw_ground - qv_ground) * solver.cl
    T_ground = p_ground / (density_ground * R_ground)

    entropy_ground = (1 - qw_ground) * solver.entropy_air(T_ground, 1 - qw_ground, density_ground)
    entropy_ground += qv_ground * solver.entropy_vapour(T_ground, qv_ground, density_ground)
    entropy_ground += (qw_ground - qv_ground) * solver.entropy_liquid(T_ground)
    # print(f'Background moist entropy: {entropy_ground} K')

    enthalpy_ground = cp_ground * T_ground + qv_ground * solver.Lv0

    # compute profiles
    enthalpy = enthalpy_ground - solver.g * ys
    s = entropy_ground + 0 * ys
    qw = qw_ground + 0 * ys


    qv = solver.solve_qv_from_enthalpy(enthalpy, qw, s, verbose=False)

    qd = 1 - qw
    ql = qw - qv
    R = qv * solver.Rv + qd * solver.Rd
    cv = qd * solver.cvd + qv * solver.cvv + ql * solver.cl
    cp = qd * solver.cpd + qv * solver.cpv + ql * solver.cl
    T = (enthalpy - qv * solver.Lv0) / cp
    logdensity = (1 / R) * (cv * np.log(T) - s - qd * solver.Rd * np.log(solver.Rd * qd)
                            - qv * solver.Rv * (np.log(qv) + np.log(solver.Rv)) + qv * solver.c0 + ql * solver.c1
                            )
    density = np.exp(logdensity)
    p = density * R * T

    ql0 = np.copy(ql)
    qv0 = np.copy(qv)
    qw0 = np.copy(qw)

    rad_max = 2_000
    rad = np.sqrt(xs ** 2 + (ys - 1.0*rad_max) ** 2)
    mask = rad < rad_max
    density -= mask * (pert * density / 300) * (np.cos(np.pi * (rad / rad_max) / 2)**2)

    qv = solver.solve_qv_from_p(density, qw, p)
    R = (1 - qw) * solver.Rd + qv * solver.Rv
    cp = (1 - qw) * solver.cpd + qv * solver.cpv + (qw - qv) * solver.cl
    T = p / (density * R)

    s = (1 - qw) * solver.entropy_air(T, 1 - qw, density)
    s += qv * solver.entropy_vapour(T, qv, density)
    s += (qw - qv) * solver.entropy_liquid(T)

    # print('T min-max:', T.min() - 273, T.max() - 273)
    # print('Density min-max:', density.min(), density.max())
    # print('Pressure min-max:', p.min(), p.max())
    # print('qv min-max:', qv.min(), qv.max(), '\n')

    return u, v, density, s, qw, qv

tends = np.array([400, 800, 1000, 1200])
a = 0.5
if run_model:
    solver = TwoPhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=cfl, a=a, nz=nz, upwind=True, nprocx=nproc)
    u, v, density, s, qw, qv = initial_condition(solver.xs, solver.zs, solver, pert=2.0)
    solver.set_initial_condition(u, v, density, s, qw)
    exit(0)

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
elif rank == 0:
    plt.rcParams['font.size'] = '12'

    solver_plot = TwoPhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=cfl, a=a, nz=nz, upwind=True, nprocx=1)
    _, _, h0, s0, qw0, qv0 = initial_condition(solver_plot.xs, solver_plot.zs, solver_plot, pert=0.0)
    ql0 = qw0 - qv0

    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    plot_func_entropy = lambda s: s.project_H1(s.s)
    plot_func_density = lambda s: s.project_H1(s.h - h0)
    plot_func_water = lambda s: s.project_H1(s.q - qw0)
    plot_func_vapour = lambda s: s.project_H1(s.solve_qv_from_entropy(s.h, s.q, s.s) - qv0)
    plot_func_liquid = lambda s: s.project_H1(s.q - s.solve_qv_from_entropy(s.h, s.q, s.s) - ql0)

    fig_list = [plt.subplots(2, 2, sharex=True, sharey=True) for _ in range(5)]

    pfunc_list = [
        plot_func_entropy, plot_func_density,
        plot_func_water, plot_func_vapour, plot_func_liquid
    ]

    labels = ["entropy", "density", "water", "vapour", "liquid"]

    energy = []
    for i, tend in enumerate(tends):
        filepaths = [solver_plot.get_filepath(data_dir, exp_name_short, proc=i, nprocx=nproc, time=tend) for i in range(nproc)]
        solver_plot.load(filepaths)
        energy.append(solver_plot.integrate(solver_plot.energy()))

        for (fig, axs), plot_fun in zip(fig_list, pfunc_list):
            ax = axs[i // 2][i % 2]
            ax.tick_params(labelsize=8)
            im = solver_plot.plot_solution(ax, dim=2, plot_func=plot_fun)
            cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt))
            cbar.ax.tick_params(labelsize=8)


    for (fig, ax), label in zip(fig_list, labels):
        plot_name = f'{label}_{exp_name_short}'
        fp = solver_plot.get_filepath(plot_dir, plot_name, ext='png')
        fig.savefig(fp, bbox_inches="tight")
