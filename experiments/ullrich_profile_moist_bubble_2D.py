from matplotlib import pyplot as plt
import matplotlib.ticker as ticker
from moist_euler_dg.three_phase_euler_2D import TwoPhaseEuler2D
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
args = parser.parse_args()

run_model = True # whether to run model - set false to just plot previous run
xlim = 20_000
zlim = 10_000

nz = args.n
nx = 2 * nz
eps = 0.8
g = 9.81
poly_order = 3

exp_name_short = 'ullrich-two-phase-bubble'
experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{poly_order}'
data_dir = os.path.join('data', experiment_name)
plot_dir = os.path.join('plots', experiment_name)

if rank == 0:
    print(f"---------- Ullrich bubble with nx={nx}, nz={nz}, cfl={eps}")
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    if not os.path.exists(data_dir): os.makedirs(data_dir)

comm.barrier()
#

zmap = lambda x, z: z * zlim
xmap = lambda x, z: xlim * (x - 0.5)

solver = TwoPhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=0.5, nz=nz, upwind=True, nprocx=size)

def initial_condition(xs, ys, solver, pert):

    u = 0 * ys
    v = 0 * ys

    # set hydrostatic pressure/density profile
    TE = 310.0
    TP = 240.0
    GRAVITY = solver.g
    T0 = 0.5*(TE+TP)
    b = 2.0
    KP = 3.0
    GAMMA = 0.005
    P0 = 100000
    RD = solver.Rd

    A = 1.0/GAMMA
    B = (TE - TP)/((TE + TP)*TP)
    C = 0.5*(KP + 2.0)*(TE - TP)/(TE*TP)
    H = RD*T0/GRAVITY

    fac   = ys/(b*H)
    fac2  = fac*fac
    cp    = np.cos(2.0*np.pi/9.0)
    cpk   = np.power(cp, KP)
    cpkp2 = np.power(cp, KP+2)
    fac3  = cpk - (KP/(KP+2.0))*cpkp2

    torr_1 = (A*GAMMA/T0)*np.exp(GAMMA*(ys)/T0) + B*(1.0 - 2.0*fac2)*np.exp(-fac2)
    torr_2 = C*(1.0 - 2.0*fac2)*np.exp(-fac2)

    int_torr_1 = A*(np.exp(GAMMA*ys/T0) - 1.0) + B*ys*np.exp(-fac2)
    int_torr_2 = C*ys*np.exp(-fac2)

    tempInv = torr_1 - torr_2*fac3
    T = 1.0 / tempInv
    p = P0*np.exp(-GRAVITY*int_torr_1/RD + GRAVITY*int_torr_2*fac3/RD)
    density = p / (solver.Rd * T)

    # add buoyancy perturbation
    rad_max = 2_000
    rad = np.sqrt(xs ** 2 + (ys - 1.0 * rad_max) ** 2)
    mask = rad < rad_max
    density -= mask * (pert * density / 300) * (np.cos(np.pi * (rad / rad_max) / 2) ** 2)

    # set moisture
    qw = 0.02 + 0.0 * ys
    qv = solver.solve_qv_from_p(density, qw, p)
    qd = 1 - qw
    ql = qw - qv
    R = qv * solver.Rv + qd * solver.Rd

    # calculate entropy
    T = p / (density * R)
    s = qd * solver.entropy_air(T, qd, density)
    s += qv * solver.entropy_vapour(T, qv, density)
    s += ql * solver.entropy_liquid(T)

    return u, v, density, s, qw, qv


u, v, density, s, qw, qv = initial_condition(solver.xs, solver.zs, solver, pert=2.0)
solver.set_initial_condition(u, v, density, s, qw)

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

        solver.save(solver.get_filepath(data_dir, exp_name_short))

# plotting
if rank == 0:
    plt.rcParams['font.size'] = '12'

    solver_plot = TwoPhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=0.5, nz=nz, upwind=True, nprocx=1)
    _, _, _, s0, qw0, qv0 = initial_condition(solver_plot.xs, solver_plot.zs, solver_plot, pert=0.0)
    ql0 = qw0 - qv0

    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    plot_func_entropy = lambda s: s.project_H1(s.s - s0)
    plot_func_density = lambda s: s.project_H1(s.h)
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
        filepaths = [solver_plot.get_filepath(data_dir, exp_name_short, proc=i, nprocx=size, time=tend) for i in range(size)]
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