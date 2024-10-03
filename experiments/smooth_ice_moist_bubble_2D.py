from matplotlib import pyplot as plt
# from moist_euler_dg.three_phase_euler_2D import ThreePhaseEuler2D
from moist_euler_dg.fortran_three_phase_euler_2D import FortranThreePhaseEuler2D as ThreePhaseEuler2D
from moist_euler_dg.one_phase_euler_2D import OnePhaseEuler2D
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
parser.add_argument('--o', type=int, help='Polynomial order')
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
poly_order = args.o
a = 0.5
upwind = True

exp_name_short = 'smooth-ice-bubble'
experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{poly_order}'
data_dir = os.path.join('data', experiment_name)
plot_dir = os.path.join('plots', experiment_name)

if rank == 0:
    print(f"---------- Smooth ice bubble with nx={nx}, nz={nz}, cfl={cfl}")
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    if not os.path.exists(data_dir): os.makedirs(data_dir)

comm.barrier()
#

zmap = lambda x, z: z * zlim
xmap = lambda x, z: xlim * (x - 0.5)

def initial_condition(xs, ys, solver, pert):

    u = 0 * ys
    v = 0 * ys

    dry_theta = 300
    dexdy = -g / (solver.cpd * dry_theta)
    ex = 1 + dexdy * ys
    p = 1_00_000.0 * ex ** (solver.cpd / solver.Rd)
    density = p / (solver.Rd * ex * dry_theta)

    # qw = solver.rh_to_qw(0.95, p, density)
    T = p / (solver.Rd * density)
    qw = solver.liq_saturation_fraction(T, density)
    # qw = 0 * qw + 1e-10
    qd = 1 - qw

    R = solver.Rd * qd + solver.Rv * qw
    T = p / (R * density)

    # assert (qw <= solver.saturation_fraction(T, density)).all()

    rad_max = 2_000
    rad = np.sqrt(xs ** 2 + (ys - 2.0 * rad_max) ** 2)
    # mask = rad < rad_max
    # density -= mask * (pert * density / 300) * (np.cos(np.pi * (rad / rad_max) / 2) ** 2)
    density -= (pert * density / 300) * np.exp(-(2 * rad / rad_max) ** 2)

    T = p / (R * density)
    # assert (qw <= solver.saturation_fraction(T, density)).all()

    s = qd * solver.entropy_air(T, qd, density)
    s += qw * solver.entropy_vapour(T, qw, density)

    qv, ql, qi = solver.solve_fractions_from_entropy(density, qw, s)
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

tends = np.array([0.0, 200.0, 300.0, 400.0])
# tends = np.array([0.0, 200.0, 400.0, 479.0])

conservation_data_fp = os.path.join(data_dir, 'conservation_data.npy')
time_list = []
energy_list = []
entropy_var_list = []
water_var_list = []

if run_model:
    solver = OnePhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=cfl, a=a, nz=nz, upwind=upwind, nprocx=nproc)
    u, v, density, s, qw, qv, ql, qi = initial_condition(solver.xs, solver.zs, solver, pert=2.0)
    solver.set_initial_condition(u, v, density, s, qw)
    for i, tend in enumerate(tends):
        t0 = time.time()
        while solver.time < tend:
            time_list.append(solver.time)
            energy_list.append(solver.energy())
            entropy_var_list.append(solver.integrate(solver.h * solver.s**2))
            water_var_list.append(solver.integrate(solver.h * solver.q**2))

            dt = min(solver.get_dt(), tend - solver.time)
            solver.time_step(dt=dt)
        t1 = time.time()

        if rank == 0:
            print("Simulation time (unit less):", solver.time)
            print("Wall time:", time.time() - t0, '\n')

        solver.save(solver.get_filepath(data_dir, exp_name_short))

    if rank == 0:
        conservation_data = np.zeros((4, len(time_list)))
        conservation_data[0, :] = np.array(time_list)
        conservation_data[1, :] = np.array(energy_list)
        conservation_data[2, :] = np.array(entropy_var_list)
        conservation_data[3, :] = np.array(water_var_list)
        np.save(conservation_data_fp, conservation_data)

        print('Energy error:', (energy_list[-1] - energy_list[0]) / energy_list[0])

    print('Time of first limit:', solver.first_water_limit_time)
        

# plotting
elif rank == 0:
    plt.rcParams['font.size'] = '12'

    conservation_data = np.load(conservation_data_fp)
    time_list = conservation_data[0, :]
    mask = time_list <= np.inf
    energy_list = conservation_data[1, :][mask]
    entropy_var_list = conservation_data[2, :][mask]
    water_var_list = conservation_data[3, :][mask]
    time_list = time_list[mask]

    e_diff = abs(np.diff(energy_list))
    print('Time max energy growth:', time_list[np.argmax(e_diff) + 1])
    
    energy_list = (energy_list - energy_list[0]) / energy_list[0]
    entropy_var_list = (entropy_var_list - entropy_var_list[0]) / entropy_var_list[0]
    water_var_list = (water_var_list - water_var_list[0]) / water_var_list[0]

    print('Energy error:', energy_list[-1])
    print('Entropy var error:', entropy_var_list[-1])
    print('Water var error:', water_var_list[-1])

    plt.figure()
    plt.plot(time_list, energy_list, label='Energy')
    plt.plot(time_list, entropy_var_list, label='Entropy variance')
    plt.plot(time_list, water_var_list, label='Water variance')
    plt.grid()
    plt.legend()
    plt.yscale('symlog', linthresh=1e-15)
    fp = os.path.join(plot_dir, f'conservation_{exp_name_short}')
    plt.savefig(fp, bbox_inches="tight")

    solver_plot = OnePhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=a, nz=nz, upwind=upwind, nprocx=1)
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

