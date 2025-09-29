from matplotlib import pyplot as plt
from moist_euler_dg.fortran_two_phase_euler_2D import FortranTwoPhaseEuler2D as TwoPhaseEuler
import numpy as np
import time
import os
import argparse
from mpi4py import MPI
import matplotlib.ticker as ticker
import scipy


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, help='Number of vertical cells')
parser.add_argument('--nx', type=int, help='Number of horizontal cells')
parser.add_argument('--o', type=int, help='Polynomial order')
parser.add_argument('--nproc', type=int, help='Number of procs', default=1)
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

xlim = 300_000
zlim = 10_000
# maps to define geometry these can be arbitrary - maps [0, 1]^2 to domain
zmap = lambda x, z: z * zlim
xmap = lambda x, z: xlim * x #* (x - 0.5)

nz = args.nz
nx = args.nx
nproc = args.nproc
run_model = (not args.plot) # whether to run model - set false to just plot previous run
poly_order = args.o

cfl = 0.5
g = 9.81
a = 0.5
upwind = True

exp_name_short = 'dry-gravity-wave'
experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{poly_order}'
data_dir = os.path.join('data', experiment_name)
plot_dir = os.path.join('plots', experiment_name)

if rank == 0:
    print(f"---------- Moist gravity wave with nx={nx}, nz={nz}, cfl={cfl}")
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    if not os.path.exists(data_dir): os.makedirs(data_dir)

comm.barrier()


def dry_density_profile(solver, zs):

    R, cv, cp, g = solver.Rd, solver.cvd, solver.cpd, solver.g
    p0 = 100_000
    T0 = 300.0

    N = 0.01
    N2 = N ** 2

    pot_temp_profile = T0 * np.exp(N2 * zs / g)
    ex = 1 + (g ** 2 / (cp * T0 * N2)) * (np.exp(-N2 * zs / g) - 1)
    density_profile = (p0 / (R * pot_temp_profile)) * ex ** (cv / R)

    return density_profile



def initial_condition(solver, pert):
    # solve for rho s.t: (d/dz) p(rho, s, qw) + g * rho = 0

    # setup moisture and moist potential temperature/entropy profiles
    mpt_sfc = 300.0 # sft moist potential temp
    N = 0.01 # Brunt-Vaisala frequency
    qw_sfc = 1e-12 # constant water mass fraction
    density = dry_density_profile(solver, solver.zs)

    mpt_profile = mpt_sfc * np.exp(N ** 2 * solver.zs / solver.g)
    qw = qw_sfc + np.zeros_like(mpt_profile)

    # add perturbation to moist potential temperature
    hc = 10_000.0
    xc = 150_000.0
    ac = 5000
    mpt_pert = pert * np.sin(np.pi * solver.zs / hc) / (1 + ((solver.xs - xc) / ac) ** 2)

    mpt = mpt_profile + mpt_pert
    s = solver.moist_potential_temperature_to_entropy(mpt, qw)

    u = 20 + 0 * s
    v = 0 * s

    return u, v, density, s, qw, mpt_profile


tends = np.array([0.0, 1200, 2400, 3600])

conservation_data_fp = os.path.join(data_dir, 'conservation_data.npy')
time_list = []
energy_list = []
entropy_var_list = []
water_var_list = []

if run_model:
    solver = TwoPhaseEuler(xmap, zmap, poly_order, nx, g=g, cfl=cfl, a=a, nz=nz, upwind=upwind, nprocx=nproc)
    u, v, density, s, qw, mpt_profile = initial_condition(solver, pert=0.01)
    solver.set_initial_condition(u, v, density, s, qw)

    for i, tend in enumerate(tends):
        t0 = time.time()
        while solver.time < tend:
            time_list.append(solver.time)
            energy_list.append(solver.energy())
            entropy_var_list.append(solver.integrate(solver.h * solver.s ** 2))
            water_var_list.append(solver.integrate(solver.h * solver.q ** 2))

            dt = min(solver.get_dt(), tend - solver.time)
            solver.time_step(dt=dt)
        t1 = time.time()

        if rank == 0:
            print("Simulation time (s):", solver.time)
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

elif rank == 0:
    plt.rcParams['font.size'] = '12'

    solver_plot = TwoPhaseEuler(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=a, nz=nz, upwind=upwind, nprocx=1)
    u0, v0, density0, s0, qw0, mpt_profile = initial_condition(solver_plot, pert=0.0)

    mpt0 = solver_plot.moist_potential_temperature(s0, qw0)
    qv0 = solver_plot.solve_qv_from_entropy(density0, qw0, s0)
    ql0 = qw0 - qv0

    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    plot_func_mpt = lambda s: s.project_H1(s.moist_potential_temperature(s.s, s.q) - mpt0)
    plot_func_entropy = lambda s: s.project_H1(s.s - s0)
    plot_func_density = lambda s: s.project_H1(s.h)
    plot_func_water = lambda s: s.project_H1(s.q - qw0)
    plot_func_vapour = lambda s: s.project_H1(s.solve_qv_from_entropy(s.h, s.q, s.s) - qv0)
    plot_func_liquid = lambda s: s.project_H1(s.q - s.solve_qv_from_entropy(s.h, s.q, s.s) - ql0)

    fig_list = [plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.4, 4.8)) for _ in range(6)]
    pfunc_list = [
        plot_func_mpt, plot_func_entropy, plot_func_density,
        plot_func_water, plot_func_vapour, plot_func_liquid,
    ]

    labels = ["moist_potential_temperature", "entropy", "density", "water", "vapour", "liquid"]

    energy = []
    for i, tend in enumerate(tends):
        filepaths = [solver_plot.get_filepath(data_dir, exp_name_short, proc=i, nprocx=nproc, time=tend) for i in range(nproc)]
        solver_plot.load(filepaths)
        energy.append(solver_plot.integrate(solver_plot.energy()))

        for (fig, axs), plot_fun, label in zip(fig_list, pfunc_list, labels):

            # if label == 'moist_potential_temperature':
            #     levels = np.linspace(0)
            # else:
            #     levels = 1000
            ax = axs[i // 2][i % 2]
            ax.tick_params(labelsize=8)
            im = solver_plot.plot_solution(ax, dim=2, plot_func=plot_fun)
            # if label == 'entropy':
            #     cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), label='Entropy (K)')
            # elif label == 'density':
            #     cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), label='Density ($\text{kg m}^{-3}$)')
            # else:
            #     cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), label=f'{label.capitalize() mass fraction'})
            cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt))
            cbar.ax.tick_params(labelsize=8)

            if (i // 2) == 1:
                ax.set_xlabel('x (m)', fontsize='xx-small')
            if (i % 2) == 0:
                ax.set_ylabel('z (m)', fontsize='xx-small')
            # fig.tight_layout(w_pad=1.0, h_pad=1.0)
            fig.tight_layout()

    for (fig, ax), label in zip(fig_list, labels):

        plot_name = f'{label}_{exp_name_short}'
        fp = solver_plot.get_filepath(plot_dir, plot_name, ext='png')
        print(fp)
        fig.savefig(fp, bbox_inches="tight")
