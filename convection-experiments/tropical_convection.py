import numpy as np


from matplotlib import pyplot as plt
from moist_euler_dg.three_phase_euler_2D import ThreePhaseEuler2D
from moist_euler_dg.fortran_three_phase_euler_2D import FortranThreePhaseEuler2D
from moist_euler_dg.euler_2D import Euler2D
import numpy as np
import time
import os
import argparse
from mpi4py import MPI
import matplotlib.ticker as ticker

# test case parameters
domain_width = 10_000 # width of domain in metres
domain_height = 10_000 # height of domain in metres
run_time = 3600 * 1 # total run time in seconds

p_surface = 1_00_000.0 # surface pressure in Pa
SST = 300 # sea surface temperature in Kelvin

cooling_rate = 5.0 / (3600 * 24) # cools 10 K per day
boundary_layer_top = 1250.0 # height of boundary layer - diffusion applied within boundary layer

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument('--order', type=int, help='Polynomial order')
parser.add_argument('--nx', type=int, help='Number of cells in horizontal')
parser.add_argument('--nz', type=int, help='Number of cells in vertical')
parser.add_argument('--nproc', type=int, help='Number of procs', default=1)
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

# maps to define geometry these can be arbitrary - maps [0, 1]^2 to domain
zmap = lambda x, z: z * domain_height
xmap = lambda x, z: domain_width * (x - 0.5)

# number of cells in the vertical and horizontal direction
nz = args.nz
nx = args.nx

nproc = args.nproc
run_model = (not args.plot) # whether to run model - set false to just plot previous run

g = 9.81 # gravitational acceleration
poly_order = args.order # spatial order of accuracy
a = 0.5 # kinetic energy dissipation parameter
upwind = True

# experiment name - change this for new experiments!
exp_name_short = 'tropical-convection'

experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{poly_order}'
data_dir = os.path.join('data', experiment_name)
plot_dir = os.path.join('plots', experiment_name)

if rank == 0:
    print(f"---------- {exp_name_short} with nx={nx}, nz={nz}")
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    if not os.path.exists(data_dir): os.makedirs(data_dir)

comm.barrier()



def tropical_rce_initial_condition(
    solver,
    Ts=300.0,
    p0=1014.8e2,
    seed=1234,
    add_noise=True,
    noise_layers=5,
    qtop=1.0e-14,
):
    """
    Construct a 2D tropical RCE-like initial condition based on the RCEMIP
    analytic sounding.

    Parameters
    ----------
    x : ndarray, shape (nx,)
        Horizontal coordinates [m].
    z : ndarray, shape (nz,)
        Vertical coordinates [m].
    Ts : float
        Fixed sea-surface temperature [K].
    p0 : float
        Surface pressure [Pa].
    seed : int
        Random seed for thermal perturbations.
    add_noise : bool
        Whether to add weak low-level thermal noise.
    noise_layers : int
        Number of lowest vertical levels to perturb.
    qtop : float
        Stratospheric water vapour mixing ratio [kg/kg].

    Returns
    -------
    state : dict
        Dictionary containing 2D arrays with shape (nz, nx):
        T, Tv, p, rho, qv, ql, qi, u, w, theta, plus coordinates X, Z.
    """

    x = solver.xs
    z = solver.zs
    Rd = solver.Rd
    cp = solver.cpd
    # ---------------------------------------------------------------------
    # Constants
    # ---------------------------------------------------------------------
    kappa = Rd / cp
    p_ref = 1.0e5

    # RCEMIP-like sounding parameters
    q0 = 18.65e-3       # kg/kg, for Ts = 300 K
    Gamma = 0.0067      # K/m
    zt = 15_000.0       # m
    zq1 = 4_000.0       # m
    zq2 = 7_500.0       # m

    # ---------------------------------------------------------------------
    # Background vertical profiles
    # ---------------------------------------------------------------------

    # Water vapour mixing ratio qv(z)
    qv = q0 * np.exp(-z / zq1) * np.exp(-(z / zq2) ** 2)
    qv = np.where(z <= zt, qv, qtop)

    # Virtual temperature profile Tv(z)
    Tv0 = Ts * (1.0 + 0.608 * q0)
    Tvt = Tv0 - Gamma * zt

    Tv = np.where(
        z <= zt,
        Tv0 - Gamma * z,
        Tvt,
    )

    # Actual temperature from virtual temperature
    T = Tv / (1.0 + 0.608 * qv)

    # Hydrostatic pressure profile
    pt = p0 * (Tvt / Tv0) ** (g / (Rd * Gamma))

    p_trop = p0 * ((Tv0 - Gamma * z) / Tv0) ** (g / (Rd * Gamma))
    p_strat = pt * np.exp(-g * (z - zt) / (Rd * Tvt))

    p = np.where(z <= zt, p_trop, p_strat)

    # Density from ideal gas law using virtual temperature
    density = p / (Rd * Tv)

    # ---------------------------------------------------------------------
    # Small low-level thermal noise
    # ---------------------------------------------------------------------
    if add_noise:
        rng = np.random.default_rng(seed + rank)
        for i in range(noise_layers):
            ii = i // (poly_order + 1)
            jj = i % (poly_order + 1)
            T[:, ii, :, jj] += 0.1 * rng.uniform(-1.0, 1.0, size=T[:, 0, :, 0].shape) * abs(T[:, 0, :, 0])

        # Update virtual temperature and density consistently with perturbed T
        Tv = T * (1.0 + 0.608 * qv)
        density = p / (Rd * Tv)

    # ---------------------------------------------------------------------
    # Moisture species and velocity
    # ---------------------------------------------------------------------
    ql = np.zeros_like(T)
    qi = np.zeros_like(T)

    u = np.zeros_like(T)
    w = np.zeros_like(T)

    s = solver.entropy(density, qv, T=T)

    return u, w, density, s, qv, T


# save data at these times
tends = np.array([0.0, (1 / 3), (2 / 3), 1.0]) * run_time

time_list = []
energy_list = []
water_mass_list = []
dry_mass_list = []

conservation_data_fp = os.path.join(data_dir, 'conservation_data.npy')

if run_model:
    solver = FortranThreePhaseEuler2D(
        xmap, zmap, order=poly_order, nx=nx, g=g, cfl=0.5, a=a, nz=nz,
        upwind=upwind, nprocx=nproc, forcing=None, b=0.5, sst=SST
    )
    u, v, density, s, qw, T = tropical_rce_initial_condition(solver, add_noise=True)
    # np.random.seed(42 + rank)
    # noise = 2 * (np.random.random(density.shape) - 0.5)
    # density += 0.01 * density * noise
    solver.set_initial_condition(u, v, density, s, qw)
    # plt.figure(1)
    # plt.title('Temperature')
    # plt.plot(solver.zs[0, :-1, 0, :].ravel(), solver.T[0, :-1, 0, :].ravel())
    # plt.figure(2)
    # plt.title('Pressure')
    # plt.plot(solver.zs[0, :-1, 0, :].ravel(), solver.p[0, :-1, 0, :].ravel())
    # plt.figure(3)
    # plt.title('qw')
    # plt.plot(solver.zs[0, :-1, 0, :].ravel(), solver.q[0, :-1, 0, :].ravel())
    # plt.figure(4)
    # plt.title('ql')
    # plt.plot(solver.zs[0, :-1, 0, :].ravel(), solver.ql[0, :-1, 0, :].ravel())
    # im = solver.plot_solution(plt.gca(), dim=2, plot_func=lambda s: s.project_H1(s.T))
    # plt.colorbar(im)
    # plt.show()

    time_list.append(solver.time)
    energy_list.append(solver.energy())
    water_mass_list.append(solver.integrate(solver.q * solver.h))
    dry_mass_list.append(solver.integrate((1 - solver.q) * solver.h))

    for i, tend in enumerate(tends):
        t0 = time.time()
        while solver.time < tend:
            dt = solver.get_dt()

            time_list.append(solver.time)
            energy_list.append(solver.energy())
            water_mass_list.append(solver.integrate(solver.q * solver.h))
            dry_mass_list.append(solver.integrate((1 - solver.q) * solver.h))
            solver.time_step(dt=dt)

        t1 = time.time()

        if rank == 0:
            print('s bot range:', solver.s[solver.ip_vert_ext].min(), solver.s[solver.ip_vert_ext].max())
            print("Simulation time (unit less):", solver.time)
            print('Relative energy change:', (energy_list[-1] - energy_list[0]) / energy_list[0])
            print('Relative water mass change:', (water_mass_list[-1] - water_mass_list[0]) / water_mass_list[0])
            print('Relative dry mass change:', (dry_mass_list[-1] - dry_mass_list[0]) / dry_mass_list[0])
            print("Wall time:", time.time() - t0, '\n')

        solver.save(solver.get_filepath(data_dir, exp_name_short))

    if rank == 0:
        print('Relative energy change:', (energy_list[-1] - energy_list[0]) / energy_list[0])
        print('Relative water mass change:', (water_mass_list[-1] - water_mass_list[0]) / water_mass_list[0])
        print('Relative dry mass change:', (dry_mass_list[-1] - dry_mass_list[0]) / dry_mass_list[0])
        print("Bottom temp range:", solver.T[:, 0, :, 0].min(), solver.T[:, 0, :, 0].max())

        conservation_data = np.zeros((4, len(time_list)))
        conservation_data[0, :] = np.array(time_list)
        conservation_data[1, :] = np.array(energy_list)
        conservation_data[2, :] = np.array(water_mass_list)
        conservation_data[3, :] = np.array(dry_mass_list)

        np.save(conservation_data_fp, conservation_data)

# plotting
elif rank == 0:
    plt.rcParams['font.size'] = '12'

    #
    solver_plot = ThreePhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=a, nz=nz, upwind=upwind, nprocx=1)
    # base state of the initial condition (excludes bubble perturbation)
    _, _, h0, s0, qw0, T0 = tropical_rce_initial_condition(solver_plot, add_noise=False)
    qv0, ql0, qi0 = solver_plot.solve_fractions_from_entropy(h0, qw0, s0)


    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)


    plot_func_entropy = lambda s: s.project_H1(s.s)
    plot_func_density = lambda s: s.project_H1(s.h)
    plot_func_water = lambda s: s.project_H1(s.q)
    plot_func_vapour = lambda s: s.project_H1(s.solve_fractions_from_entropy(s.h, s.q, s.s)[0])
    plot_func_liquid = lambda s: s.project_H1(s.solve_fractions_from_entropy(s.h, s.q, s.s)[1])
    plot_func_ice = lambda s: s.project_H1(s.solve_fractions_from_entropy(s.h, s.q, s.s)[2])
    plot_func_u = lambda s: s.project_H1(s.u)
    plot_func_w = lambda s: s.project_H1(s.w)
    plot_func_T = lambda s: s.project_H1(s.T)

    pfunc_list = [
        plot_func_entropy, plot_func_density,
        plot_func_water, plot_func_vapour, plot_func_liquid, plot_func_ice,
        plot_func_u, plot_func_w, plot_func_T
    ]

    labels = ["entropy", "density", "water", "vapour", "liquid", "ice", "u", "w", "T"]
    vmaxs = dict()
    vmins = dict()

    fig_list = [plt.subplots(2, 2, sharex=True, sharey=True) for _ in range(len(labels))]

    for i, tend in enumerate(tends[::-1]):
        i = 3 - i
        filepaths = [solver_plot.get_filepath(data_dir, exp_name_short, proc=j, nprocx=nproc, time=tend) for j in range(nproc)]
        solver_plot.load(filepaths)

        print("Bottom layer temp range:", solver_plot.T[:, 0, :, 0].min(), solver_plot.T[:, 0, :, 0].max())
        print("Bottom cell temp range:", solver_plot.T[:, 0].min(), solver_plot.T[:, 0].max(), "\n")

        for (fig, axs), plot_fun, label in zip(fig_list, pfunc_list, labels):

            if label in ('entropy', 'density'):
                if label in vmaxs.keys():
                    vmax = vmaxs[label]
                    vmin = vmins[label]
                else:
                    vmax = plot_fun(solver_plot).max()
                    vmin = plot_fun(solver_plot).min()
                    vmaxs[label] = vmax
                    vmins[label] = vmin
            else:
                vmin = vmax = None

            ax = axs[i // 2][i % 2]
            ax.tick_params(labelsize=8)
            im = solver_plot.plot_solution(ax, dim=2, plot_func=plot_fun, vmin=vmin, vmax=vmax)
            cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt))
            cbar.ax.tick_params(labelsize=8)
            plt.tight_layout()

    for (fig, ax), label in zip(fig_list, labels):
        plot_name = f'{label}_{exp_name_short}'
        fp = solver_plot.get_filepath(plot_dir, plot_name, ext='png')
        fig.savefig(fp, bbox_inches="tight")

    conservation_data = np.load(conservation_data_fp)
    time_list = conservation_data[0, :]
    energy_list = conservation_data[1, :]
    water_mass_list = conservation_data[2, :]
    dry_mass_list = conservation_data[3, :]

    energy_list = (energy_list - energy_list[0]) / energy_list[0]
    water_mass_list = (water_mass_list - water_mass_list[0]) / water_mass_list[0]
    dry_mass_list = (dry_mass_list - dry_mass_list[0]) / dry_mass_list[0]

    print('Energy error:', energy_list[-1])
    print('Water mass error:', water_mass_list[-1])
    print('Dry mass error:', dry_mass_list[-1])

    plt.figure()
    plt.plot(time_list, energy_list, label='Energy')
    plt.plot(time_list, water_mass_list, label='Water mass')
    plt.plot(time_list, dry_mass_list, label='Dry mass')
    plt.grid()
    plt.legend()
    # plt.yscale('symlog', linthresh=1e-15)
    fp = os.path.join(plot_dir, f'conservation_{exp_name_short}')
    plt.savefig(fp, bbox_inches="tight")