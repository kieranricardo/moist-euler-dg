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
run_time = 3000 # total run time in seconds

p_surface = 1_00_000.0 # surface pressure in Pa
SST = 300.0 # sea surface temperature in Kelvin

cooling_rate = 5.0 / (3600 * 24) # cools 10 K per day
boundary_layer_top = 1250.0 # height of boundary layer - diffusion applied within boundary layer
Ksurf = 20.0 # diffusivity at surface - quadratically decreases to 0 at boundary_layer_top

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
exp_name_short = 'forced-convection'
experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{poly_order}'

data_dir = os.path.join('data', experiment_name)
plot_dir = os.path.join('plots', experiment_name)
data_dump_dir = os.path.join(plot_dir, 'data-dump')
movie_dir = os.path.join(plot_dir, 'movies')

if rank == 0:
    print(f"---------- {exp_name_short} with nx={nx}, nz={nz}")
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    if not os.path.exists(data_dir): os.makedirs(data_dir)
    if not os.path.exists(data_dump_dir): os.makedirs(data_dump_dir)
    if not os.path.exists(movie_dir): os.makedirs(movie_dir)

comm.barrier()


def neutrally_stable_dry_profile(solver):
    # create a hydrostatically balanced pressure and density profile
    dexdy = -g / (solver.cpd * SST)
    ex = 1 + dexdy * solver.zs
    p = p_surface * ex ** (solver.cpd / solver.Rd)
    density = p / (solver.Rd * ex * SST)

    return density, p


def stable_dry_profile(solver):

    # set hydrostatic pressure/density profile
    TE = 310.0
    TP = 240.0
    GRAVITY = solver.g
    T0 = 0.5 * (TE + TP)
    b = 2.0
    KP = 3.0
    GAMMA = 0.009 # lapse rate
    P0 = 100000
    RD = solver.Rd

    ys = solver.zs

    A = 1.0 / GAMMA
    B = (TE - TP) / ((TE + TP) * TP)
    C = 0.5 * (KP + 2.0) * (TE - TP) / (TE * TP)
    H = RD * T0 / GRAVITY

    fac = ys / (b * H)
    fac2 = fac * fac
    cp = np.cos(2.0 * np.pi / 9.0)
    cpk = np.power(cp, KP)
    cpkp2 = np.power(cp, KP + 2)
    fac3 = cpk - (KP / (KP + 2.0)) * cpkp2

    torr_1 = (A * GAMMA / T0) * np.exp(GAMMA * (ys) / T0) + B * (1.0 - 2.0 * fac2) * np.exp(-fac2)
    torr_2 = C * (1.0 - 2.0 * fac2) * np.exp(-fac2)

    int_torr_1 = A * (np.exp(GAMMA * ys / T0) - 1.0) + B * ys * np.exp(-fac2)
    int_torr_2 = C * ys * np.exp(-fac2)

    tempInv = torr_1 - torr_2 * fac3
    T = 1.0 / tempInv
    p = P0 * np.exp(-GRAVITY * int_torr_1 / RD + GRAVITY * int_torr_2 * fac3 / RD)
    density = p / (solver.Rd * T)

    return density, p

def initial_condition(solver):
    # initial wind is zero
    u = np.zeros_like(solver.xs)
    w = np.zeros_like(solver.xs)

    # density, p = neutrally_stable_dry_profile(solver)
    density, p = stable_dry_profile(solver)

    # add arbitrary moisture profile

    qw_sfc = solver.rh_to_qw(0.95, p[0, 0, 0, 0], density[0, 0, 0, 0])
    # qw = qw_sfc * np.exp(-solver.zs / 1000)

    # rh = np.zeros_like(density)
    # rh[solver.zs <= boundary_layer_top] = 0.95
    #
    # tmp = np.exp(-(solver.zs - boundary_layer_top) / 1000)[solver.zs > boundary_layer_top]
    # rh[solver.zs > boundary_layer_top] = 0.95 * tmp
    #
    rh = 0.95 + (1 - np.exp(-(solver.zs / 250)**2)) * (0.01 - 0.95)
    # rh = 0.0001
    qw = solver.rh_to_qw(rh, p, density)
    # qw = 1e-12
    # qw[solver.zs <= boundary_layer_top]

    # model must be initialized with entropy not temperature
    # so convert density, pressure, qw profile to a density, entropy, qw profile
    s = solver.entropy(density, qw, p=p)

    # can also do s = solver.entropy(density, qw, T=T) to use temperature profiles

    return u, w, density, s, qw


def diffusive_forcing(solver, state, dstatedt):
    u, w, h, s, q, T, mu, p, ie = solver.get_vars(state)
    dudt, dwdt, dhdt, dsdt, dqdt, *_ = solver.get_vars(dstatedt)

    # internal cooling
    y = (solver.zs - boundary_layer_top) / (solver.zs.max() - boundary_layer_top)
    T_forcing = -cooling_rate * y**2 * (solver.zs >= boundary_layer_top)  # constantly cool at a rate of 1K per day
    s_forcing = T_forcing * solver.cvd / T  # convert temperature forcing to entropy forcing

    # forcing in boundary layer
    # forcing in boundary layer
    K = Ksurf * (1.0 - (solver.zs / boundary_layer_top)) ** 2 * (solver.zs <= boundary_layer_top)

    def ddz(arr):
        out = solver.ddz(arr)

        ip = solver.ip_vert_int
        im = solver.im_vert_int

        num_flux = 0.5 * (arr[ip] + arr[im])
        out[ip] += (num_flux - arr[ip]) * solver.norm_grad_zeta[ip] / solver.weights_z[-1]
        out[im] -= (num_flux - arr[im]) * solver.norm_grad_zeta[im] / solver.weights_z[-1]

        return out

    # F = ddz(T)  # diffusive flux
    # # bottom boundary condition
    # ip = solver.ip_vert_ext
    # F[ip] += (SST - T[ip]) * solver.norm_grad_zeta[ip] / solver.weights_z[-1]
    # T_forcing = -ddz(K * F)
    # s_forcing += T_forcing * solver.cvd / T

    F = ddz(s)  # diffusive flux
    # bottom boundary condition
    ip = solver.ip_vert_ext
    F[ip] += (s0 - s[ip]) * solver.norm_grad_zeta[ip] / solver.weights_z[-1]
    s_forcing = -ddz(K * F)

    dsdt += s_forcing

# save data at these times
n = int(run_time / 5)
tends = np.arange(n + 1) * run_time / n

time_list = []
energy_list = []
conservation_data_fp = os.path.join(data_dir, 'conservation_data.npy')

plot_func_entropy = lambda s: s.s
plot_func_density = lambda s: s.h
plot_func_water = lambda s: s.q
plot_func_vapour = lambda s: s.solve_fractions_from_entropy(s.h, s.q, s.s)[0]
plot_func_liquid = lambda s: s.solve_fractions_from_entropy(s.h, s.q, s.s)[1]
plot_func_ice = lambda s: s.solve_fractions_from_entropy(s.h, s.q, s.s)[2]
plot_func_u = lambda s: s.u
plot_func_w = lambda s: s.w
plot_func_T = lambda s: s.T

pfunc_list = [
    plot_func_entropy, plot_func_density,
    plot_func_water, plot_func_vapour, plot_func_liquid, plot_func_ice,
    plot_func_u, plot_func_w, plot_func_T
]

labels = ["entropy", "density", "water", "vapour", "liquid", "ice", "u", "w", "T"]


if run_model:
    
    solver = FortranThreePhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=a, nz=nz, upwind=upwind, nprocx=nproc, forcing=diffusive_forcing)
    u, v, density, s, qw = initial_condition(solver)

    s0 = np.copy(s[solver.ip_vert_ext])
    s1 = np.copy(s[solver.im_vert_ext])

    np.random.seed(42 + rank)
    noise = 2 * (np.random.random(density.shape) - 0.5)
    density += 0.01 * density * noise
    solver.set_initial_condition(u, v, density, s, qw)

    time_list.append(solver.time)
    energy_list.append(solver.energy())

    shape = (len(tends),) + solver.xs.shape
    data_dict = dict((label, np.zeros(shape)) for label in labels)

    for i, tend in enumerate(tends):
        
        t0 = time.time()
        while solver.time < tend:
            dt = solver.get_dt()
            dt = min(dt, tend - solver.time)

            time_list.append(solver.time)
            energy_list.append(solver.energy())
            solver.time_step(dt=dt)

        t1 = time.time()

        for label, pfunc in zip(labels, pfunc_list):
            
            data_dict[label][i, :] = pfunc(solver)

        if rank == 0:
            print('s bot range:', solver.s[solver.ip_vert_ext].min(), solver.s[solver.ip_vert_ext].max())
            print("Simulation time (unit less):", solver.time)
            print('Relative energy change:', (energy_list[-1] - energy_list[0]) / energy_list[0])
            print("Wall time:", time.time() - t0, '\n')

        # solver.save(solver.get_filepath(data_dir, exp_name_short))
        

    if rank == 0:
        conservation_data = np.zeros((2, len(time_list)))
        conservation_data[0, :] = np.array(time_list)
        conservation_data[1, :] = np.array(energy_list)

        np.save(conservation_data_fp, conservation_data)

    for label, arr in data_dict.items():
        fp = os.path.join(data_dump_dir, f"{label}_part_{rank}_of_{size}.npy")
        print(fp)
        np.save(fp, arr)

    label = 'xcoord'
    arr = solver.xs
    fp = os.path.join(data_dump_dir, f"{label}_part_{rank}_of_{size}.npy")
    print(fp)
    np.save(fp, arr)

    label = 'zcoord'
    arr = solver.zs
    fp = os.path.join(data_dump_dir, f"{label}_part_{rank}_of_{size}.npy")
    print(fp)
    np.save(fp, arr)
    

# plotting
else:

    from matplotlib.animation import FFMpegWriter as MovieWriter
    import matplotlib.animation as animation

    def _get_fps(label):
        return [os.path.join(data_dump_dir, f"{label}_part_{i}_of_{nproc}.npy") for i in range(nproc)]

    def _load_data(label):
        data = np.concatenate([np.load(fp) for fp in _get_fps(label)], axis=1)
        for i in range(len(tends)):
            solver_plot.project_H1(data[i])

        return data

    def _make_movie(label, data):

        def update_plot(frame_number, plot, ax):

            global idx
            global vmin
            global vmax

            plot[0].remove()
            plot[0] = ax.tricontourf(xcoord.ravel(), zcoord.ravel(), data[idx].ravel(), cmap='nipy_spectral', levels=1000, vmin=vmin, vmax=vmax)
            idx += 1
            idx = idx % data.shape[0]

        global idx
        global vmin
        global vmax

        if vmin == vmax:
            vmin = data.min()
            vmax = data.max()

        if label in ["u", "w"]:
            vmax = 20.0
            vmin = -vmax

        idx = 0

        fig = plt.figure()
        ax = fig.add_subplot(111)
        levels = np.linspace(vmin, vmax, 1000)
        plot = [ax.tricontourf(xcoord.ravel(), zcoord.ravel(), data[0].ravel(), cmap='nipy_spectral', levels=levels, vmin=vmin, vmax=vmax)]
        cbar = plt.colorbar(plot[0], ax=ax)

        moviewriter = MovieWriter(fps=30)
        fp = os.path.join(movie_dir, f"{label}_{experiment_name}_part_{rank+1}_of_{size}.mp4")
        with moviewriter.saving(fig, fp, dpi=100):
            moviewriter.grab_frame()
            for _ in range(data.shape[0]):

                update_plot(0, plot, ax)
                moviewriter.grab_frame()
        

    solver_plot = ThreePhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=a, nz=nz, upwind=upwind, nprocx=1)
    
    xcoord = np.concatenate([np.load(fp) for fp in _get_fps('xcoord')], axis=0)
    zcoord = np.concatenate([np.load(fp) for fp in _get_fps('zcoord')], axis=0)
    
    labels = ["entropy", "density", "water", "vapour", "ice",  "T", "u", "w"]


    # if size > 1:
    #     labels = [labels[i] for i in range(rank, len(labels), size)]

    print(f'Rank {rank} running {labels}')

    for label in labels:
        print(f'\n{label}: loading data')
        all_data = _load_data(label)
        print(f'{label}: making movie')
        t0 = time.time()

        global vmin
        global vmax
        vmin = all_data[0].min()
        vmax = all_data[0].max()

        rank_size = (all_data.shape[0] // size)
        data = np.copy(all_data[rank * rank_size:(rank + 1) * rank_size])
        del all_data

        _make_movie(label, data)
        print(f'Time: {time.time() - t0}s')

        if rank == 0:
            filepaths = [
                f"file '{label}_{experiment_name}_part_{rank + 1}_of_{size}.mp4'\n" for rank in range(size)]

            with open(os.path.join(movie_dir, f'{label}_filepaths.txt'), 'w') as f:
                f.writelines(filepaths)

        break

    
        