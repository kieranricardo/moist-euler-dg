from matplotlib import pyplot as plt
from moist_euler_dg.unstable_two_phase_euler_2D import UnstableTwoPhaseEuler2D
from moist_euler_dg.fortran_two_phase_euler_2D import FortranTwoPhaseEuler2D
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
run_time = 1195.0 # total run time in seconds

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
a = 0.0 # kinetic energy dissipation parameter
upwind = False

# experiment name - change this for new experiments!
exp_name_short = 'bad-stability'
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

# save data at these times
n = int(run_time / 5)
tends = np.arange(n + 1) * run_time / n

time_list = []
energy_list = []
conservation_data_fp = os.path.join(data_dir, 'conservation_data.npy')

plot_func_entropy = lambda s: s.s
plot_func_density = lambda s: s.h
plot_func_water = lambda s: s.q
plot_func_u = lambda s: s.u
plot_func_w = lambda s: s.w
plot_func_T = lambda s: s.T

pfunc_list = [
    plot_func_entropy, plot_func_density,
    plot_func_water,
    plot_func_u, plot_func_w, plot_func_T
]

labels = ["entropy", "density", "water", "u", "w", "T"]


if run_model:
    
    solver = UnstableTwoPhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=a, nz=nz, upwind=upwind, nprocx=nproc)
    u, v, density, s, qw, qv = initial_condition(solver.xs, solver.zs, solver, pert=60.0)

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

            levels = np.linspace(vmin, vmax, 1000)

            plot[0].remove()
            plot[0] = ax.tricontourf(
                xcoord.ravel(),
                zcoord.ravel(),
                data[idx].ravel(),
                cmap='nipy_spectral', levels=levels, vmin=vmin, vmax=vmax)
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
        

    solver_plot = UnstableTwoPhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=a, nz=nz, upwind=upwind, nprocx=1)
    
    xcoord = np.concatenate([np.load(fp) for fp in _get_fps('xcoord')], axis=0)
    zcoord = np.concatenate([np.load(fp) for fp in _get_fps('zcoord')], axis=0)
    
    # labels = ["entropy", "density", "water", "vapour", "ice",  "T", "u", "w"]

    labels = ["entropy", "water", "density", "vapour", "ice", "T", "u", "w"]


    # if size > 1:
    #     labels = [labels[i] for i in range(rank, len(labels), size)]

    print(f'Rank {rank} running {labels}')

    for label in labels[1:2]:
        print(f'\n{label}: loading data')
        all_data = _load_data(label)
        print(f'{label}: making movie')
        t0 = time.time()

        global vmin
        global vmax
        vmin = all_data[0].min()
        vmax = all_data[0].max()

        if (vmin == vmax):
            vmin = vmin - 0.1 * vmax
            vmax = vmax + 0.1 * vmax

        rank_size = (all_data.shape[0] // size)
        data = np.copy(all_data[rank * rank_size:(rank + 1) * rank_size])
        del all_data

        _make_movie(label, data)
        print(f'Time: {time.time() - t0}s')

        # if rank == 0:
        #     filepaths = [
        #         f"file '{label}_{experiment_name}_part_{rank + 1}_of_{size}.mp4'\n" for rank in range(size)]
        #
        #     with open(os.path.join(movie_dir, f'{label}_filepaths.txt'), 'w') as f:
        #         f.writelines(filepaths)

        # break

    
        