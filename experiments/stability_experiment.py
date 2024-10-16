from matplotlib import pyplot as plt
from moist_euler_dg.unstable_two_phase_euler_2D import UnstableTwoPhaseEuler2D
from moist_euler_dg.fortran_two_phase_euler_2D import FortranTwoPhaseEuler2D
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
args = parser.parse_args()

xlim = 10_000
zlim = 10_000

nz = args.n
nproc = args.nproc
nx = nz
cfl = 0.5
g = 9.81
poly_order = args.o
a = 0.0
upwind = False

exp_name_short = 'stability-experiment'
experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{poly_order}'
data_dir = os.path.join('data', experiment_name)
plot_dir = os.path.join('plots', experiment_name)

if rank == 0:
    print(f"---------- Ice bubble with nx={nx}, nz={nz}, cfl={cfl}")
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


run_time = 1500

# unstable run
solver = UnstableTwoPhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=cfl, a=a, nz=nz, upwind=upwind, nprocx=nproc)
u, v, density, s, qw, qv = initial_condition(solver.xs, solver.zs, solver, pert=60.0)
solver.set_initial_condition(u, v, density, s, qw)
solver.var_stable = False

time_list = []
energy_list = []
entropy_var_list = []
water_var_list = []

while solver.time < run_time:

    try:
        time_list.append(solver.time)
        energy_list.append(solver.energy())
        entropy_var_list.append(solver.integrate(solver.h * solver.s ** 2))
        water_var_list.append(solver.integrate(solver.h * solver.q ** 2))

        solver.time_step()
    except RuntimeError as e:
        print(e)
        print('Run failed at time:', solver.time)
        break


energy_list = np.array(energy_list)
entropy_var_list = np.array(entropy_var_list)
water_var_list = np.array(water_var_list)

plt.plot(time_list, (energy_list - energy_list[0]) / energy_list[0], 'r--', label='Method 2 energy')
# plt.plot(time_list, (entropy_var_list - entropy_var_list[0]) / entropy_var_list[0], 'b--', label='Unstable entropy variance')
plt.plot(time_list, (water_var_list - water_var_list[0]) / water_var_list[0], 'g--', label='Method 2 water variance')
print('Time of first limit:', solver.first_water_limit_time)

### stable run
solver = FortranTwoPhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=cfl, a=a, nz=nz, upwind=upwind, nprocx=nproc)
u, v, density, s, qw, qv = initial_condition(solver.xs, solver.zs, solver, pert=60.0)
solver.set_initial_condition(u, v, density, s, qw)
solver.var_stable = True

time_list = []
energy_list = []
entropy_var_list = []
water_var_list = []

while solver.time < run_time:

    try:
        time_list.append(solver.time)
        energy_list.append(solver.energy())
        entropy_var_list.append(solver.integrate(solver.h * solver.s ** 2))
        water_var_list.append(solver.integrate(solver.h * solver.q ** 2))

        solver.time_step()
    except RuntimeError as e:
        print(e)
        print('Run failed at time:', solver.time)
        break


energy_list = np.array(energy_list)
entropy_var_list = np.array(entropy_var_list)
water_var_list = np.array(water_var_list)

plt.plot(time_list, (energy_list - energy_list[0]) / energy_list[0], 'r', label='Method 1 energy')
# plt.plot(time_list, (entropy_var_list - entropy_var_list[0]) / entropy_var_list[0], 'b', label='Stable entropy variance')
plt.plot(time_list, (water_var_list - water_var_list[0]) / water_var_list[0], 'g', label='Method 1 water variance')

print('Time of first limit:', solver.first_water_limit_time)
plt.ylabel('Relative error')
plt.xlabel('Time (s)')
plt.grid()
plt.legend()
plt.yscale('symlog', linthresh=1e-15)
fp = os.path.join(plot_dir, f'conservation_{exp_name_short}')
plt.savefig(fp, bbox_inches="tight")




