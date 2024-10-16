from matplotlib import pyplot as plt
from moist_euler_dg.unstable_three_phase_euler_2D import UnstableThreePhaseEuler2D as ThreePhaseEuler2D
from moist_euler_dg.fortran_three_phase_euler_2D import FortranThreePhaseEuler2D
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
a = 0.5
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


run_time = 1000

# unstable run
solver = ThreePhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=cfl, a=a, nz=nz, upwind=upwind, nprocx=nproc)
u, v, density, s, qw, qv, ql, qi = initial_condition(solver.xs, solver.zs, solver, pert=2.0)
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

plt.plot(time_list, (energy_list - energy_list[0]) / energy_list[0], 'r--', label='Unstable energy')
plt.plot(time_list, (entropy_var_list - entropy_var_list[0]) / entropy_var_list[0], 'b--', label='Unstable entropy variance')
plt.plot(time_list, (water_var_list - water_var_list[0]) / water_var_list[0], 'g--', label='Unstable water variance')
print('Time of first limit:', solver.first_water_limit_time)

### stable run
solver = FortranThreePhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=cfl, a=a, nz=nz, upwind=upwind, nprocx=nproc)
u, v, density, s, qw, qv, ql, qi = initial_condition(solver.xs, solver.zs, solver, pert=2.0)
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

plt.plot(time_list, (energy_list - energy_list[0]) / energy_list[0], 'r', label='Stable energy')
plt.plot(time_list, (entropy_var_list - entropy_var_list[0]) / entropy_var_list[0], 'b', label='Stable entropy variance')
plt.plot(time_list, (water_var_list - water_var_list[0]) / water_var_list[0], 'g', label='Stable water variance')

print('Time of first limit:', solver.first_water_limit_time)

plt.grid()
plt.legend()
plt.yscale('symlog', linthresh=1e-15)
fp = os.path.join(plot_dir, f'conservation_{exp_name_short}')
plt.savefig(fp, bbox_inches="tight")




