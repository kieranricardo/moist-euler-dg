from matplotlib import pyplot as plt
# from moist_euler_dg.three_phase_euler_2D import ThreePhaseEuler2D
from moist_euler_dg.fortran_two_phase_euler_2D import FortranTwoPhaseEuler2D as TwoPhaseEuler2D
import numpy as np
import time
import os
import argparse
import matplotlib.ticker as ticker


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

    qw *= 2
    enthalpy, T, p, ie, mu, qv, ql = solver.get_thermodynamic_quantities(density, s, qw)
    #  0.3410208713540216 0.10594892674155956 0.6589791286459784
    # print('qw min-max:', qw.min(), qw.max())
    # print('T min-max:', T.min() - 273, T.max() - 273)
    # print('Density min-max:', density.min(), density.max())
    # print('Pressure min-max:', p.min(), p.max())
    # print('qv/qw min-max:', (qv/qw).min(), (qv/qw).max())
    # print('all vapour mean:', (qv == qw).mean())
    # print('ql/qw min-max:', (ql/qw).min(), (ql/qw).max())
    # print('qi/qw min-max:', (qi/qw).min(), (qi/qw).max(), '\n')


    return u, v, density, s, qw, qv, ql


tend = 50

cfl = 0.5
g = 9.81
poly_order = 3
a = 0.0
upwind = False

nx = nz = 8
zlim = xlim = 10_000
zmap = lambda x, z: z * zlim
xmap = lambda x, z: xlim * (x - 0.5)

energy_errors = []
entropy_var_errors = []
water_var_errors = []
dts = np.array([0.1, 0.2, 0.3, 0.4, 0.5])

# energy_errors = [1.0319002461779809e-11, 6.740595977011774e-11, 2.037252762021018e-10, 4.393992454139475e-10, 7.670496134464608e-10]
# entropy_var_errors = [3.3716707350242196e-13, 2.2473262093997326e-13, 1.331496486709241e-13, 4.790389100084936e-13, 3.0362753559258123e-12]
# water_var_errors = [1.530631309231234e-11, 1.1971922810833799e-10, 3.9969844913486167e-10, 9.646683424382586e-10, 1.863721788056275e-09]

for dt in dts:
    nteps = int(tend / dt)

    solver = TwoPhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=cfl, a=a, nz=nz, upwind=upwind, nprocx=1)
    u, v, density, s, qw, qv, ql = initial_condition(solver.xs, solver.zs, solver, pert=10.0)
    solver.set_initial_condition(u, v, density, s, qw)

    # plt.figure(1)
    # plt.tricontourf(solver.xs.ravel(), solver.zs.ravel(), solver.s.ravel(), levels=100)
    # plt.show()
    # exit(0)

    E0 = solver.energy()
    entropy_var0 = solver.integrate(0.5 * solver.h * solver.s ** 2)
    water_var0 = solver.integrate(0.5 * solver.h * solver.q**2)

    for _ in range(nteps):
        solver.time_step(dt=dt)

    E1 = solver.energy()
    entropy_var1 = solver.integrate(0.5 * solver.h * solver.s ** 2)
    water_var1 = solver.integrate(0.5 * solver.h * solver.q ** 2)

    energy_errors.append(abs(E1 - E0) / E0)
    entropy_var_errors.append(abs(entropy_var1 - entropy_var0) / entropy_var0)
    water_var_errors.append(abs(water_var1 - water_var0) / water_var0)

    print(solver.first_water_limit_time)
    print(solver.ql.max())
    print()

print(energy_errors)
print(entropy_var_errors)
print(water_var_errors)

plt.loglog(dts, energy_errors, label='Energy')
plt.loglog(dts, entropy_var_errors, label='Entropy variance')
plt.loglog(dts, water_var_errors, label='Water variance')

line_3rd = (dts / dts.max()) **3 * (0.75 * energy_errors[-1])
line_2nd = (dts / dts.max()) **2 * (1.25 * water_var_errors[-1])

plt.loglog(dts, line_2nd, linestyle='dotted', color='black', label='2nd order')
plt.loglog(dts, line_3rd, '--', color='black', label='3rd order')

plt.ylabel('Relative error')
plt.xlabel('Timestep (s)')
plt.legend()

plot_dir = 'plots'
if not os.path.exists(plot_dir): os.makedirs(plot_dir)
plt.savefig(os.path.join(plot_dir, f'two-phase-conservation_convergence-nx{nx}-nz{nz}-p{poly_order}.png'))

plt.show()

# nsteps =
#
# for cfl in
#
# if run_model:
#     solver = ThreePhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=cfl, a=a, nz=nz, upwind=upwind, nprocx=nproc)
#     u, v, density, s, qw, qv, ql, qi = initial_condition(solver.xs, solver.zs, solver, pert=2.0)