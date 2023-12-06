from matplotlib import pyplot as plt
from moist_euler_dg.single_species_euler_2D import SingleSpeciesEuler2D
from moist_euler_dg.equilibrium_euler_2D import EquilibriumEuler2D
import numpy as np
import time
import os
import argparse

parser = argparse.ArgumentParser()
# Optional argument
parser.add_argument('--n', type=int, help='Number of cells')
parser.add_argument('--cfl', type=float, help='Number of cells')
args = parser.parse_args()

if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'


dev = 'cpu'
xlim = 20_000
ylim = 10_000

nx = ny = 3
g = 9.81
poly_order = 3

plot_dir = f'./plots'
if not os.path.exists(plot_dir): os.makedirs(plot_dir)

solver = EquilibriumEuler2D(
    (0.0, xlim), (0, ylim), poly_order, nx, ny, g=g,
    eps=0.8, device=dev, solution=None, a=0.0, upwind=False
)


def initial_condition(xs, ys, solver):

    u = 0 * ys
    v = 0 * ys

    density_ground = 1.2
    p_ground = 1_00_000.0
    qw_ground = 0.02

    qv_ground = solver.solve_qv_from_p(density_ground, qw_ground, p_ground)
    R_ground = (1 - qw_ground) * solver.Rd + qv_ground * solver.Rv
    cp_ground = (1 - qw_ground) * solver.cpd + qv_ground * solver.cpv + (qw_ground - qv_ground) * solver.cl
    T_ground = p_ground / (density_ground * R_ground)

    entropy_ground = (1 - qw_ground) * solver.entropy_air(T_ground, 1 - qw_ground, density_ground)
    entropy_ground += qv_ground * solver.entropy_vapour(T_ground, qv_ground, density_ground)
    entropy_ground += (qw_ground - qv_ground) * solver.entropy_liquid(T_ground)
    moist_pt_ground = solver.get_moist_pt(entropy_ground, qw_ground)

    enthalpy_ground = cp_ground * T_ground + qv_ground * solver.Lv0

    enthalpy = enthalpy_ground - solver.g * ys
    s = entropy_ground + 0 * ys
    qw = qw_ground + 0 * ys
    qv = solver.solve_qv_from_enthalpy(enthalpy, qw, s, verbose=True)

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

    moist_pt = solver.get_moist_pt(s, qw)
    rad_max = 2_000
    rad = np.sqrt((xs - 0.5 * xlim) ** 2 + (ys - 1.1*rad_max) ** 2)
    mask = rad < rad_max
    moist_pt += mask * 2 * (np.cos(np.pi * (rad / rad_max) / 2)**2)

    s = solver.moist_pt2entropy(moist_pt, qw)

    hqw = density * qw
    return u, v, density, s * density, hqw, qv


u, v, density, hs, hqw, qv = initial_condition(
    solver.xs,
    solver.ys,
    solver
)


cfls = [0.1, 0.2, 0.3, 0.4, 0.5]
names = ['energy', 'entropy_variance', 'water_variance']
errors = dict((name, []) for name in names)

for eps in cfls:
    solver = EquilibriumEuler2D(
        (0.0, xlim), (0, ylim), poly_order, nx, ny, g=g,
        eps=eps, device=dev, solution=None, a=0.0, upwind=False
    )
    solver.set_initial_condition(u, v, density, hs, hqw)

    while solver.time < 1000:
        solver.time_step()

    for name in names:
        errors[name].append(abs(solver.diagnostics[name][0] - solver.diagnostics[name][-1]) / solver.diagnostics[name][0])


for name in names:
    plt.loglog(cfls, errors[name], label=name.replace('_', ' '))

plt.yscale('symlog', linthresh=1e-15)
plt.ylim([-1e-16, 1.1 * max(errors['energy'])])

plt.xlabel('CFL')
plt.ylabel('Relative error')
plt.legend()
plt.savefig('./plots/conservation_test.png')
plt.show()

