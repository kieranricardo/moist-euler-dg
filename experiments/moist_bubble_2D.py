from matplotlib import pyplot as plt
from moist_euler_dg.single_species_euler_2D import SingleSpeciesEuler2D
from moist_euler_dg.equilibrium_euler_2D import EquilibriumEuler2D
import numpy as np
import time
import os

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'


dev = 'cpu'
xlim = 2_000
ylim = 2_000

nx = ny = 10
eps = 0.8
g = 9.81
poly_order = 3
#
angle = 0 * (np.pi / 180)
solver = EquilibriumEuler2D(
    (-0.5 * xlim, 0.5 * xlim), (0, ylim), poly_order, nx, ny, g=g,
    eps=eps, device=dev, solution=None, a=0.0,
    dtype=np.float64, angle=angle
)



def initial_condition(xs, ys, solver):

    u = 0 * ys
    v = 0 * ys

    density_ground = 1.2
    p_ground = 1_10_000.0
    qw_ground = 0.2
    qv_ground = solver.solve_qv_from_p(np.array([density_ground]), np.array([qw_ground]), np.array([p_ground]))[0]

    R_ground = (1 - qw_ground) * solver.Rd + qv_ground * solver.Rv
    cp_ground = (1 - qw_ground) * solver.cpd + qv_ground * solver.cpv + (qw_ground - qv_ground) * solver.cl
    T_ground = p_ground / (density_ground * R_ground)

    enthalpy_ground = cp_ground * T_ground + qv_ground * solver.Lv0

    entropy_ground = (1 - qw_ground) * solver.entropy_air(T_ground, 1 - qw_ground, density_ground)
    entropy_ground += qv_ground * solver.entropy_vapour(T_ground, qv_ground, density_ground)
    entropy_ground += (qw_ground - qv_ground) * solver.entropy_liquid(T_ground)

    enthalpy = enthalpy_ground - solver.g * ys
    s = entropy_ground + 0 * ys
    qw = qw_ground + 0 * ys

    print('Enthalpy min-max:', enthalpy.min(), enthalpy.max())

    qv = solver.solve_qv_from_enthalpy(enthalpy, qw, s)

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

    qv2 = solver.solve_qv_from_entropy(density, qw, s)

    print('qv error:', (abs(qv2 - qv) / qv).max())

    moist_pt = solver.get_moist_pt(s, qw)
    rad = np.sqrt(xs ** 2 + (ys - 260) ** 2)
    mask = rad < 250
    moist_pt += mask * 0.5 * (1 + np.cos(np.pi * rad / 250.0))
    s = solver.moist_pt2entropy(moist_pt, qw)

    qv = solver.solve_qv_from_entropy(density, qw, s)
    qd = 1 - qw
    ql = qw - qv
    R = qv * solver.Rv + qd * solver.Rd
    cv = qd * solver.cvd + qv * solver.cvv + ql * solver.cl
    cp = qd * solver.cpd + qv * solver.cpv + ql * solver.cl
    logT = (1 / cv) * (s + R * np.log(density) + qd * solver.Rd * np.log(solver.Rd * qd) + qv * solver.Rv * np.log(solver.Rv * qv) - qv * solver.c0 - ql * solver.c1)
    T = np.exp(logT)

    p = density * R * T

    print('T min-max:', T.min() - 273, T.max() - 273)
    print('Density min-max:', density.min(), density.max())
    print('Pressure min-max:', p.min(), p.max())
    print('qv min-max:', qv.min(), qv.max())

    hqw = density * qw
    return u, v, density, s * density, hqw, qv

# def initial_condition(xs, ys, Rd, cpd, cvd, p0, g, gamma, cvv, Rv, cl, solver):
#
#     u = 0 * xs
#     v = 0 * xs
#     b = 300
#
#     # keep equivalent potential temperature 320K
#     # set q to
#     # p = h * R * T
#     # (dpdy / h) = -g
#     # R * dTdy + dlog(h)dy * R * T
#
#     dexdy = -g / 300
#
#     density_0 = 1.2
#
#     const = cpd * (Rd * 300 / p0) ** (Rd / cvd)
#     ex0 = const * density_0 ** (Rd / cvd) # surface density is 1.2 kg/m^3
#     ex = ex0 + ys * dexdy
#
#     density = (ex / const) ** (cvd / Rd)
#
#     rad = np.sqrt(xs ** 2 + (ys - 260) ** 2)
#     mask = rad < 250
#     b += mask * 0.5 * (1 + np.cos(np.pi * rad / 250.0))
#
#     p = (b * Rd * density)**gamma * (1 / p0)**(Rd / cvd)
#
#     # go from p, density, qw to entropy
#     # get temperature?
#     qw = 0.02 + 0 * xs
#     qd = 1.0 - qw
#
#     qv = solver.solve_qv_from_p(density, qw, p)
#     qv = np.minimum(qv, qw)
#     ql = qw - qv
#     R = qv * Rv + qd * Rd
#     T = p / (density * R)
#
#     # calculate entropy
#     s = qd * solver.entropy_air(T, qd, density) + qv * solver.entropy_vapour(T, qv, density) + ql * solver.entropy_liquid(T)
#
#     hqw = density * qw
#     # T^(cv*) = exp(s) density^(R*) qd^(qd Rd) qv^(qv Rv)
#     print('Input qv min max:', qv.min(), qv.max())
#     print('Input ql min max:', ql.min(), ql.max())
#     print('Input T min max:', T.min(), T.max())
#     print('Input P min max:', p.min(), p.max(), '\n')
#     return u, v, density, s * density, hqw, qv

u, v, density, hs, hqw, qv = initial_condition(
    solver.xs,
    solver.ys,
    solver
)


solver.set_initial_condition(u, v, density, hs, hqw)

E0 = solver.integrate(solver.energy()).numpy()
print('Energy:', E0)

plot_func = lambda s: s.project_H1(s.get_moist_pt())
plot_func_2 = lambda s: s.project_H1(s.state['hs'] / s.state['h'])

# vmin = 299.8
# vmax = 301.2

vmin = vmax = None

# fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
# im = solver.plot_solution(ax, dim=2, vmin=vmin, vmax=vmax, plot_func=plot_func)
# cbar = plt.colorbar(im, ax=ax)
# cbar.ax.tick_params(labelsize=8)
# plt.show()
# exit(0)

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
fig2, axs2 = plt.subplots(2, 2, sharex=True, sharey=True)

fig.suptitle('Pressure')
fig2.suptitle('Specific entropy')

T = 360

ax = axs[0][0]
im = solver.plot_solution(ax, dim=2, vmin=vmin, vmax=vmax, plot_func=plot_func)
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=8)

ax = axs2[0][0]
im = solver.plot_solution(ax, dim=2, plot_func=plot_func_2)
cbar = plt.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=8)

for i in range(1, 4):

    tend = solver.time + (T / 3)
    t0 = time.time()
    while solver.time <= tend:
        solver.time_step()
    t1 = time.time()
    print('Walltime:', t1 - t0, "s. Simulation time: ", solver.time, "s.")

    ax = axs[i // 2][i % 2]
    im = solver.plot_solution(ax, dim=2, vmin=vmin, vmax=vmax, plot_func=plot_func)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)

    ax = axs2[i // 2][i % 2]
    im = solver.plot_solution(ax, dim=2, plot_func=plot_func_2)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)

E = solver.integrate(solver.energy()).numpy()
print('Energy:', E)
print("Relative energy change:", (E - E0) / E0)


fig.savefig(f'./plots/2D-moist-bubble-snaps-n{nx}-p{poly_order}.png')
fig2.savefig(f'./plots/2D-moist-bubble-snaps-entropy-n{nx}-p{poly_order}.png')

# plt.show()

# Outputs should be:
# Energy: 1222199635921.0732
# Energy: 1222199529850.5483
# Relative energy change: -8.678657871011961e-08

