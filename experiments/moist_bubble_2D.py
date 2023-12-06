from matplotlib import pyplot as plt
from moist_euler_dg.single_species_euler_2D import SingleSpeciesEuler2D
from moist_euler_dg.equilibrium_euler_2D import EquilibriumEuler2D
import numpy as np
import time
import os
import argparse

exp_name = '2D-moist-bubble'

parser = argparse.ArgumentParser()
# Optional argument
parser.add_argument('--n', type=int, help='Number of cells')
parser.add_argument('--cfl', type=float, help='Number of cells')
args = parser.parse_args()

if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

run_model = True # whether to run model - set false to just plot previous run
start_idx = -1 # set to 0, 1, 2, 3 to start model from part way through previous run
dev = 'cpu'
xlim = 20_000
ylim = 10_000

nx = ny = args.n
eps = args.cfl
g = 9.81
poly_order = 3

print(f"\n\n\n ---------- Moist bubble with nx={nx}, ny={ny}, cfl={eps}")

#
plot_dir = f'./plots/{exp_name}-n{nx}-p{poly_order}-cfl{eps}'
if not os.path.exists(plot_dir): os.makedirs(plot_dir)


angle = 0 * (np.pi / 180)
solver = EquilibriumEuler2D(
    (0.0, xlim), (0, ylim), poly_order, nx, ny, g=g,
    eps=eps, device=dev, solution=None, a=0.0,
    dtype=np.float64, angle=angle
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
    print(f'Background moist PT: {moist_pt_ground} K')

    enthalpy_ground = cp_ground * T_ground + qv_ground * solver.Lv0



    enthalpy = enthalpy_ground - solver.g * ys
    s = entropy_ground + 0 * ys
    qw = qw_ground + 0 * ys

    print('Enthalpy min-max:', enthalpy.min(), enthalpy.max())

    qv = solver.solve_qv_from_enthalpy(enthalpy, qw, s, verbose=True)
    print('qv min-max:', qv.min(), qv.max(), '\n')

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

    qv2 = solver.solve_qv_from_entropy(density, qw, s, verbose=True)
    print('qv min-max:', qv2.min(), qv2.max())


    print('qv error:', (abs(qv2 - qv) / qv).max(), '\n')

    moist_pt = solver.get_moist_pt(s, qw)

    rad_max = 2_000
    rad = np.sqrt((xs - 0.5 * xlim) ** 2 + (ys - 1.1*rad_max) ** 2)
    mask = rad < rad_max
    moist_pt += mask * 2 * (np.cos(np.pi * (rad / rad_max) / 2)**2)
    # moist_pt += mask * 0.5 * (1 + np.cos(np.pi * rad / 250.0))

    s = solver.moist_pt2entropy(moist_pt, qw)

    # just for diagnostic info - can remove
    qv = solver.solve_qv_from_entropy(density, qw, s, verbose=True)
    print('qv min-max:', qv.min(), qv.max())

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
    print('qv min-max:', qv.min(), qv.max(), '\n')

    hqw = density * qw
    return u, v, density, s * density, hqw, qv


run_time = 2000

if start_idx >= 0:
    solver.load_restarts(f"./data/{exp_name}-n{nx}-p{poly_order}-cfl{eps}-part-{start_idx}")
    solver.time = (start_idx + 1) * run_time / 4
else:
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

vmin = vmax = None

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
fig2, axs2 = plt.subplots(2, 2, sharex=True, sharey=True)

fig.suptitle('Moist equivalent potential temperature')
fig2.suptitle('Specific entropy')

if run_model:
    for i in range(start_idx+1, 4):

        tend = solver.time + (run_time / 4)
        t0 = time.time()
        while solver.time <= tend:
            solver.time_step()
        t1 = time.time()
        print('Walltime:', t1 - t0, "s. Simulation time: ", solver.time, "s.")
        solver.save_restarts(f"./data/{exp_name}-n{nx}-p{poly_order}-cfl{eps}-part-{i}")


for i in range(4):
    solver.load_restarts(f"./data/{exp_name}-n{nx}-p{poly_order}-cfl{eps}-part-{i}")

    ax = axs[i // 2][i % 2]
    ax.set_xlim(0.25 * xlim, 0.75 * xlim)
    im = solver.plot_solution(ax, dim=2, vmin=vmin, vmax=vmax, plot_func=plot_func)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)

    ax = axs2[i // 2][i % 2]
    ax.set_xlim(0.25 * xlim, 0.75 * xlim)
    im = solver.plot_solution(ax, dim=2, plot_func=plot_func_2)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)

E = solver.integrate(solver.energy()).numpy()
print('Energy:', E)
print("Relative energy change:", (E - E0) / E0)
print('Estimated relative energy change:', np.array(solver.diagnostics['dEdt']).mean() * solver.time / E0)

fig.savefig(f'{plot_dir}/{exp_name}-snaps-n{nx}-p{poly_order}.png')
fig2.savefig(f'{plot_dir}/{exp_name}-snaps-entropy-n{nx}-p{poly_order}.png')

plt.figure(3)
names = ['entropy_variance', 'water_variance', 'energy']

for name in names:
    arr = np.array(solver.diagnostics[name])
    arr = (arr - arr[0]) / arr[0]
    plt.plot(solver.diagnostics['time'], arr, label=name.replace('_', ' '))
    plt.yscale('symlog', linthresh=1e-15)

plt.legend()
plt.ylabel("Relative error")
plt.xlabel("Time (s)")
plt.savefig(f'{plot_dir}/{exp_name}-conservation-n{nx}-p{poly_order}-cfl{solver.eps}.png')

plt.figure(4)
plt.title("Total water vapour")
plt.plot(solver.diagnostics['time'], solver.diagnostics['vapour'])
plt.savefig(f'{plot_dir}/{exp_name}-vapour-n{nx}-p{poly_order}-cfl{solver.eps}.png')

plt.figure(5)
plt.title("Gibbs error")
plt.plot(solver.diagnostics['time'], solver.diagnostics['gibbs_error'])
plt.savefig(f'{plot_dir}/{exp_name}-gibbs_error-n{nx}-p{poly_order}-cfl{solver.eps}.png')

plt.figure(6)
plt.title("dEdt")
plt.plot(solver.diagnostics['time'], solver.diagnostics['dEdt'])
plt.savefig(f'{plot_dir}/{exp_name}-dEdt-n{nx}-p{poly_order}-cfl{solver.eps}.png')
#
# solver.save_restarts(f"./data/2D-moist-bubble-n{nx}-p{poly_order}-cfl{eps}")
# plt.show()

# Outputs should be:
# Energy: 1222199635921.0732
# Energy: 1222199529850.5483
# Relative energy change: -8.678657871011961e-08

