from matplotlib import pyplot as plt
from moist_euler_dg.single_species_euler_2D import SingleSpeciesEuler2D
from moist_euler_dg.equilibrium_euler_2D import EquilibriumEuler2D
import numpy as np
import time
import os
import argparse

exp_name = 'noniso-2D-moist-bubble'

parser = argparse.ArgumentParser()
# Optional argument
parser.add_argument('--n', type=int, help='Number of cells')
args = parser.parse_args()

if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

run_model = True # whether to run model - set false to just plot previous run
start_idx = -1 # set to 0, 1, 2, 3 to start model from part way through previous run
dev = 'cpu'
xlim = 20_000
ylim = 10_000

nx = ny = args.n
eps = 0.8
g = 9.81
poly_order = 3

print(f"\n\n\n ---------- Moist bubble with nx={nx}, ny={ny}, cfl={eps}")

#
plot_dir = f'./plots/{exp_name}-n{nx}-p{poly_order}'
if not os.path.exists(plot_dir): os.makedirs(plot_dir)

angle = 0 * (np.pi / 180)
solver = EquilibriumEuler2D(
    (0.0, xlim), (0, ylim), poly_order, nx, ny, g=g,
    eps=eps, device=dev, solution=None, a=0.0,
    dtype=np.float64, angle=angle, a_bdry=0.5
)

def initial_condition(xs, ys, solver):

    u = 0 * ys
    v = 0 * ys

    dry_theta = 300
    dexdy = -g  / (solver.cpd * dry_theta)
    ex = 1 + dexdy * ys
    p = 1_00_000.0 * ex**(solver.cpd / solver.Rd)
    density = p / (solver.Rd * ex * dry_theta)
    qw = 0.02 + 0.0 * ys

    print('exner min-max:', ex.min(), ex.max())
    print('Density min-max:', density.min(), density.max())
    print('Pressure min-max:', p.min(), p.max())

    qv = solver.solve_qv_from_p(density, qw, p)
    print('qv min-max:', qv.min(), qv.max(), '\n')

    qd = 1 - qw
    ql = qw - qv
    R = qv * solver.Rv + qd * solver.Rd

    T = p / (density * R)
    s = qd * solver.entropy_air(T, qd, density)
    s += qv * solver.entropy_vapour(T, qv, density)
    s += ql * solver.entropy_liquid(T)

    # qv2 = solver.solve_qv_from_entropy(density, qw, s, verbose=True)
    # print('qv min-max:', qv2.min(), qv2.max())
    # print('qv error:', (abs(qv2 - qv) / qv).max(), '\n')
    #
    moist_pt = solver.get_moist_pt(s, qw)

    rad_max = 2_000
    rad = np.sqrt((xs - 0.5 * xlim) ** 2 + (ys - 1.1*rad_max) ** 2)
    mask = rad < rad_max
    moist_pt += mask * 2 * (np.cos(np.pi * (rad / rad_max) / 2)**2)

    s = solver.moist_pt2entropy(moist_pt, qw)
    #
    # # just for diagnostic info - can remove
    # qv = solver.solve_qv_from_entropy(density, qw, s, verbose=True)
    # qd = 1 - qw
    # ql = qw - qv
    # R = qv * solver.Rv + qd * solver.Rd
    # cv = qd * solver.cvd + qv * solver.cvv + ql * solver.cl
    # cp = qd * solver.cpd + qv * solver.cpv + ql * solver.cl
    # logT = (1 / cv) * (s + R * np.log(density) + qd * solver.Rd * np.log(solver.Rd * qd) + qv * solver.Rv * np.log(solver.Rv * qv) - qv * solver.c0 - ql * solver.c1)
    # T = np.exp(logT)
    #
    # p = density * R * T
    #
    # print('T min-max:', T.min() - 273, T.max() - 273)
    # print('Density min-max:', density.min(), density.max())
    # print('Pressure min-max:', p.min(), p.max())
    # print('qv min-max:', qv.min(), qv.max(), '\n')

    hqw = density * qw
    return u, v, density, s * density, hqw, qv


run_time = 2000

if start_idx >= 0:
    solver.load_restarts(f"./data/{exp_name}-n{nx}-p{poly_order}-part-{start_idx}")
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
        solver.save_restarts(f"./data/{exp_name}-n{nx}-p{poly_order}-part-{i}")


for i in range(4):
    solver.load_restarts(f"./data/{exp_name}-n{nx}-p{poly_order}-part-{i}")

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
plt.savefig(f'{plot_dir}/{exp_name}-conservation-n{nx}-p{poly_order}.png')

plt.figure(4)
plt.title("Total water vapour")
plt.plot(solver.diagnostics['time'], solver.diagnostics['vapour'])
plt.savefig(f'{plot_dir}/{exp_name}-vapour-n{nx}-p{poly_order}.png')

plt.figure(5)
plt.title("Gibbs error")
plt.plot(solver.diagnostics['time'], solver.diagnostics['gibbs_error'])
plt.savefig(f'{plot_dir}/{exp_name}-gibbs_error-n{nx}-p{poly_order}.png')

plt.figure(6)
plt.title("dEdt")
plt.plot(solver.diagnostics['time'], solver.diagnostics['dEdt'])
plt.savefig(f'{plot_dir}/{exp_name}-dEdt-n{nx}-p{poly_order}.png')

# Relative energy change: -8.678657871011961e-08

