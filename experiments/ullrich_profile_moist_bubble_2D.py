from matplotlib import pyplot as plt
from moist_euler_dg.single_species_euler_2D import SingleSpeciesEuler2D
from moist_euler_dg.equilibrium_euler_2D import EquilibriumEuler2D
import numpy as np
import time
import os
import argparse

exp_name = 'ullrich-2D-moist-bubble'

parser = argparse.ArgumentParser()
# Optional argument
parser.add_argument('--n', type=int, help='Number of cells')
args = parser.parse_args()

if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

run_model = True # whether to run model - set false to just plot previous run
start_idx = -1 # set to 0, 1, 2, 3 to start model from part way through previous run
dev = 'cpu'
xlim = 10_000
ylim = 30_000

nx = args.n
ny = 3 * nx
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


def initial_condition(xs, ys, solver, pert):

    u = 0 * ys
    v = 0 * ys

    TE = 310.0
    TP = 240.0
    GRAVITY = solver.g
    T0 = 0.5*(TE+TP)
    b = 2.0
    KP = 3.0
    GAMMA = 0.005
    P0 = 100000
    RD = solver.Rd

    A = 1.0/GAMMA
    B = (TE - TP)/((TE + TP)*TP)
    C = 0.5*(KP + 2.0)*(TE - TP)/(TE*TP)
    H = RD*T0/GRAVITY

    fac   = ys/(b*H)
    fac2  = fac*fac
    cp    = np.cos(2.0*np.pi/9.0)
    cpk   = np.power(cp, KP)
    cpkp2 = np.power(cp, KP+2)
    fac3  = cpk - (KP/(KP+2.0))*cpkp2

    torr_1 = (A*GAMMA/T0)*np.exp(GAMMA*(ys)/T0) + B*(1.0 - 2.0*fac2)*np.exp(-fac2)
    torr_2 = C*(1.0 - 2.0*fac2)*np.exp(-fac2)

    int_torr_1 = A*(np.exp(GAMMA*ys/T0) - 1.0) + B*ys*np.exp(-fac2)
    int_torr_2 = C*ys*np.exp(-fac2)

    tempInv = torr_1 - torr_2*fac3
    T = 1.0 / tempInv
    p = P0*np.exp(-GRAVITY*int_torr_1/RD + GRAVITY*int_torr_2*fac3/RD)
    density = p / (solver.Rd * T)

    qw = 0.02 + 0.0 * ys

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

    moist_pt = solver.get_moist_pt(s, qw)

    rad_max = 2_000
    rad = np.sqrt((xs - 0.5 * xlim) ** 2 + (ys - 1.1 * rad_max) ** 2)
    mask = rad < rad_max
    moist_pt += mask * pert * (np.cos(np.pi * (rad / rad_max) / 2) ** 2)

    s = solver.moist_pt2entropy(moist_pt, qw)

    hqw = density * qw
    return u, v, density, s * density, hqw, qv


run_time = 500

u, v, density, hs, hqw, qv = initial_condition(solver.xs, solver.ys, solver, pert=0.0)
solver.set_initial_condition(u, v, density, hs, hqw)
base_moist_pt = solver.get_moist_pt()
base_entropy = solver.state['hs'] / solver.state['h']

# fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
# im = solver.plot_solution(ax, dim=2, plot_func=lambda s: s.project_H1(s.get_moist_pt()))
# cbar = plt.colorbar(im, ax=ax)
# cbar.ax.tick_params(labelsize=8)
# plt.show()
# exit(0)

if start_idx >= 0:
    solver.load_restarts(f"./data/{exp_name}-n{nx}-p{poly_order}-part-{start_idx}")
    solver.time = (start_idx + 1) * run_time / 4
else:
    u, v, density, hs, hqw, qv = initial_condition(solver.xs, solver.ys, solver, pert=10.0)
    solver.set_initial_condition(u, v, density, hs, hqw)

E0 = solver.integrate(solver.energy()).numpy()
print('Energy:', E0)


plot_func = lambda s: s.project_H1(s.get_moist_pt() - base_moist_pt)
plot_func_2 = lambda s: s.project_H1((s.state['hs'] / s.state['h']) - base_entropy)

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
    # ax.set_xlim(0.25 * xlim, 0.75 * xlim)
    im = solver.plot_solution(ax, dim=2, vmin=vmin, vmax=vmax, plot_func=plot_func)
    cbar = plt.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=8)

    ax = axs2[i // 2][i % 2]
    # ax.set_xlim(0.25 * xlim, 0.75 * xlim)
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
