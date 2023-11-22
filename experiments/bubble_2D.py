from matplotlib import pyplot as plt
import matplotlib.animation as animation
from moist_euler_dg.euler_2D import Euler2D
from moist_euler_dg.dry_euler_2D import DryEuler2D
from matplotlib.animation import FFMpegWriter as MovieWriter
import numpy as np
import torch
import time
import os

if not os.path.exists('./plots'): os.makedirs('./plots')
if not os.path.exists('./data'): os.makedirs('./data')

plt.rcParams['font.size'] = '12'

mode = 'snap'

dev = 'cpu'
xlim = 2_000
ylim = 2_000

nx = ny = 10
eps = 0.125 * 0.5 * 0.5
g = 9.81
poly_order = 3
#
angle = 0 * (np.pi / 180)
solver = DryEuler2D(
    (-0.5 * xlim, 0.5 * xlim), (0, ylim), poly_order, nx, ny, g=g,
    eps=eps, device=dev, solution=None, a=0.0,
    dtype=np.float64, angle=angle
)

def initial_condition(xs, ys, R, cp, cv, p0, g, gamma):

    u = 0 * xs
    v = 0 * xs
    b = 300

    dexdy = -g / 300

    density_0 = 1.2

    const = cp * (R * 300 / p0) ** (R / cv)
    ex0 = const * density_0 ** (R / cv) # surface density is 1.2 kg/m^3
    ex = ex0 + ys * dexdy

    density = (ex / const) ** (cv / R)

    rad = np.sqrt(xs ** 2 + (ys - 260) ** 2)
    mask = rad < 250
    b += mask * 0.5 * (1 + np.cos(np.pi * rad / 250.0))

    p = (b * R * density)**gamma * (1 / p0)**(R / cv)
    s = cv * np.log(p / density**gamma)

    print('Pressure min:', p.min())
    return u, v, density, s * density

u, v, density, sb = initial_condition(
    solver.xs,
    solver.ys,
    solver.R,
    solver.cp,
    solver.cv,
    solver.p0,
    solver.g,
    solver.gamma
)

solver.set_initial_condition(u, v, density, sb)

E0 = solver.integrate(solver.energy()).numpy()
print('Energy:', E0)

# plot_func = lambda s: s.project_H1(s.hb / s.h)
plot_func = lambda s: (s.hb / s.h)

vmin = 299.8
vmax = 301.2

# fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
# im = solver.plot_solution(ax, dim=2, vmin=vmin, vmax=vmax, plot_func=plot_func)
# cbar = plt.colorbar(im, ax=ax)
# cbar.ax.tick_params(labelsize=8)
# plt.show()
# exit(0)

def update_plot(frame_number, plot, solver, ax):
    for _ in range(iplot):
        solver.time_step(order=3)
    # plot[0].remove()
    for c in plot[0].collections:
        c.remove()
    plot[0] = solver.plot_solution(ax, vmin=vmin, vmax=vmax, dim=2, plot_func=plot_func)

if mode == 'run':
    sim_time = 0
    ke = []
    ie = []
    pe = []
    times = []
    while solver.time < 100:
        ke.append(solver.integrate(solver.ke()).numpy())
        pe.append(solver.integrate(solver.pe()).numpy())
        ie.append(solver.integrate(solver.ie()).numpy())
        times.append(solver.time)
        solver.time_step()

    ke = np.array(ke)
    pe = np.array(pe)
    ie = np.array(ie)
    plt.plot(times, ke - ke[0], label='K')
    plt.plot(times, pe - pe[0], label='P')
    plt.plot(times, ie - ie[0], label='I')
    plt.ylabel('E - E(t=0)')
    plt.xlabel('Time (s)')
    plt.legend()
    plt.savefig('./plots/2D-bubble-energy.png')
    plt.show()

elif mode == 'plot':

    iplot = 100
    fig = plt.figure()
    ax = fig.add_subplot(111)#, projection="3d")
    # ax.set_zlim(290, 310)
    #ax.set_zlim(0.8 * solver.hb.min(), 1.2 * solver.hb.max())
    # ax.set_zlim(7, 9)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    # ax.set_zlabel("h")
    plot = [solver.plot_solution(ax, vmin=vmin, vmax=vmax, plot_func=plot_func, dim=2)]
    plt.colorbar(plot[0])
    ani = animation.FuncAnimation(
        fig, update_plot, 1, fargs=(plot, solver, ax), interval=10
    )

    plt.show()
    print("Simulation time (unit less):", solver.time)

elif mode == 'snap':
    fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
    T = 100

    for i in range(4):

        tend = solver.time + (T / 4)
        t0 = time.time()
        while solver.time <= tend:
            solver.time_step()

        t1 = time.time()
        print('Walltime:', t1 - t0, "s. Simulation time: ", solver.time, "s.")
        ax = axs[i // 2][i % 2]

        im = solver.plot_solution(ax, dim=2, vmin=vmin, vmax=vmax, plot_func=plot_func)
        cbar = plt.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)
    plt.savefig(f'./plots/2D-dry-bubble-snaps-n{nx}-p{poly_order}.png')
    # plt.show()

if mode == 'movie':
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    plot = [solver.plot_solution(ax1, plot_func=plot_func, dim=2, vmin=vmin, vmax=vmax)]
    plt.colorbar(plot[0])

    moviewriter = MovieWriter(fps=30)
    iplot = 100
    tend = 1000
    with moviewriter.saving(fig1, f"../out/bubble_2D_ncells_{nx}_order_{poly_order}_time_{int(solver.time)}.mp4", dpi=100):
        while solver.time <= tend:
            update_plot(0, plot, solver, ax1)
            moviewriter.grab_frame()

    np.save(
        f"./data/u_bubble_2D_ncells_{nx}_order_{poly_order}_time_{int(solver.time)}.npy",
        solver.u.numpy()
    )
    np.save(
        f"./data/v_bubble_2D_ncells_{nx}_order_{poly_order}_time_{int(solver.time)}.npy",
        solver.v.numpy()
    )
    np.save(
        f"./data/density_bubble_2D_ncells_{nx}_order_{poly_order}_time_{int(solver.time)}.npy",
        solver.h.numpy()
    )

    np.save(
        f"./data/dens_pot_temp_bubble_2D_ncells_{nx}_order_{poly_order}_time_{int(solver.time)}.npy",
        solver.h.numpy()
    )

E = solver.integrate(solver.energy()).numpy()
print('Energy:', E)
print("Relative energy change:", (E - E0) / E0)
