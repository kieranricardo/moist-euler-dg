from matplotlib import pyplot as plt
import matplotlib.animation as animation
from moist_euler_dg.dry_euler_2D import DryEuler2D
from matplotlib.animation import FFMpegWriter as MovieWriter
import numpy as np
import torch
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

plot_func = lambda s: (s.hb / s.h)

vmin = 299.8
vmax = 301.2

# fig, ax = plt.subplots(1, 1, sharex=True, sharey=True)
# im = solver.plot_solution(ax, dim=2, vmin=vmin, vmax=vmax, plot_func=plot_func)
# cbar = plt.colorbar(im, ax=ax)
# cbar.ax.tick_params(labelsize=8)
# plt.show()
# exit(0)

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

E = solver.integrate(solver.energy()).numpy()
print('Energy:', E)
print("Relative energy change:", (E - E0) / E0)

plt.savefig(f'./plots/2D-dry-bubble-snaps-n{nx}-p{poly_order}.png')
plt.show()
