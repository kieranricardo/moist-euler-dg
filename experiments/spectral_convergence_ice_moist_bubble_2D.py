from matplotlib import pyplot as plt
from moist_euler_dg.fortran_three_phase_euler_2D import FortranThreePhaseEuler2D as ThreePhaseEuler2D
import numpy as np
import os
import matplotlib.ticker as ticker
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--n', type=int, help='Number of cells')
parser.add_argument('--nproc', type=int, help='Number of procs')
args = parser.parse_args()

exp_name_short = 'ice-bubble'
xlim = 10_000
zlim = 10_000
cfl = 0.5
g = 9.81
a = 0.5
upwind = True
zmap = lambda x, z: z * zlim
xmap = lambda x, z: xlim * (x - 0.5)

nx = nz = args.n
nproc = args.nproc
orders = [3, 4, 5, 6]
tend = 600

plot_dir = os.path.join('plots', f'{exp_name_short}-spectral-convergence-nx{nx}-nz{nz}')
if not os.path.exists(plot_dir): os.makedirs(plot_dir)

labels = ['entropy']
plot_funcs = [lambda s: s.project_H1(s.s)]

for label, plot_func in zip(labels, plot_funcs):
  fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)

  for i, order in enumerate(orders):
    experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{order}'
    data_dir = os.path.join('data', experiment_name)

    solver = ThreePhaseEuler2D(xmap, zmap, order, nx, g=g, cfl=0.5, a=a, nz=nz, upwind=upwind, nprocx=1)

    filepaths = [solver.get_filepath(data_dir, exp_name_short, proc=i, nprocx=nproc, time=tend) for i in range(nproc)]
    solver.load(filepaths)

    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    ax = axs[i // 2][i % 2]
    ax.tick_params(labelsize=8)
    im = solver.plot_solution(ax, dim=2, plot_func=plot_func)
    cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt))
    cbar.ax.tick_params(labelsize=8)

  fn = f'{label}_{exp_name_short}_nx{nx}_nz{nz}.png'
  plt.savefig(os.path.join(plot_dir, fn))
