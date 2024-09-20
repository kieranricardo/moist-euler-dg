from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter as MovieWriter
import matplotlib.animation as animation
from mpi4py import MPI
import numpy as np
import time
import os
import argparse


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument('--order', type=int, help='Polynomial order')
parser.add_argument('--nx', type=int, help='Number of cells in horizontal')
parser.add_argument('--nz', type=int, help='Number of cells in vertical')
args = parser.parse_args()
nz = args.nz
nx = args.nx
poly_order = args.order # spatial order of accuracy

exp_name_short = 'forced-convection'
experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{poly_order}'

plot_dir = os.path.join('plots', experiment_name)
data_dump_dir = os.path.join(plot_dir, 'data-dump')
movie_dir = os.path.join(plot_dir, 'movies')

if not os.path.exists(movie_dir): os.makedirs(movie_dir)

label = 'xcoord'
fp = os.path.join(data_dump_dir, f'{label}.npy')
xs = np.load(fp)

label = 'zcoord'
fp = os.path.join(data_dump_dir, f'{label}.npy')
zs = np.load(fp)

def update_plot(frame_number, plot, ax):

    global idx
    global vmin
    global vmax

    plot[0].remove()
    plot[0] = ax.tricontourf(xs.ravel(), zs.ravel(), data[idx].ravel(), cmap='nipy_spectral', levels=1000, vmin=vmin, vmax=vmax)
    idx += 1
    idx = idx % data.shape[0]


# update_plot(0, plot, ax)
# update_plot(100, plot, ax)
# plt.show()
# ani = animation.FuncAnimation(
#     fig, update_plot, 1, fargs=(plot, ax), interval=10
# )
#
# plt.show()
#
# # 1000s long simulation
# # 300 frames
# # each frame 3s
#
# frame_dt = 3
#
# tend = 1000

labels = ["entropy", "density", "water", "vapour", "ice",  "T", "u", "w"]

if size > 1:
    assert size == len(labels)
    labels = labels[rank:rank+1]

print(f'Rank {rank} running {labels}')

for label in labels[:1]:

    fp = os.path.join(data_dump_dir, f'{label}.npy')
    data = np.load(fp)

    vmin = data[0].min()
    vmax = data[0].max()

    if vmin == vmax:
        vmin = data.min()
        vmax = data.max()

    if label == "u":
        vmax = 20.0
        vmin = -vmax

    if label == "w":
        vmax = 20.0
        vmin = -vmax

    idx = 0

    print(fp)
    print(vmin, vmax)
    print()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    levels = np.linspace(vmin, vmax, 1000)
    plot = [ax.tricontourf(xs.ravel(), zs.ravel(), data[0].ravel(), cmap='nipy_spectral', levels=levels, vmin=vmin, vmax=vmax)]
    cbar = plt.colorbar(plot[0], ax=ax)

    moviewriter = MovieWriter(fps=30)
    fp = os.path.join(movie_dir, f"{label}_{experiment_name}.mp4")
    with moviewriter.saving(fig, fp, dpi=100):
        moviewriter.grab_frame()
        for _ in range(data.shape[0]):

            update_plot(0, plot, ax)
            moviewriter.grab_frame()
    

