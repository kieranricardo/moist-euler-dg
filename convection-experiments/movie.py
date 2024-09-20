from matplotlib import pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FFMpegWriter as MovieWriter
import matplotlib.animation as animation

import numpy as np
import time
import os

experiment_name = "forced-convection-nx-32-nz-32-p3"
plot_dir = os.path.join('plots', experiment_name)
data_dump_dir = os.path.join(plot_dir, 'data-dump')

label = 'xcoord'
fp = os.path.join(data_dump_dir, f'{label}_forced-convection_nx_32_nz_32_p3_upwind_True_time_0H0m0s.npy')
xs = np.load(fp)

label = 'zcoord'
fp = os.path.join(data_dump_dir, f'{label}_forced-convection_nx_32_nz_32_p3_upwind_True_time_0H0m0s.npy')
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

labels = ["entropy", "density", "water", "vapour", "liquid", "ice", "u", "w", "T"]
labels = ["u", "w"]
for label in labels:

    fp = os.path.join(data_dump_dir, f'{label}_forced-convection_nx_32_nz_32_p3_upwind_True_time_0H0m0s.npy')
    data = np.load(fp)

    vmin = data[0].min()
    vmax = data[0].max()

    if vmax == vmin:
        vmin = data.min()
        vmax = data.max()

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
    fp = os.path.join(plot_dir, f"{label}_{experiment_name}.mp4")
    with moviewriter.saving(fig, fp, dpi=100):
        moviewriter.grab_frame()
        for _ in range(data.shape[0]):

            update_plot(0, plot, ax)
            moviewriter.grab_frame()


