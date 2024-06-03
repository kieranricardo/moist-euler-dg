from matplotlib import pyplot as plt
from moist_euler_dg.three_phase_euler_2D import ThreePhaseEuler2D
import numpy as np
import time
import os
import argparse
from mpi4py import MPI
import matplotlib.ticker as ticker


run_model = True # whether to run model - set false to just plot previous run
xlim = 10_000
zlim = 10_000
zmap = lambda x, z: z * zlim
xmap = lambda x, z: xlim * (x - 0.5)

nz = 16
nx = nz
eps = 0.8
g = 9.81
poly_order = 3

exp_name_short = 'ice-bubble'
experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{poly_order}'
data_dir = os.path.join('data', experiment_name)
plot_dir = os.path.join('plots', experiment_name)

solver_plot = ThreePhaseEuler2D(xmap, zmap, poly_order, nx, g=g, cfl=1.0, a=0.0, nz=nz, upwind=True, nprocx=1)

size = 8

tend = 0
filepaths = [solver_plot.get_filepath(data_dir, exp_name_short, proc=i, nprocx=size, time=tend) for i in range(size)]
solver_plot.load(filepaths)
E_start = solver_plot.energy()

tend = 300
filepaths = [solver_plot.get_filepath(data_dir, exp_name_short, proc=i, nprocx=size, time=tend) for i in range(size)]
solver_plot.load(filepaths)
E0 = solver_plot.energy()

state = np.copy(solver_plot.state)
u, w, h, s, q = solver_plot.get_vars(state)
u, w = solver_plot.cov_to_phy(u, w)

solver_plot.a = 0.0

dstatedt = solver_plot.solve(state)
dudt, dwdt, dhdt, dsdt, dqdt = solver_plot.get_vars(dstatedt)
dudt, dwdt = solver_plot.cov_to_phy(dudt, dwdt)

enthalpy, T, p, ie, mu, qv, ql = solver_plot.get_thermodynamic_quantities(h, s, q)

dEdu = h * u
dEdw = h * w
dEdh = 0.5 * (u**2 + w**2) + enthalpy + solver_plot.g * solver_plot.zs
dEds = h * T
dEdq = h * mu

dEdt = dEdu * dudt + dEdw * dwdt + dEdh * dhdt + dEds * dsdt + dEdq * dqdt
dEdt = solver_plot.integrate(dEdt)

#-54702223.40350479
dt = solver_plot.get_dt()
print('Estimated energy change:', dEdt * dt / E_start)

solver_plot.time_step(dt=dt)
E1 = solver_plot.energy()
print('Actual energy change:', (E1 - E0) / E_start)
