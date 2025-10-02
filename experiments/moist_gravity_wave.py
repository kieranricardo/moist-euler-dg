from matplotlib import pyplot as plt
from moist_euler_dg.fortran_two_phase_euler_2D import FortranTwoPhaseEuler2D as TwoPhaseEuler
from moist_euler_dg import utils
import numpy as np
import time
import os
import argparse
from mpi4py import MPI
import matplotlib.ticker as ticker
import scipy
import cmocean


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

parser = argparse.ArgumentParser()
parser.add_argument('--nz', type=int, help='Number of vertical cells')
parser.add_argument('--nx', type=int, help='Number of horizontal cells')
parser.add_argument('--o', type=int, help='Polynomial order')
parser.add_argument('--nproc', type=int, help='Number of procs', default=1)
parser.add_argument('--plot', action='store_true')
args = parser.parse_args()

xlim = 300_000
zlim = 10_000
# maps to define geometry these can be arbitrary - maps [0, 1]^2 to domain
zmap = lambda x, z: z * zlim
xmap = lambda x, z: xlim * x #* (x - 0.5)

nz = args.nz
nx = args.nx
nproc = args.nproc
run_model = (not args.plot) # whether to run model - set false to just plot previous run
poly_order = args.o

cfl = 0.5
g = 9.81
a = 0.5
upwind = True

exp_name_short = 'moist-gravity-wave'
experiment_name = f'{exp_name_short}-nx-{nx}-nz-{nz}-p{poly_order}'
data_dir = os.path.join('data', experiment_name)
plot_dir = os.path.join('plots', experiment_name)

if rank == 0:
    print(f"---------- Moist gravity wave with nx={nx}, nz={nz}, cfl={cfl}")
    if not os.path.exists(plot_dir): os.makedirs(plot_dir)
    if not os.path.exists(data_dir): os.makedirs(data_dir)

comm.barrier()


def dry_density_profile(solver, zs):

    R, cv, cp, g = solver.Rd, solver.cvd, solver.cpd, solver.g
    p0 = 100_000
    T0 = 300.0

    N = 0.01
    N2 = N ** 2

    pot_temp_profile = T0 * np.exp(N2 * zs / g)
    ex = 1 + (g ** 2 / (cp * T0 * N2)) * (np.exp(-N2 * zs / g) - 1)
    density_profile = (p0 / (R * pot_temp_profile)) * ex ** (cv / R)

    return density_profile


def _dg_get_ddz_matrix(solver):

    zs = solver.zs[0, :, 0]
    dz = np.diff(zs[:, 0])[0]
    weight = solver.weights_z[-1]

    assert np.allclose(np.diff(zs[:, 0]), dz)

    data = []
    row_idx = []
    col_idx = []

    n = solver.order + 1
    for element_idx in range(solver.nz):

        shift = element_idx * n
        for i in range(n):
            for j in range(n):

                data.append(solver.D[i, j] * 2 / dz)
                row_idx.append(i + shift)
                col_idx.append(j + shift)

                if (i == j) and (i == 0) and (element_idx > 0):
                    data[-1] += 0.5 * (dz / 2) / solver.weights_z[-1]

                if (i == j) and (i == (n - 1)) and (element_idx < (solver.nz - 1)):
                    data[-1] += -0.5 * (dz / 2) / weight

    for element_idx in range(1, solver.nz):
        im = element_idx * n - 1
        ip = element_idx * n

        scale = (dz / 2) / weight

        # centred
        data.extend([0.5 * scale, -0.5 * scale])
        row_idx.extend([im, ip])
        col_idx.extend([ip, im])
    
    sat_scale = (dz / 2) / weight
    data[0] += sat_scale
    data = np.array(data)
    row_idx = np.array(row_idx)
    col_idx = np.array(col_idx)

    ddz_mat = scipy.sparse.coo_matrix((data, (row_idx, col_idx))).tocsc()
    inv_ddz = scipy.sparse.linalg.splu(ddz_mat)

    return zs.ravel(), dz, ddz_mat, inv_ddz, sat_scale


def _sbp_get_ddz_matrix(solver):
    
    zs = np.linspace(0, solver.zs.max(), 1000)
    
    sz = zs.size
    dz = np.diff(zs).mean()
    weight = 1.0
    
    data = []
    row_idx = []
    col_idx = []

    data.extend([-1 / dz, 1 / dz])
    row_idx.extend([0, 0])
    col_idx.extend([0, 1])

    for i in range(1, sz - 1):
        data.extend([-1 / (2 * dz), 1 / (2 * dz)])
        row_idx.extend([i, i])
        col_idx.extend([i-1, i+1])


    data.extend([-1 / dz, 1 / dz])
    row_idx.extend([sz - 1, sz - 1])
    col_idx.extend([sz - 2, sz - 1])
    
    sat_scale = 1.0 * dz / weight
    data[0] += sat_scale # SAT term

    data = np.array(data)
    row_idx = np.array(row_idx)
    col_idx = np.array(col_idx)

    ddz_mat = scipy.sparse.coo_matrix((data, (row_idx, col_idx))).tocsc()
    inv_ddz = scipy.sparse.linalg.splu(ddz_mat)
    
    return zs.ravel(), dz, ddz_mat, inv_ddz, sat_scale


def get_profiles(solver, p_sfc, mpt_sfc, N, qw_sfc):
    # solve for rho s.t: (d/dz) p(rho, s, qw) + g * rho = 0

    # get derivative matrix (included DG jumps and SAT boundary term
    zs, dz, ddz_mat, inv_ddz, sat_scale = _dg_get_ddz_matrix(solver)

    # setup moisture and moist potential temperature/entropy profiles
    qw = qw_sfc + np.zeros_like(zs)
    mpt_profile = mpt_sfc * np.exp(N ** 2 * zs / solver.g)
    s = solver.moist_potential_temperature_to_entropy(mpt_profile, qw)

    # initial guess for density profile
    density = dry_density_profile(solver, zs)

    # newton iteration
    ftol = 1e-2

    prev_error = np.inf
    for ii in range(100):

        enthalpy, T, p, ie, _, qv, ql = solver.get_thermodynamic_quantities(density, s, qw)

        qd = (1 - qw)
        R = qd * solver.Rd + qv * solver.Rv
        cv = qd * solver.cpd + qv * solver.cvv + ql * solver.cl

        dpdrho = (p / density) * ((R / cv) + 1)

        y = ddz_mat @ p + g * density
        y[0] -= p_sfc * sat_scale
        
        error = np.sqrt(np.mean(y**2))
        #     if ii == 0:
        #         print(error)

        #     if ii % 10 == 0:
        #         print('Relative error reduction:', abs(prev_error - error) / prev_error)

        if ii > 0:
            rel_error_reduction = abs(prev_error - error) / prev_error

        if abs(prev_error - error) < ftol * prev_error:
            if rank == 0:
                print(f'ftol condition satisfied at iteration {ii}')
            break

        prev_error = error

        # inner loop
        rhs = -y
        drho = np.zeros_like(rhs)

        rtol = 1e-8

        for jj in range(50):

            drho = inv_ddz.solve(rhs - g * drho) / dpdrho

            lhs = ddz_mat @ (dpdrho * drho) + g * drho
            resid = rhs - lhs

            if np.linalg.norm(resid) <= rtol * np.linalg.norm(rhs):
                break

        if np.linalg.norm(resid) > rtol * np.linalg.norm(rhs):
            print('Inner loop failed')

        density = density + drho

    if rank == 0:
        print(f'Average hydrostatic balance error = {error:.4g} N')
        print(f'Pressure surface error = {(p[0] - p_sfc):.4g} Pa')
    
    return density, zs


def hr_profiles(solver, p_sfc, mpt_sfc, N, qw_sfc):
    nz_hr = 64
    nx_hr = 1
    solver_hr = TwoPhaseEuler(
        xmap, zmap, 6, nx_hr, g=g, cfl=0.4, a=a, nz=nz_hr, upwind=upwind, nprocx=1
    )

    density_hr, _ = get_profiles(solver_hr, p_sfc, mpt_sfc, N, qw_sfc)

    density_hr = density_hr.reshape((nz_hr, -1))
    dz_hr = zlim / nz_hr

    ip = (slice(1, None), 0)
    im = (slice(0, -1), -1)

    density_avg = 0.5 * (density_hr[ip] + density_hr[im])
    density_hr[ip] = density_avg
    density_hr[im] = density_avg

    x_in, _ = utils.gll(solver_hr.order, iterative=True)
    lagrange_polys = []
    for i in range(len(x_in)): 
        x_data = np.zeros_like(x_in)
        x_data[i] = 1.0
        lagrange_polys.append(scipy.interpolate.lagrange(x_in, x_data))
        
    def _eval_solution_at_point(z, coeffs, dz):

        cell_idx = int(z / dz)
        zeta = (2 * (z - cell_idx * dz) / dz) - 1
        
        if (cell_idx == coeffs.shape[0]) and (zeta == -1):
            zeta = 1
            cell_idx = coeffs.shape[0] - 1
        
        lagrange_poly_data = np.array([poly(zeta) for poly in lagrange_polys])
        out = (lagrange_poly_data * coeffs[cell_idx]).sum()
        
        return out

    zs_eval = np.linspace(0, 10_000, 1000)
    density_eval = []
    for z in zs_eval:
        density_eval.append(_eval_solution_at_point(z, density_hr, dz_hr))
        
    poly_fit = np.polynomial.chebyshev.Chebyshev.fit(zs_eval, density_eval, deg=5)
    density_lr = poly_fit(solver.zs[0, :, 0].ravel())
    # density_lr = []
    # for z in solver.zs[0, :, 0].ravel():
    #     density_lr.append(_eval_solution_at_point(z, density_hr, dz_hr))
        
    return np.array(density_lr), solver.zs[0, :, 0].ravel()

def initial_condition(solver, pert):
    mpt_sfc = 300.0 # sft moist potential temp
    N = 0.01 # Brunt-Vaisala frequency
    qw_sfc = 0.02 # constant water mass fraction
    p_sfc = 100_000
    
    mpt_profile = mpt_sfc * np.exp(N ** 2 * solver.zs / solver.g)
    qw = qw_sfc + np.zeros_like(mpt_profile)
    
    density, zs = hr_profiles(solver, p_sfc, mpt_sfc, N, qw_sfc)
    # density, zs = get_profiles(solver, p_sfc, mpt_sfc, N, qw_sfc)
    
    # output density on model grid
    if (zs.size == (solver.nz * (solver.order + 1))) and np.allclose(zs, solver.zs[0, :, 0].ravel()):
        density = (np.ones_like(solver.zs[:, :1, :, :1]) * density.reshape((1, solver.nz, 1, solver.order + 1)))
    else:
        density = np.interp(solver.zs, zs, density)

    mpt_profile = mpt_sfc * np.exp(N ** 2 * solver.zs / solver.g)
    qw = qw_sfc + np.zeros_like(mpt_profile)

    # add perturbation to moist potential temperature
    hc = 10_000.0
    xc = 150_000.0
    ac = 5000
    mpt_pert = pert * np.sin(np.pi * solver.zs / hc) / (1 + ((solver.xs - xc) / ac) ** 2)

    mpt = mpt_profile + mpt_pert
    s = solver.moist_potential_temperature_to_entropy(mpt, qw)

    u = 20 + 0 * s
    v = 0 * s

    return u, v, density, s, qw, mpt_profile


tends = np.array([0.0, 1200, 2400, 3600])

conservation_data_fp = os.path.join(data_dir, 'conservation_data.npy')
time_list = []
energy_list = []
entropy_var_list = []
water_var_list = []

if run_model:
    solver = TwoPhaseEuler(xmap, zmap, poly_order, nx, g=g, cfl=cfl, a=a, nz=nz, upwind=upwind, nprocx=nproc)
    u, v, density, s, qw, mpt_profile = initial_condition(solver, pert=0.01)
    solver.set_initial_condition(u, v, density, s, qw)

    for i, tend in enumerate(tends):
        t0 = time.time()
        while solver.time < tend:
            time_list.append(solver.time)
            energy_list.append(solver.energy())
            entropy_var_list.append(solver.integrate(solver.h * solver.s ** 2))
            water_var_list.append(solver.integrate(solver.h * solver.q ** 2))

            dt = min(solver.get_dt(), tend - solver.time)
            solver.time_step(dt=dt)
        t1 = time.time()

        if rank == 0:
            print("Simulation time (s):", solver.time)
            print("Wall time:", time.time() - t0, '\n')

        solver.save(solver.get_filepath(data_dir, exp_name_short))

    if rank == 0:
        conservation_data = np.zeros((4, len(time_list)))
        conservation_data[0, :] = np.array(time_list)
        conservation_data[1, :] = np.array(energy_list)
        conservation_data[2, :] = np.array(entropy_var_list)
        conservation_data[3, :] = np.array(water_var_list)
        np.save(conservation_data_fp, conservation_data)

        print('Energy error:', (energy_list[-1] - energy_list[0]) / energy_list[0])

elif rank == 0:
    plt.rcParams['font.size'] = '12'

    solver_plot = TwoPhaseEuler(xmap, zmap, poly_order, nx, g=g, cfl=0.5, a=a, nz=nz, upwind=upwind, nprocx=1)
    u0, v0, density0, s0, qw0, mpt_profile = initial_condition(solver_plot, pert=0.0)

    mpt0 = solver_plot.moist_potential_temperature(s0, qw0)
    qv0 = solver_plot.solve_qv_from_entropy(density0, qw0, s0)
    ql0 = qw0 - qv0

    def fmt(x, pos):
        a, b = '{:.2e}'.format(x).split('e')
        b = int(b)
        return r'${} \times 10^{{{}}}$'.format(a, b)

    plot_func_mpt = lambda s: s.project_H1(s.moist_potential_temperature(s.s, s.q) - mpt0)
    plot_func_entropy = lambda s: s.project_H1(s.s - s0)
    plot_func_density = lambda s: s.project_H1(s.h - density0)
    plot_func_water = lambda s: s.project_H1(s.q - qw0)
    plot_func_vapour = lambda s: s.project_H1(s.solve_qv_from_entropy(s.h, s.q, s.s) - qv0)
    plot_func_liquid = lambda s: s.project_H1(s.q - s.solve_qv_from_entropy(s.h, s.q, s.s) - ql0)

    fig_list = [plt.subplots(2, 2, sharex=True, sharey=True, figsize=(7.4, 4.8)) for _ in range(6)]
    pfunc_list = [
        plot_func_mpt, plot_func_entropy, plot_func_density,
        plot_func_water, plot_func_vapour, plot_func_liquid,
    ]

    labels = ["moist_potential_temperature", "entropy", "density", "water", "vapour", "liquid"]

    energy = []
    for i, tend in enumerate(tends):
        filepaths = [solver_plot.get_filepath(data_dir, exp_name_short, proc=i, nprocx=nproc, time=tend) for i in range(nproc)]
        solver_plot.load(filepaths)
        energy.append(solver_plot.integrate(solver_plot.energy()))

        for (fig, axs), plot_fun, label in zip(fig_list, pfunc_list, labels):

            ax = axs[i // 2][i % 2]
            ax.tick_params(labelsize=8)

            if label == 'moist_potential_temperature':
                im = solver_plot.plot_solution(ax, dim=2, plot_func=plot_fun, km=True, levels=1000, cmap=cmocean.cm.thermal)
            else:
                levels = 1000
                im = solver_plot.plot_solution(ax, dim=2, plot_func=plot_fun, km=True, levels=1000)

            # if label == 'entropy':
            #     cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), label='Entropy (K)')
            # elif label == 'density':
            #     cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), label='Density ($\text{kg m}^{-3}$)')
            # else:
            #     cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt), label=f'{label.capitalize() mass fraction'})
            cbar = plt.colorbar(im, ax=ax, format=ticker.FuncFormatter(fmt))
            cbar.ax.tick_params(labelsize=8)

            if (i // 2) == 1:
                ax.set_xlabel('x (km)', fontsize='xx-small')
            if (i % 2) == 0:
                ax.set_ylabel('z (km)', fontsize='xx-small')
            # fig.tight_layout(w_pad=1.0, h_pad=1.0)
            fig.tight_layout()

    for (fig, ax), label in zip(fig_list, labels):

        plot_name = f'{label}_{exp_name_short}'
        fp = solver_plot.get_filepath(plot_dir, plot_name, ext='png')
        print(fp)
        fig.savefig(fp, bbox_inches="tight")

    # full moist pt plot
    plt.figure(figsize=(12, 6))
    levels = np.linspace(-3e-3, 3e-3, 13)
    solver_plot.plot_contours(plt.gca(), plot_func=plot_func_mpt, km=True, levels=levels)
    im = solver_plot.plot_solution(plt.gca(), dim=2, plot_func=plot_func_mpt, km=True, levels=levels, cmap=cmocean.cm.thermal)
    cbar = plt.colorbar(im, ax=plt.gca(), format=ticker.FuncFormatter(fmt), label=r'$\theta_e$ (K)')
    cbar.ax.tick_params(labelsize=8)
    plt.gca().set_xlabel('x (km)')
    plt.gca().set_ylabel('z (km)')
    plt.tight_layout()

    plot_name = f'moist_potential_temperature_{exp_name_short}_final'
    fp = solver_plot.get_filepath(plot_dir, plot_name, ext='png')
    print(fp)
    plt.savefig(fp)
