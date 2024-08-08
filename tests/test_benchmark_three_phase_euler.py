import pytest
import numpy as np
from moist_euler_dg.fortran_three_phase_euler_2D import FortranThreePhaseEuler2D
from moist_euler_dg.three_phase_euler_2D import ThreePhaseEuler2D
from _moist_euler_dg import three_phase_thermo


@pytest.fixture()
def pysolver():
    xlim = 50_000
    zlim = 10_000
    # maps to define geometry these can be arbitrary - maps [0, 1]^2 to domain
    zmap = lambda x, z: z * zlim
    xmap = lambda x, z: xlim * (x - 0.5)

    # number of cells in the vertical and horizontal direction
    nz = 16
    nx = 32

    g = 9.81  # gravitational acceleration
    poly_order = 3  # spatial order of accuracy
    a = 0.5  # kinetic energy dissipation parameter
    upwind = True

    solver_ = ThreePhaseEuler2D(
        xmap, zmap, poly_order, nx, g=g, cfl=1.5, a=a, nz=nz, upwind=upwind, nprocx=1
    )

    solver_.set_initial_condition(*initial_condition(solver_))

    return solver_


@pytest.fixture()
def fsolver():
    xlim = 50_000
    zlim = 10_000
    # maps to define geometry these can be arbitrary - maps [0, 1]^2 to domain
    zmap = lambda x, z: z * zlim
    xmap = lambda x, z: xlim * (x - 0.5)

    # number of cells in the vertical and horizontal direction
    nz = 16
    nx = 32

    g = 9.81  # gravitational acceleration
    poly_order = 3  # spatial order of accuracy
    a = 0.5  # kinetic energy dissipation parameter
    upwind = True

    solver_ = FortranThreePhaseEuler2D(
        xmap, zmap, poly_order, nx, g=g, cfl=1.5, a=a, nz=nz, upwind=upwind, nprocx=1
    )

    solver_.set_initial_condition(*initial_condition(solver_))

    return solver_


def initial_condition(solver_):
    # initial velocity is zero
    u = np.zeros_like(solver_.zs)
    v = np.zeros_like(solver_.zs)

    # create a hydrostatically balanced pressure and density profile
    dry_theta = 300
    dexdy = -solver_.g / (solver_.cpd * dry_theta)
    ex = 1 + dexdy * solver_.zs
    p = 1_00_000.0 * ex ** (solver_.cpd / solver_.Rd)
    density = p / (solver_.Rd * ex * dry_theta)

    qw = solver_.rh_to_qw(0.95, p, density)
    qd = 1 - qw

    R = solver_.Rd * qd + solver_.Rv * qw
    T = p / (R * density)
    s = qd * solver_.entropy_air(T, qd, density)
    s += qw * solver_.entropy_vapour(T, qw, density)

    return u, v, density, s, qw


def test_solves_equivalent(fsolver, pysolver):

    state = np.zeros_like(fsolver.state)

    u1, w1, *arrs1 = fsolver.get_vars(state)
    _, _, *arrs2 = fsolver.get_vars(fsolver.state)

    u_phys, w_phys = 2 * (np.random.random(u1.shape) - 0.5), 2 * (np.random.random(w1.shape) - 0.5)
    u_cov, w_cov = fsolver.phys_to_cov(u_phys, w_phys)
    u1[:] = u_cov
    w_cov[:] = w_cov

    for arr1, arr2 in zip(arrs1, arrs2):
        pert = 2 * (np.random.random(arr1.shape) - 0.5)
        arr1[:] = arr2 * (1 + 0.1 * pert)

    out1 = np.zeros_like(state)
    out2 = np.zeros_like(state)

    fsolver.a = 0.0
    pysolver.a = fsolver.a

    fsolver.solve(state, out1)
    pysolver.solve(state, out2)

    names = ['u', 'w', 'h', 's', 'q']

    for name, arr1, arr2 in zip(names, fsolver.get_vars(out1), fsolver.get_vars(out2)):
        print(name, np.allclose(arr1, arr2))

    # out1 = fsolver.solve(state)
    # out2 = pysolver.solve(state)

    assert np.allclose(out1, out2)

def test_benchmark_py_solve(benchmark, pysolver):

    state = pysolver.state
    dstatedt = np.zeros_like(state)
    benchmark(pysolver.solve, state, dstatedt)


def test_benchmark_f_solve(benchmark, fsolver):

    state = fsolver.state
    dstatedt = np.zeros_like(state)
    benchmark(fsolver.solve, state, dstatedt)


def test_benchmark_py_get_thermodynamic_quantities(benchmark, pysolver):
    benchmark(pysolver.get_thermodynamic_quantities, pysolver.h, pysolver.s, pysolver.q)


def test_benchmark_f_get_thermodynamic_quantities(benchmark, fsolver):
    benchmark(fsolver.get_thermodynamic_quantities, fsolver.h, fsolver.s, fsolver.q)
