import pytest
import numpy as np
from moist_euler_dg.fortran_three_phase_euler_2D import FortranThreePhaseEuler2D
from moist_euler_dg.three_phase_euler_2D import ThreePhaseEuler2D
from _moist_euler_dg import three_phase_thermo


@pytest.fixture()
def solver():
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


def test_benchmark_solve(benchmark, solver):

    state = solver.state
    dstatedt = np.zeros_like(state)
    benchmark(solver.solve, state, dstatedt)


def test_benchmark_solve_fractions_from_entropy(benchmark, solver):

    benchmark(solver.solve_fractions_from_entropy, solver.h, solver.q, solver.s)


def test_benchmark_get_thermodynamic_quantities(benchmark, solver):
    benchmark(solver.get_thermodynamic_quantities, solver.h, solver.s, solver.q)

