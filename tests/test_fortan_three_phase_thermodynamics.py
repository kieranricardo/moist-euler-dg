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


def test_moisture_fraction_solver(solver):

    _, _, h, s, qw, *_= solver.get_vars(solver.state)
    enthalpy, T, p, ie, mu, qv, ql = solver.get_thermodynamic_quantities(h, s, qw)
    np.allclose((qw - qv), 0.0)

    gv = solver.gibbs_vapour(T, qv, h)
    qd = 1 - qw
    gd = solver.gibbs_air(T, qd, h)
    assert np.allclose(gv - gd, mu)

    _, _, h, s, qw, *_ = solver.get_vars(solver.state)
    qw *= 2.0
    enthalpy, T, p, ie, mu, qv, ql = solver.get_thermodynamic_quantities(h, s, qw)
    qi = qw - (qv + ql)

    print('qv min:', qv.min())
    has_liquid = ql > 1e-10
    has_ice = qi > 1e-10
    has_vapour = qv > 1e-10

    assert not (has_liquid & (T < (solver.T0 - 1e-10))).any()
    assert not (has_ice & (T > (solver.T0 + 1e-10))).any()

    gv = solver.gibbs_vapour(T, qv, h)
    gl = solver.gibbs_liquid(T)
    gi = solver.gibbs_ice(T)

    assert np.allclose(qw, qv + ql + qi)
    # assert np.allclose(gv[has_liquid & has_vapour], gl[has_liquid & has_vapour])
    # assert np.allclose(gv[has_ice & has_vapour], gi[has_ice & has_vapour])
    # assert np.allclose(gl[has_ice & has_liquid], gi[has_ice & has_liquid])

    qd  = 1 - qw
    gd = solver.gibbs_air(T, qd, h)
    # assert np.allclose((gv - gd)[has_vapour], mu[has_vapour])
    # assert np.allclose((gv - gd)[has_vapour], mu[has_vapour])

    density = 0.6275959315151061;
    entropy = 2531.0038776852075;
    qw = 0.01330944126634543;
    density = np.array([density]);
    entropy = np.array([entropy]);
    qw = np.array([qw]);

    enthalpy, T, p, ie, mu, qv, ql = solver.get_thermodynamic_quantities(density, entropy, qw)
