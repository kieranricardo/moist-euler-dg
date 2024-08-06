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


# def test_thermo(solver):
#
#     dry_theta = 300
#     dexdy = -solver.g / (solver.cpd * dry_theta)
#     ex = np.array([1.,]) - 7000 * solver.g / (solver.cpd * dry_theta)
#     p = 1_00_000.0 * ex ** (solver.cpd / solver.Rd)
#     density = p / (solver.Rd * ex * dry_theta)
#
#     qw = solver.rh_to_qw(0.95, p, density)
#     qd = 1 - qw
#     R = solver.Rd * qd + solver.Rv * qw
#     T = p / (R * density)
#     entropy = qd * solver.entropy_air(T, qd, density)
#     entropy += qw * solver.entropy_vapour(T, qw, density)
#
#     print('T:', T - solver.T0)
#
#     ####
#     qv, ql, qi = solver.solve_fractions_from_entropy(density, qw, entropy)
#
#     print(qv[0], ql[0], qi[0])
#
#     ind = np.array([0.0])
#     qv[:] = qw
#     ql[:] = 0.0
#     qi[:] = 0.0
#     three_phase_thermo.solve_fractions_from_entropy(
#         qv.ravel(), ql.ravel(), qi.ravel(), ind.ravel(), density.ravel(), entropy.ravel(), qw.ravel(), qv.size,
#         solver.Rd, solver.logRd, solver.Rv, solver.logRv, solver.cvd, solver.cvv, solver.cpv, solver.cpd, solver.cl, solver.ci,
#         solver.T0, solver.logT0, solver.p0, solver.logp0, solver.Lf0, solver.Ls0, solver.c0, solver.c1, solver.c2
#     )
#
#     print(qv[0], ql[0], qi[0])
#     print(ind)
#
#     ####
#     qw *= 2
#     qv, ql, qi = solver.solve_fractions_from_entropy(density, qw, entropy)
#
#     print(qv[0], ql[0], qi[0])
#
#     ind = np.array([0.0])
#     qv[:] = qw
#     ql[:] = 0.0
#     qi[:] = 0.0
#     three_phase_thermo.solve_fractions_from_entropy(
#         qv.ravel(), ql.ravel(), qi.ravel(), ind.ravel(), density.ravel(), entropy.ravel(), qw.ravel(), qv.size,
#         solver.Rd, solver.logRd, solver.Rv, solver.logRv, solver.cvd, solver.cvv, solver.cpv, solver.cpd, solver.cl, solver.ci,
#         solver.T0, solver.logT0, solver.p0, solver.logp0, solver.Lf0, solver.Ls0, solver.c0, solver.c1, solver.c2
#     )
#
#     print(qv[0], ql[0], qi[0])
#     print(ind)
#
#
#     assert False



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

    _, _, h, s, qw, *_ = solver.get_vars(solver.state)
    qw *= 2.0
    enthalpy, T, p, ie, mu, qv, ql = solver.get_thermodynamic_quantities(h, s, qw)
    qi = qw - (qv + ql)

    has_liquid = ql > 1e-10
    has_ice = qi > 1e-10

    assert not (has_liquid & (T < (solver.T0 - 1e-10))).any()
    assert not (has_ice & (T > (solver.T0 + 1e-10))).any()

    gv = solver.gibbs_vapour(T, qv, h)
    gl = solver.gibbs_liquid(T)
    gi = solver.gibbs_ice(T)

    assert np.allclose(qw, qv + ql + qi)
    assert np.allclose(gv[has_liquid], gl[has_liquid])
    assert np.allclose(gv[has_ice], gi[has_ice])
    assert np.allclose(gl[has_ice & has_liquid], gi[has_ice & has_liquid])



