import numpy as np
from moist_euler_dg.ice_equilibrium_euler_2D import IceEquilibriumEuler2D
from moist_euler_dg.equilibrium_euler_2D import EquilibriumEuler2D

from matplotlib import pyplot as plt


def error(solver, density, entropy, qw, qv, ql):
    logdensity = np.log(density)
    qd = 1 - qw
    qi = qw - (qv + ql)
    R = qv * solver.Rv + qd * solver.Rd
    cv = qd * solver.cvd + qv * solver.cvv + ql * solver.cl + qi * solver.ci
    logqv = np.log(qv)

    cvlogT = entropy + R * logdensity + qd * solver.Rd * np.log(solver.Rd * qd) + qv * solver.Rv * logqv
    cvlogT += -qv * solver.c0 - ql * solver.c1 - qi * solver.c2
    logT = (1 / cv) * cvlogT
    T = np.exp(logT)

    gibbs_v = -solver.cvv * T * (logT - np.log(solver.T0)) + solver.Rv * T * (logdensity * logqv - np.log(solver.rho0)) + solver.Ls0 * (1 - T / solver.T0)
    gibbs_l = -solver.cl * T * (logT - np.log(solver.T0)) + solver.Lf0 * (1 - T / solver.T0)
    gibbs_i = -solver.ci * T * (logT - np.log(solver.T0))
    val = (gibbs_v - gibbs_l) ** 2 + (gibbs_l - gibbs_i) ** 2 + + (gibbs_v - gibbs_i) ** 2

    return val, T


solver = IceEquilibriumEuler2D((-0.5 , 0.5 ), (0, 1), 3, 2, 2, g=10, eps=0.2)
solver_e = EquilibriumEuler2D((-0.5 , 0.5 ), (0, 1), 3, 2, 2, g=10, eps=0.2)

# get a reasonable entropy a ground level
density = 1.2
qw = 0.02
qd = 1.0 - qw
p = 1_00_000 * 0.95

qv = solver_e.solve_qv_from_p(density, qw, p)
qd = 1 - qw
ql = qw - qv
R = qv * solver_e.Rv + qd * solver_e.Rd

T = p / (density * R)
entropy = qd * solver_e.entropy_air(T, qd, density)
entropy += qv * solver_e.entropy_vapour(T, qv, density)
entropy += ql * solver_e.entropy_liquid(T)
print('qv:', qv)
print('T:', T)
print('Entropy:', entropy)


# solver.c0 = solver_e.c0
entropy = qd * solver.entropy_air(T, qd, density) + qv * solver.entropy_vapour(T, qv, density) + ql * solver.entropy_liquid(T)
print('Entropy:', entropy)

val, T = error(solver, density, entropy, qw, qv, ql)
print('T:', T, '\n')

qv, ql = solver.solve_qv_from_entropy(density, qw, entropy, verbose=True, iters=10, tol=0.0)
print('qv:', qv, '\n')

qv, ql, qi = solver.solve_fractions_from_entropy(density, qw, entropy, verbose=True, iters=30, tol=0.0)

R = qv * solver.Rv + qd * solver.Rd
cv = qd * solver.cvd + qv * solver.cvv + ql * solver.cl + qi * solver.ci
cvlogT = entropy + R * np.log(density) + qd * solver.Rd * np.log(solver.Rd * qd) + qv * solver.Rv * np.log(qv)
cvlogT += -qv * solver.c0 - ql * solver.c1 - qi * solver.c2
logT = (1 / cv) * cvlogT
T = np.exp(logT)
logpv = np.log(qv) + np.log(solver.Rv) + np.log(density) + logT

print('\nqv:', qv)
print('ql:', ql)
print('qi:', qi)
print('T:', T)

# state = {'h': density, 'hs': density * entropy, 'hqw': density * qw}
# ie, die_d, p, qv, ql, qi = solver.get_thermodynamics_quantities(state, mathlib=np)
#
# print('\nqv:', qv)
# print('ql:', ql)
# print('qi:', qi)
# print('T:', die_d['hs'])

