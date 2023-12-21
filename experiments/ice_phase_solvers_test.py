import numpy as np
from moist_euler_dg.ice_equilibrium_euler_2D import IceEquilibriumEuler2D
from moist_euler_dg.equilibrium_euler_2D import EquilibriumEuler2D

from matplotlib import pyplot as plt


def get_T(solver, density, entropy, qw, qv, ql):
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
    return T


def error(solver, density, entropy, qw, qv, ql):
    logdensity = np.log(density)
    logqv = np.log(qv)
    T = get_T(solver, density, entropy, qw, qv, ql)

    gibbs_v = -solver.cvv * T * (logT - np.log(solver.T0)) + solver.Rv * T * (logdensity * logqv - np.log(solver.rho0)) + solver.Ls0 * (1 - T / solver.T0)
    gibbs_l = -solver.cl * T * (logT - np.log(solver.T0)) + solver.Lf0 * (1 - T / solver.T0)
    gibbs_i = -solver.ci * T * (logT - np.log(solver.T0))
    val = (gibbs_v - gibbs_l) ** 2 + (gibbs_l - gibbs_i) ** 2 + + (gibbs_v - gibbs_i) ** 2

    return val, T


solver = IceEquilibriumEuler2D((-0.5 , 0.5 ), (0, 1), 3, 2, 2, g=10, eps=0.2)
solver_e = EquilibriumEuler2D((-0.5 , 0.5 ), (0, 1), 3, 2, 2, g=10, eps=0.2)

# get a reasonable entropy a ground level
# density = 1.2
# qw = 0.02
# p = 1_00_000

# tricky point
T = 202.28980656504663
p = 25193.616121234074
entropy = 2500.0
density = 1.2
qw = 0.02
qd = 1 - qw

qv, ql, qi = solver.solve_fractions_from_entropy(density, qw, entropy)
T = get_T(solver, density, entropy, qw, qv, ql)
print(f'qv={qv}, ql={ql}, qv={qi}')
print(f'T={T}')

T = get_T(solver, density, entropy, qw, qw, 0.0)
print(f'All vapour T={T}')

T = get_T(solver, density, entropy, qw, 1e-12, qw-1e-12)
print(f'All liquid T={T}')

T = get_T(solver, density, entropy, qw, 1e-12, 0.0)
print(f'All ice T={T}')

exit(0)
# entropy = qd * solver.entropy_air(T, qd, density) + qv * solver.entropy_vapour(T, qv, density) + ql * solver.entropy_liquid(T)



dT = 10
T = solver.T0 + dT
print(f'T={T - solver.T0} C, gl={solver.gibbs_liquid(T)}, gi={solver.gibbs_ice(T)}')

T = solver.T0 - dT
print(f'T={T - solver.T0}, gl={solver.gibbs_liquid(T)}, gi={solver.gibbs_ice(T)}')

# gv - gl = 0: --> Rv * T * log(qv * rho) = gl -
T = solver.T0 + 0.1
tmp = solver.gibbs_ice(T) - (-solver.cvv * T * np.log(T / solver.T0) + solver.Rv * T * np.log(1 / solver.rho0) + solver.Ls0 * (1 - T / solver.T0))
tmp /= solver.Rv * T
print(f'T={T-solver.T0} C ice-vapour mix vapour saturation density:', np.exp(tmp))

T = solver.T0 + 1
tmp = solver.gibbs_ice(T) - (-solver.cvv * T * np.log(T / solver.T0) + solver.Rv * T * np.log(1 / solver.rho0) + solver.Ls0 * (1 - T / solver.T0))
tmp /= solver.Rv * T
print(f'T={T-solver.T0} C ice-vapour mix vapour saturation density:', np.exp(tmp))

T = solver.T0 + 10
tmp = solver.gibbs_ice(T) - (-solver.cvv * T * np.log(T / solver.T0) + solver.Rv * T * np.log(1 / solver.rho0) + solver.Ls0 * (1 - T / solver.T0))
tmp /= solver.Rv * T
print(f'T={T-solver.T0} C ice-vapour mix vapour saturation density:', np.exp(tmp))

T = solver.T0 + 100
tmp = solver.gibbs_ice(T) - (-solver.cvv * T * np.log(T / solver.T0) + solver.Rv * T * np.log(1 / solver.rho0) + solver.Ls0 * (1 - T / solver.T0))
tmp /= solver.Rv * T
print(f'T={T-solver.T0} C ice-vapour mix vapour saturation density:', np.exp(tmp), '\n\n')



#####
T = solver.T0 - 0.1
tmp = solver.gibbs_liquid(T) - (-solver.cvv * T * np.log(T / solver.T0) + solver.Rv * T * np.log(1 / solver.rho0) + solver.Ls0 * (1 - T / solver.T0))
tmp /= solver.Rv * T
print(f'T={T-solver.T0} C ice-vapour mix vapour saturation density:', np.exp(tmp))

T = solver.T0 - 1
tmp = solver.gibbs_liquid(T) - (-solver.cvv * T * np.log(T / solver.T0) + solver.Rv * T * np.log(1 / solver.rho0) + solver.Ls0 * (1 - T / solver.T0))
tmp /= solver.Rv * T
print(f'T={T-solver.T0} C ice-vapour mix vapour saturation density:', np.exp(tmp))

T = solver.T0 - 10
tmp = solver.gibbs_liquid(T) - (-solver.cvv * T * np.log(T / solver.T0) + solver.Rv * T * np.log(1 / solver.rho0) + solver.Ls0 * (1 - T / solver.T0))
tmp /= solver.Rv * T
print(f'T={T-solver.T0} C ice-vapour mix vapour saturation density:', np.exp(tmp))

T = solver.T0 - 100
tmp = solver.gibbs_liquid(T) - (-solver.cvv * T * np.log(T / solver.T0) + solver.Rv * T * np.log(1 / solver.rho0) + solver.Ls0 * (1 - T / solver.T0))
tmp /= solver.Rv * T
print(f'T={T-solver.T0} C ice-vapour mix vapour saturation density:', np.exp(tmp))

exit(0)
# qw = solver.rh_to_qw(0.9, p, density)
# qd = 1 - qw
# qv, ql, qi = solver.solve_fractions_from_p(density, 1.2 * qw, p)
# R = qv * solver_e.Rv + qd * solver_e.Rd
# T = p / (density * R)
# print('qv:', qv)
# print('ql:', ql)
# print('qi:', qi)
# print('T:', T)
# exit(0)
# qv = solver_e.solve_qv_from_p(density, qw, p)
#
# qd = 1 - qw
# ql = qw - qv
# R = qv * solver_e.Rv + qd * solver_e.Rd
# T = p / (density * R)
#
# entropy = qd * solver.entropy_air(T, qd, density) + qv * solver.entropy_vapour(T, qv, density) + ql * solver.entropy_liquid(T)
# print('Entropy:', entropy)
#
# val, T = error(solver, density, entropy, qw, qv, ql)
# print('T:', T, '\n')

qv, ql, qi = solver.solve_fractions_from_entropy(density, qw, entropy, verbose=True, iters=30, tol=0.0)

R = qv * solver.Rv + qd * solver.Rd
cv = qd * solver.cvd + qv * solver.cvv + ql * solver.cl + qi * solver.ci
cvlogT = entropy + R * np.log(density) + qd * solver.Rd * np.log(solver.Rd * qd) + qv * solver.Rv * np.log(qv)
cvlogT += -qv * solver.c0 - ql * solver.c1 - qi * solver.c2
logT = (1 / cv) * cvlogT
T = np.exp(logT)
logpv = np.log(qv) + np.log(solver.Rv) + np.log(density) + logT
p = R * density * T

print('\nqv:', qv)
print('ql:', ql)
print('qi:', qi)
print('p:', p)
print('T:', T, '\n')

print(solver.gibbs_vapour(T, qv, density))
print(solver.gibbs_ice(T))
exit(0)

qv, ql, qi = solver.solve_fractions_from_p(density, qw, p)

print('Pressure solve qv:', qv)
print('Pressure solve ql:', ql)
print('Pressure solve qi:', qi, '\n')

specific_ie = cv * T + qv * solver.Ls0 + ql * solver.Lf0
enthalpy = specific_ie + p / density
qv, ql, qi = solver.solve_fractions_from_enthalpy(enthalpy, qw, entropy)
print('Enthalpy solve qv:', qv)
print('Enthalpy solve ql:', ql)
print('Enthalpy solve qi:', qi, '\n')


##########

enthalpy = 284828.18790717854
entropy = 2538.5894564493274
qw = 0.02
qv, ql, qi = solver.solve_fractions_from_enthalpy(enthalpy, qw, entropy)
print('Enthalpy solve qv:', qv)
print('Enthalpy solve ql:', ql)
print('Enthalpy solve qi:', qi, '\n')

R = qv * solver.Rv + qd * solver.Rd
cv = qd * solver.cvd + qv * solver.cvv + ql * solver.cl + qi * solver.ci
cp = qd * solver.cpd + qv * solver.cpv + ql * solver.cl + qi * solver.ci

T = (enthalpy - qv * solver.Ls0 - ql * solver.Lf0) / cp
print('T:', T)
logdensity = (1 / R) * (cv * np.log(T) - entropy - qd * solver.Rd * np.log(solver.Rd * qd)
                        - qv * solver.Rv * np.log(qv) + qv * solver.c0 + ql * solver.c1 + qi * solver.c2)
density = np.exp(logdensity)

qv, ql, qi = solver.solve_fractions_from_entropy(density, qw, entropy, verbose=False)
print('Entropy solve qv:', qv)
print('Entropy solve ql:', ql)
print('Entropy solve qi:', qi, '\n')




# state = {'h': density, 'hs': density * entropy, 'hqw': density * qw}
# ie, die_d, p, qv, ql, qi = solver.get_thermodynamics_quantities(state, mathlib=np)
#
# print('\nqv:', qv)
# print('ql:', ql)
# print('qi:', qi)
# print('T:', die_d['hs'])

