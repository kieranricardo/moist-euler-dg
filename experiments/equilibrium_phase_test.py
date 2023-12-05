import numpy as np
from matplotlib import pyplot as plt
from moist_euler_dg.equilibrium_euler_2D import EquilibriumEuler2D

cpd = 1_004.0
Rd = 287.0
cvd = cpd - Rd
gamma = cpd / cvd

cpv = 1_885.0
Rv = 461.0
cvv = cpv - Rv
# thermodynamic constants
# cpd = 1_005.0
# cvd = 718.0
# Rd = cpd - cvd
# gamma = cpd / cvd
#
# cpv = 1_872.3
# cvv = 1_410.8
# Rv = cpv - cvv
# gammav = cpv / cvv

cl = 4_186.0

p0_ex = 100_000.0

T0 = 273.15
p0 = psat0 = 611.2
Lv0 = 3.1285e6

c0 = cpv + (Lv0 / T0) - cpv * np.log(T0) + Rv * np.log(p0)
c1 = cl - cl * np.log(T0)
########


angle = 0 * (np.pi / 180)
solver = EquilibriumEuler2D(
    (-0.5 , 0.5 ), (0, 1), 3, 2, 2, g=10,
    eps=0.2, a=0.0,
    dtype=np.float64, angle=angle
)

g = 9.81

b = 300
dexdy = -g / 300

density_0 = 1.2

const = cpd * (Rd * 300 / p0_ex) ** (Rd / cvd)
ex0 = const * density_0 ** (Rd / cvd) # surface density is 1.2 kg/m^3
ex = ex0

density = (ex / const) ** (cvd / Rd)

p = (b * Rd * density)**gamma * (1 / p0_ex)**(Rd / cvd)

# go from p, density, qw to entropy
# get temperature?
qw = 0.0002
qd = 1.0 - qw


def saturation_function_p(qv, density, qw, p):
    qd = 1 - qw
    ql = qw - qv

    R = qv * Rv + qd * Rd
    T = p / (density * R)

    pv = qv * Rv * density * T

    return gibbs_vapour(T, pv) - gibbs_liquid(T)


def gibbs_vapour(T, pv):
    return -cpv * T * np.log(T / T0) + Rv * T * np.log(pv / p0) + Lv0 * (1 - T / T0)


def gibbs_liquid(T):
    return -cl * T * np.log(T / T0)


def solve_qv_from_p(density, qw, p):
    qv = 1e-3

    for _ in range(10):
        qd = 1 - qw
        ql = qw - qv
        R = qv * Rv + qd * Rd
        T = p / (density * R)
        pv = qv * Rv * density * T

        dTdqv = -(p / density) * (1 / R**2) * Rv
        dpvdqv = qv * Rv * density * dTdqv + Rv * density * T

        dgvdT = -cpv * np.log(T / T0) - cpv + Rv * np.log(pv / p0) - Lv0 / T0
        dgvdpv = Rv * T / pv

        dgldT = -cl * np.log(T / T0) - cl

        dgvdqv = dgvdT * dTdqv + dgvdpv * dpvdqv
        dgldqv = dgldT * dTdqv

        grad = dgvdqv - dgldqv
        val = gibbs_vapour(T, pv) - gibbs_liquid(T)

        qv = qv - (val / grad)

    return qv

def entropy_vapour(T, qv, density):
    return cvv * np.log(T) - Rv * np.log(qv * Rv) - Rv * np.log(density) + c0

def entropy_liquid(T):
    return cl * np.log(T) + c1

# def entropy_air(T, pd):
#     return cpd * np.log(T) - Rd * np.log(pd)
#

def entropy_air(T, qd, density):
    return cvd * np.log(T) - Rd * np.log(qd * Rd) - Rd * np.log(density)


qvs = np.linspace(1e-10, qw, 1_000_000)
error = abs(saturation_function_p(qvs, density, qw, p))
qv = qvs[np.argmin(error)]
print('Closest 0 (brute force):', min(error))
print('qv:', qv)

print(f"Checks:", (cvd-solver.cvd)/cvd, (Rd-solver.Rd)/Rd, (cvv-solver.cvv)/cvv, (Rv-solver.Rv)/Rv, (cl-solver.cl)/cl)
print(f"Checks:", (c1-solver.c1)/c1, (c0-solver.c0)/c0)
print(f"Checks:", (Lv0-solver.Lv0)/Lv0, (T0-solver.T0)/T0, (p0-solver.p0)/p0)
exit()
qv = solver.solve_qv_from_p(density, qw, p)
print('Closest 0 (Model Newton solve):', saturation_function_p(qv, density, qw, p))
print('qv:', qv)

qv = solve_qv_from_p(density, qw, p)
print('Closest 0 (Newton solve):', saturation_function_p(qv, density, qw, p))
print('qv:', qv)

plt.figure(1)
error = abs(saturation_function_p(qvs, density, qw, p))
plt.semilogy(qvs, error)

ql = qw - qv
R = qv * Rv + qd * Rd
T = p / (density * R)
pv = qv * Rv * density * T
pd = qd * Rd * density * T

print('Compare gibbs:', gibbs_vapour(T, pv), gibbs_liquid(T))

entropy = qd * entropy_air(T, qd, density) + qv * entropy_vapour(T, qv, density) + ql * entropy_liquid(T)
print(entropy)
print('Input temp:', T)
print('Input p:', p)

def saturation_function_entropy(qv, density, qw, entropy):
    ql = qw - qv
    qd = 1 - qw

    R = qd * Rd + qv * Rv
    cv = qd * cvd + qv * cvv + ql * cl
    # cv * log(T) = entropy + R * log(density) + Rd * log(Rd * qd) + Rv * log(Rv * qv)
    logT = (1 / cv) * (entropy + R * np.log(density) + qd * Rd * np.log(Rd * qd) + qv * Rv * np.log(Rv * qv) - qv * c0 - ql * c1)
    T = np.exp(logT)

    T = np.exp((entropy - qv * c0 - ql * c1) / cv) * density**(R / cv) * (Rd * qd)**(Rd * qd / cv) * (Rv * qv)**(Rv * qv / cv)
    p = density * R * T

    return saturation_function_p(qv, density, qw, p)


def entropy_2_p(qv, density, qw, entropy):
    ql = qw - qv
    qd = 1 - qw

    R = qd * Rd + qv * Rv
    cv = qd * cvd + qv * cvv + ql * cl
    # cv * log(T) = entropy + R * log(density) + Rd * log(Rd * qd) + Rv * log(Rv * qv)
    logT = (1 / cv) * (entropy + R * np.log(density) + qd * Rd * np.log(Rd * qd) + qv * Rv * np.log(Rv * qv) - qv * c0 - ql * c1)
    T = np.exp(logT)

    T = np.exp((entropy - qv * c0 - ql * c1) / cv) * density**(R / cv) * (Rd * qd)**(Rd * qd / cv) * (Rv * qv)**(Rv * qv / cv)
    p = density * R * T

    return p


def solve_qv_from_entropy(density, qw, entropy):
    qv = 1e-3

    for _ in range(10):
        qd = 1 - qw
        ql = qw - qv
        R = qv * Rv + qd * Rd
        cv = qd * cvd + qv * cvv + ql * cl

        logT = (1 / cv) * (entropy + R * np.log(density) + qd * Rd * np.log(Rd * qd) + qv * Rv * np.log(Rv * qv) - qv * c0 - ql * c1)
        dlogTdqv = (1 / cv) * (Rv * np.log(density) + Rv * np.log(Rv * qv) + Rv - c0 + c1)
        dlogTdqv += -(1 / cv) *  logT * (cvv - cl)

        T = np.exp(logT)
        p = R * density * T
        pv = qv * Rv * density * T

        dTdqv = dlogTdqv * T

        dpvdqv = qv * Rv * density * dTdqv + Rv * density * T

        dgvdT = -cpv * np.log(T / T0) - cpv + Rv * np.log(pv / p0) - Lv0 / T0
        dgvdpv = Rv * T / pv

        dgldT = -cl * np.log(T / T0) - cl

        dgvdqv = dgvdT * dTdqv + dgvdpv * dpvdqv
        dgldqv = dgldT * dTdqv

        grad = dgvdqv - dgldqv
        val = gibbs_vapour(T, pv) - gibbs_liquid(T)

        qv = qv - (val / grad)

    return qv


print()
print('Check new sat:', saturation_function_entropy(qv, density, qw, entropy))
qv = solve_qv_from_entropy(density, qw, entropy)
print('Closest 0 (entropy Newton solve):', saturation_function_p(qv, density, qw, p))
print('qv:', qv)

qv = solver.solve_qv_from_entropy(density, qw, entropy)
print('Closest 0 (model entropy Newton solve):', saturation_function_p(qv, density, qw, p))
print('qv:', qv)
print()
qd = 1 - qw
ql = qw - qv
h = density
cv = qd * cvd + qv * cvv + ql * cl
R = qd * Rd + qv * Rv
logT = (1 / cv) * (entropy + R * np.log(density) + qd * Rd * np.log(Rd * qd) + qv * Rv * np.log(Rv * qv) - qv * c0 - ql * c1)

logT = (1 / cv) * (entropy + R * np.log(density) + qd * Rd * np.log(Rd * qd) + qv * Rv * np.log(Rv * qv) - qv * c0 - ql * c1)
T = np.exp(logT)

pv = qv * density * Rv * T

dlogTdqv = (1 / cv) * (Rv * np.log(h) + Rv * np.log(Rv * qv) + Rv - c0)
dlogTdqv += -(1 / cv) * logT * cvv
dTdqv = dlogTdqv * T

dlogTdql = (1 / cv) * (-c1)
dlogTdql += -(1 / cv) * logT * cl
dTdql = dlogTdql * T

dlogTdqd = (1 / cv) * (Rd * np.log(h) + Rd * np.log(Rd * qd) + Rd)
dlogTdqd += -(1 / cv) * logT * cvd
dTdqd = dlogTdqd * T

chemical_potential_d = cv * dTdqd + cvd * T
chemical_potential_v = cv * dTdqv + cvv * T + Lv0
chemical_potential_l = cv * dTdql + cl * T

print('New T:', T)
print('Compare chemical potentials:', chemical_potential_v, chemical_potential_l)
print('Compare gibbs:', gibbs_vapour(T, pv), gibbs_liquid(T))

plt.figure(2)
error = abs(saturation_function_entropy(qvs, density, qw, entropy))
plt.semilogy(qvs, error)

plt.figure(3)
plt.semilogy(qvs, entropy_2_p(qvs, density, qw, entropy))
# plt.show()
