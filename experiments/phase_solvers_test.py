import numpy as np
from moist_euler_dg.equilibrium_euler_2D import EquilibriumEuler2D

# TODO: saturation vapour pressure comparison?
angle = 0 * (np.pi / 180)
solver = EquilibriumEuler2D(
    (-0.5 , 0.5 ), (0, 1), 3, 2, 2, g=10,
    eps=0.2, a=0.0,
    dtype=np.float64, angle=angle
)

density = 1.2
p = 1_00_000
qw = 0.2
qd = 1.0 - qw

qv = solver.solve_qv_from_p(density, qw, p, verbose=True)
print('qv:', qv)

ql = qw - qv
R = qv * solver.Rv + qd * solver.Rd
cv = qv * solver.cvv + qd * solver.cvd + ql * solver.cl
T = p / (density * R)
pv = qv * solver.Rv * density * T
pd = qd * solver.Rd * density * T

entropy = qd * solver.entropy_air(T, qd, density) + qv * solver.entropy_vapour(T, qv, density) + ql * solver.entropy_liquid(T)
qv1 = solver.solve_qv_from_entropy(density, qw, entropy, verbose=True)
print('Compare:', abs(qv - qv1) / qv)

ie = cv * T * density + qv * solver.Lv0 * density
enthalpy = (ie + p) / density
qv2 = solver.solve_qv_from_enthalpy(enthalpy, qw, entropy, verbose=True)
print('Compare:', abs(qv - qv2) / qv)
print('qv2:', qv2)
