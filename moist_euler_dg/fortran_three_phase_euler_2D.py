import numpy as np
from moist_euler_dg.three_phase_euler_2D import ThreePhaseEuler2D
from _moist_euler_dg import three_phase_thermo

class FortranThreePhaseEuler2D(ThreePhaseEuler2D):


    def solve_fractions_from_entropy(self, density, qw, entropy, qv=None, ql=None, qi=None, iters=10, tol=1e-10):

        if qv is None:
            qv = np.copy(qw)
            ql = np.zeros_like(qw)
            qi = np.zeros_like(qw)

        mask = qv == 0
        qv[mask] = qw[mask]
        ql[mask] = 0
        qi[mask] = 0

        ind = np.zeros_like(qv)

        three_phase_thermo.solve_fractions_from_entropy(
            qv.ravel(), ql.ravel(), qi.ravel(), ind.ravel(), density.ravel(), entropy.ravel(), qw.ravel(), qv.size,
            self.Rd, self.logRd, self.Rv, self.logRv, self.cvd, self.cvv, self.cpv, self.cpd, self.cl, self.ci,
            self.T0, self.logT0, self.p0, self.logp0, self.Lf0, self.Ls0, self.c0, self.c1, self.c2
        )

        is_solved = (ind > 0)
        qi[:] = qw - (qv + ql)
        if (~is_solved).any():
            print('Thermo solver failed')

        return qv, ql, qi

    # def get_thermodynamic_quantities(self, density, entropy, qw, update_cache=False, use_cache=False):
    #
    #     qd = 1 - qw
    #
    #     if use_cache:
    #         qv, ql, qi = self.qv, self.ql, self.qi
    #     else:
    #         qv, ql, qi = np.zeros_like(density), np.zeros_like(density), np.zeros_like(density)
    #         qv[:] = qw
    #
    #     T = np.zeros_like(density)
    #     mu = np.zeros_like(density)
    #     ind = np.zeros_like(density)
    #
    #
    #     three_phase_thermo.solve_fractions_from_entropy(
    #         qv.ravel(), ql.ravel(), qi.ravel(), T.ravel(), mu.ravel(), ind.ravel(), density.ravel(), entropy.ravel(), qw.ravel(), qv.size,
    #         self.Rd, self.logRd, self.Rv, self.logRv, self.cvd, self.cvv, self.cpv, self.cpd, self.cl, self.ci,
    #         self.T0, self.logT0, self.p0, self.logp0, self.Lf0, self.Ls0, self.c0, self.c1, self.c2
    #     )
    #
    #     R = qv * self.Rv + qd * self.Rd
    #     cv = qd * self.cvd + qv * self.cvv + ql * self.cl + qi * self.ci
    #
    #
    #     p = density * R * T
    #
    #     # print('Model T min-max:', T.min(), T.max())
    #     # print('Model p min-max:', p.min(), p.max())
    #
    #     specific_ie = cv * T + qv * self.Ls0 + ql * self.Lf0
    #     enthalpy = specific_ie + p / density
    #     ie = density * specific_ie
    #
    #     if update_cache:
    #         self.qv[:] = qv
    #         self.ql[:] = ql
    #         self.qi[:] = qi
    #
    #     return enthalpy, T, p, ie, mu, qv, ql