import numpy as np
from moist_euler_dg.three_phase_euler_2D import ThreePhaseEuler2D
from _moist_euler_dg import fmoist_euler_2d_dynamics


class OnePhaseEuler2D(ThreePhaseEuler2D):

    def solve_fractions_from_entropy(self, density, qw, entropy, qv=None, ql=None, qi=None, iters=10, tol=1e-10):
        qv = np.copy(qw)
        ql = np.zeros_like(qw)
        qi = np.zeros_like(qw)

        return qv, ql, qi

    def get_thermodynamic_quantities(self, density, entropy, qw, update_cache=False, use_cache=False):
        qd = 1 - qw

        qv, ql, qi = self.solve_fractions_from_entropy(density, qw, entropy)

        R = qv * self.Rv + qd * self.Rd
        cv = qd * self.cvd + qv * self.cvv + ql * self.cl + qi * self.ci

        logqv = np.log(qv)
        logqd = np.log(qd)
        logdensity = np.log(density)

        cvlogT = entropy + R * logdensity + qd * self.Rd * (logqd + self.logRd) + qv * self.Rv * logqv
        cvlogT += -qv * self.c0 - ql * self.c1 - qi * self.c2
        logT = (1 / cv) * cvlogT
        T = np.exp(logT)

        p = density * R * T

        # print('Model T min-max:', T.min(), T.max())
        # print('Model p min-max:', p.min(), p.max())

        specific_ie = cv * T + qv * self.Ls0 + ql * self.Lf0
        enthalpy = specific_ie + p / density
        ie = density * specific_ie

        dlogTdqv = (1 / cv) * (self.Rv * logdensity + self.Rv * logqv + self.Rv - self.c0)
        dlogTdqv += -(1 / cv) * logT * self.cvv
        dTdqv = dlogTdqv * T

        dlogTdql = (1 / cv) * (- self.c1)
        dlogTdql += -(1 / cv) * logT * self.cl
        dTdql = dlogTdql * T

        dlogTdqi = (1 / cv) * (-self.c2)
        dlogTdqi += -(1 / cv) * logT * self.ci
        dTdqi = dlogTdqi * T

        dlogTdqd = (1 / cv) * (self.Rd * logdensity + self.Rd * (logqd + np.log(self.Rd)) + self.Rd)
        dlogTdqd += -(1 / cv) * logT * self.cvd
        dTdqd = T * dlogTdqd

        # these are just the Gibbs functions
        chemical_potential_d = cv * dTdqd + self.cvd * T
        chemical_potential_v = cv * dTdqv + self.cvv * T + self.Ls0
        chemical_potential_l = cv * dTdql + self.cl * T + self.Lf0
        chemical_potential_i = cv * dTdqi + self.ci * T

        mu = chemical_potential_v - chemical_potential_d

        # R = qv * self.Rv + qd * self.Rd
        # cv = qd * self.cvd + qv * self.cvv

        # cv = qd * self.cvd + qv * self.cvv
        # cvlogT = entropy + R * np.log(density) + qd * self.Rd * (np.log(qd) + self.logRd) + qv * self.Rv * np.log(qv)
        # cvlogT += -qv * self.c0
        # logT = (1 / cv) * cvlogT
        # T = np.exp(logT)

        # mu = self.gibbs_vapour(T, qv, density) - self.gibbs_air(T, qd, density)

        # p = density * R * T

        # specific_ie = cv * T + qv * self.Lv0
        # enthalpy = specific_ie + p / density
        # ie = density * specific_ie

        if update_cache:
            self.qv[:] = qv
            self.ql[:] = ql

        return enthalpy, T, p, ie, mu, qv, ql

    def _solve(self, state, dstatedt):
        u, w, h, s, q, T, mu, p, ie = self.get_vars(state)
        dudt, dwdt, dhdt, dsdt, dqdt, *_ = self.get_vars(dstatedt)

        fmoist_euler_2d_dynamics.solve(
            u.ravel(), w.ravel(), h.ravel(), s.ravel(), q.ravel(), T.ravel(), mu.ravel(), p.ravel(), ie.ravel(),
            dudt.ravel(), dwdt.ravel(), dhdt.ravel(), dsdt.ravel(), dqdt.ravel(),
            self.D.transpose(), self.weights_z[-1], self.J.ravel(),
            self.grad_xi_2.ravel(), self.grad_xi_dot_zeta.ravel(), self.grad_zeta_2.ravel(),
            self.nx, self.nz, self.order + 1,
            self.a, float(self.upwind), self.gamma
        )

        dudt -= self.g * self.u_grav
        dwdt -= self.g * self.w_grav

    def _solve_horz_boundaries(self, state, dstatedt):

        u, w, h, s, q, T, mu, p, ie = self.get_vars(state)
        dudt, dwdt, dhdt, dsdt, dqdt, *_ = self.get_vars(dstatedt)

        um, wm, hm, sm, qm, Tm, mum, pm, iem = (self.left_boundary[i].ravel() for i in range(self.nvars))
        up, wp, hp, sp, qp, Tp, mup, pp, iep = (self.right_boundary[i].ravel() for i in range(self.nvars))

        fmoist_euler_2d_dynamics.solve_horz_boundaries(
            u.ravel(), w.ravel(), h.ravel(), s.ravel(), q.ravel(), T.ravel(), mu.ravel(), p.ravel(), ie.ravel(),
            um, wm, hm, sm, qm, Tm, mum, pm, iem,
            up, wp, hp, sp, qp, Tp, mup, pp, iep,
            dudt.ravel(), dwdt.ravel(), dhdt.ravel(), dsdt.ravel(), dqdt.ravel(),
            self.D.transpose(), self.weights_z[-1], self.J.ravel(),
            self.grad_xi_2.ravel(), self.grad_xi_dot_zeta, self.grad_zeta_2.ravel(),
            self.nx, self.nz, self.order + 1,
            self.a, float(self.upwind), self.gamma,
        )

        return dstatedt