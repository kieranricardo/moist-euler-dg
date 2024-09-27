import numpy as np
from moist_euler_dg.two_phase_euler_2D import TwoPhaseEuler2D
from _moist_euler_dg import two_phase_thermo, fmoist_euler_2d_dynamics


class FortranTwoPhaseEuler2D(TwoPhaseEuler2D):


    def get_thermodynamic_quantities(self, density, entropy, qw, update_cache=False, use_cache=False):

        qd = 1 - qw
        if use_cache:
            qv, ql = self.qv, self.ql
        else:
            qv, ql = np.zeros_like(density), np.zeros_like(density)
            qv[:] = qw

        T = np.zeros_like(density)
        mu = np.zeros_like(density)
        ind = np.zeros_like(density)

        two_phase_thermo.solve_fractions_from_entropy(
            qv.ravel(), ql.ravel(), T.ravel(), mu.ravel(), ind.ravel(), density.ravel(), entropy.ravel(), qw.ravel(), qv.size,
            self.Rd, self.logRd, self.Rv, self.logRv, self.cvd, self.cvv, self.cpv, self.cpd, self.cl,
            self.T0, self.logT0, self.p0, self.logp0, self.Lv0, self.c0, self.c1
        )

        print('Ind:', ind.min(), ind.max(), (ind==0).mean())
        if (ind == 0).any():
            mask = ind == 0
            # print(f"Warning: thermo solve not converged at t={self.time}. density={density[mask][0]}; entropy={entropy[mask][0]}; qw={qw[mask][0]}")

            raise RuntimeError(f"Error: thermo solve not converged at t={self.time}. density={density[mask][0]}; entropy={entropy[mask][0]}; qw={qw[mask][0]}")

        R = qv * self.Rv + qd * self.Rd
        cv = qd * self.cvd + qv * self.cvv + ql * self.cl

        p = density * R * T

        specific_ie = cv * T + qv * self.Lv0
        enthalpy = specific_ie + p / density
        ie = density * specific_ie


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