import numpy as np
from moist_euler_dg.three_phase_euler_2D import ThreePhaseEuler2D
from _moist_euler_dg import three_phase_thermo, fmoist_euler_2d_dynamics


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

        T = np.zeros_like(density)
        mu = np.zeros_like(density)
        three_phase_thermo.solve_fractions_from_entropy(
            qv.ravel(), ql.ravel(), qi.ravel(), T.ravel(), mu.ravel(), ind.ravel(), density.ravel(), entropy.ravel(), qw.ravel(), qv.size,
            self.Rd, self.logRd, self.Rv, self.logRv, self.cvd, self.cvv, self.cpv, self.cpd, self.cl, self.ci,
            self.T0, self.logT0, self.p0, self.logp0, self.Lf0, self.Ls0, self.c0, self.c1, self.c2
        )

        is_solved = (ind > 0)
        qi[:] = qw - (qv + ql)
        if (~is_solved).any():
            print('Thermo solver failed')

        return qv, ql, qi


    def get_thermodynamic_quantities(self, density, entropy, qw, update_cache=False, use_cache=False):

        qd = 1 - qw
        if use_cache:
            qv, ql, qi = self.qv, self.ql, self.qi
            qv_cache, ql_cache, qi_cache = np.copy(qv), np.copy(ql), np.copy(qi)
        else:
            qv, ql, qi = np.zeros_like(density), np.zeros_like(density), np.zeros_like(density)
            qv[:] = qw

        T = np.zeros_like(density)
        mu = np.zeros_like(density)
        ind = np.zeros_like(density)

        three_phase_thermo.solve_fractions_from_entropy(
            qv.ravel(), ql.ravel(), qi.ravel(), T.ravel(), mu.ravel(), ind.ravel(), density.ravel(), entropy.ravel(), qw.ravel(), qv.size,
            self.Rd, self.logRd, self.Rv, self.logRv, self.cvd, self.cvv, self.cpv, self.cpd, self.cl, self.ci,
            self.T0, self.logT0, self.p0, self.logp0, self.Lf0, self.Ls0, self.c0, self.c1, self.c2
        )

        if (ind == 0).any():
            mask = ind == 0
            print(f"Warning: thermo solve not converged at t={self.time}. density={density[mask][0]}; entropy={entropy[mask][0]}; qw={qw[mask][0]}")
            # raise RuntimeError(f"Error: thermo solve not converged at t={self.time}. density={density[mask][0]}; entropy={entropy[mask][0]}; qw={qw[mask][0]}")

        R = qv * self.Rv + qd * self.Rd
        cv = qd * self.cvd + qv * self.cvv + ql * self.cl + qi * self.ci


        p = density * R * T

        specific_ie = cv * T + qv * self.Ls0 + ql * self.Lf0
        enthalpy = specific_ie + p / density
        ie = density * specific_ie


        if update_cache:
            self.qv[:] = qv
            self.ql[:] = ql
            self.qi[:] = qi

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
            self.a, float(self.upwind), self.gamma, self.b
        )

        dudt -= self.g * self.u_grav
        dwdt -= self.g * self.w_grav

        if self.sst is not None:
            assert self.sst > self.T0

            ip = self.ip_vert_ext
            normal_vel = (self.grad_xi_dot_zeta[ip] * u[ip] + self.grad_zeta_2[ip] * w[ip])

            qv_sat = self.saturation_fraction(T[ip], h[ip])
            qv = np.minimum(qv_sat, q[ip])

            density_dry = h[ip] * (1 - q[ip]) # use lowest level dry density
            density_vapour = self.saturation_density(self.sst) # use saturated vapour density at SST
            h_bdry = density_dry + density_vapour
            qv_bdry = density_vapour / h_bdry

            mask = normal_vel > 0
            water_mass_flux = h[ip] * normal_vel * (qv_bdry - qv) / (1 - q[ip])
            water_mass_flux *= mask

            # if qv_bdry > qv, positive vapour mass flux
            # use entropy of vapour at sst (this enters the domain)
            s_bdry = self.entropy_vapour(self.sst, qv_bdry, h_bdry) * (qv_bdry > qv)
            # if qv_bdry < qv, negative vapour mass flux
            # use entropy of vapour at lowest level (this leaves the domain)
            s_bdry += self.entropy_vapour(T[ip], qv, h[ip]) * (qv_bdry <= qv)

            # evaporation
            dhdt[ip] += water_mass_flux / self.weights_z[-1] # - normal_vel * h[ip]
            dqdt[ip] += (1 / h[ip]) * water_mass_flux * (1 - q[ip]) / self.weights_z[-1]
            # TODO: add this back in
            dsdt[ip] += (1 / h[ip]) * water_mass_flux * (water_mass_flux > 0) * (s_bdry - s[ip]) / self.weights_z[-1]

            # sensible heat flux
            s_bdry = self.entropy(h[ip], q[ip], T=self.sst)
            dsdt[ip] += normal_vel * (s_bdry - s[ip]) * mask / self.weights_z[-1]

            u_bdry = 0
            dudt[ip] += normal_vel * (u_bdry - u[ip]) * mask / self.weights_z[-1]

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