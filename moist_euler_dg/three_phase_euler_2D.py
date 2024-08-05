import numpy as np
from moist_euler_dg.two_phase_euler_2D import TwoPhaseEuler2D


class ThreePhaseEuler2D(TwoPhaseEuler2D):

    nvars = 9

    def __init__(self, *args, **kwargs):
        TwoPhaseEuler2D.__init__(self, *args, **kwargs)

        self.qv = np.zeros_like(self.xs)
        self.ql = np.zeros_like(self.xs)
        self.qi = np.zeros_like(self.xs)

        self.cpd = 1_004.0
        self.Rd = 287.0
        self.cvd = self.cpd - self.Rd
        self.gamma = self.cpd / self.cvd

        self.cpv = 1_885.0
        self.Rv = 461.0
        self.cvv = self.cpv - self.Rv

        self.gammav = self.cpv / self.cvv

        self.cl = 4_186.0
        self.ci = 2_106.0

        self.p0_ex = 100_000.0

        self.T0 = 273.15
        self.p0 = self.psat0 = 611.2
        self.rho0 = self.p0 / (self.Rv * self.T0)

        self.logT0 = np.log(self.T0)
        self.logp0 = np.log(self.p0)
        self.logRv = np.log(self.Rv)
        self.logRd = np.log(self.Rd)

        Lv0_ = 2.5e6
        Ls0_ = 2.834e6
        Lf0_ = Ls0_ - Lv0_

        self.Lv0 = Lv0_ + (self.cpv - self.cl) * self.T0
        self.Ls0 = Ls0_ + (self.cpv - self.ci) * self.T0
        self.Lf0 = self.Ls0 - self.Lv0  # = Lf0 + (self.ci - self.cl) * self.T0

        self.Lv0 = Lv0_ + (self.cpv - self.cl) * self.T0
        self.Ls0 = Ls0_ + (self.cpv - self.ci) * self.T0
        self.Lf0 = self.Ls0 - self.Lv0

        self.c0 = self.cpv + (self.Ls0 / self.T0) - self.cvv * self.logT0 + self.Rv * np.log(self.rho0)
        self.c1 = self.cl + (self.Lf0 / self.T0) - self.cl * self.logT0
        self.c2 = self.ci - self.ci * self.logT0

    def entropy_vapour(self, T, qv, density, np=np):
        return self.cvv * np.log(T) - self.Rv * np.log(qv * density) + self.c0

    def entropy_liquid(self, T, np=np):
        return self.cl * np.log(T) + self.c1

    def entropy_ice(self, T, np=np):
        return self.ci * np.log(T) + self.c2

    def entropy_air(self, T, qd, density, np=np):
        return self.cvd * np.log(T) - self.Rd * np.log(qd * density * self.Rd)

    def gibbs_vapour(self, T, qv, density, np=np):
        return -self.cvv * T * np.log(T / self.T0) + self.Rv * T * np.log(qv * density / self.rho0) + self.Ls0 * (1 - T / self.T0)

    def gibbs_liquid(self, T, np=np):
        return -self.cl * T * np.log(T / self.T0) + self.Lf0 * (1 - T / self.T0)

    def gibbs_ice(self, T, np=np):
        return -self.ci * T * np.log(T / self.T0)

    def get_thermodynamic_quantities(self, density, entropy, qw, update_cache=False, use_cache=False):

        qd = 1 - qw

        if use_cache:
            qv, ql, qi = self.qv, self.ql, self.qi
        else:
            qv, ql, qi = np.zeros_like(density), np.zeros_like(density), np.zeros_like(density)
            qv[:] = qw

        self.solve_fractions_from_entropy(density, qw, entropy, qv=qv, ql=ql, qi=qi)

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

        if update_cache:
            self.qv[:] = qv
            self.ql[:] = ql
            self.qi[:] = qi

        return enthalpy, T, p, ie, mu, qv, ql

    def rh_to_qw(self, rh, p, density, np=np):

        R = self.Rd
        for _ in range(100):
            T = p / (density * R)
            qv_sat = self.saturation_fraction(T, density)
            # qv_sat = pv / (density * self.Rv * T)
            qv = rh * qv_sat
            R = (1 - qv) * self.Rd + qv * self.Rv

        # qw = qv # unsaturated
        return qv

    def saturation_fraction(self, T, density, np=np):
        # -self.cvv * T * np.log(T / self.T0) + self.Rv * T * np.log(qv * density / self.rho0) + self.Ls0 * (1 - T / self.T0)
        logqsat =  self.cvv * T * np.log(T / self.T0) - self.Ls0 * (1 - T / self.T0)
        logqsat += (T <= self.T0) * self.gibbs_ice(T)
        logqsat += (T > self.T0) * self.gibbs_liquid(T)

        logqsat /= (self.Rv * T)
        return (self.rho0 / density) * np.exp(logqsat)

    def solve_fractions_from_entropy(self, density, qw, entropy, qv=None, ql=None, qi=None, iters=10, tol=1e-10):

        if qv is None:
            qv = np.copy(qw)
            ql = np.zeros_like(qw)
            qi = np.zeros_like(qw)

        logdensity = np.log(density)
        qd = 1 - qw

        # check for triple point
        qv_ = self.p0 / (self.T0 * self.Rv * density)

        ea = self.entropy_air(self.T0, qd, density)
        ev = self.entropy_vapour(self.T0, qv_, density)
        ec = entropy - qd * ea - qv_ * ev

        el = self.entropy_liquid(self.T0)
        ei = self.entropy_ice(self.T0)

        # solve:
        # el * ql + ei * qi = ec
        # ql + qi = qw - qv

        ql_ = (ec - ei * (qw - qv_)) / (el - ei)
        qi_ = (qw - qv_) - ql_

        # if verbose:
        #     print('Triple point check:')
        #     print('qw:', qw)
        #     print('qv:', qv_)
        #     print('ql:', ql_)
        #     print('qi:', qi_, '\n')

        triple = 1.0 * (ql_ >= 0.0) * (qi_ >= 0.0)

        logqw = np.log(qw)
        logqd = np.log(qd)

        # check if all vapour
        R = qw * self.Rv + qd * self.Rd
        cv = qd * self.cvd + qw * self.cvv
        cvlogT = entropy + R * logdensity + qd * self.Rd * (logqd + self.logRd) + qw * self.Rv * logqw
        cvlogT += -qw * self.c0
        logT = (1 / cv) * cvlogT
        T = np.exp(logT)
        gv = self.gibbs_vapour(T, qw, density, np=np)
        all_vapour = (gv < self.gibbs_liquid(T, np=np)) * (gv < self.gibbs_ice(T, np=np)) * 1.0

        qv[:] = (1 - triple) * qv + triple * qv_
        qi[:] = (1 - triple) * qi + triple * qi_

        qv[:] = (1 - all_vapour) * qv + all_vapour * qw
        qi[:] = (1 - all_vapour) * qi

        ql[:] = qw - (qv + qi)

        is_solved = all_vapour + triple
        assert is_solved.max() <= 1.0

        has_liquid = (ql > 0) * 0.0

        def _newton_loop(density, qw, entropy, logdensity, is_solved, has_liquid, qd, qv, ql, qi):
            for _ in range(iters):

                # solve for temperature and pv
                R = qv * self.Rv + qd * self.Rd
                cv = qd * self.cvd + qv * self.cvv + ql * self.cl + qi * self.ci

                logqv = np.log(qv)

                cvlogT = entropy + R * logdensity + qd * self.Rd * (logqd + self.logRd) + qv * self.Rv * logqv
                cvlogT += -qv * self.c0 - ql * self.c1 - qi * self.c2
                logT = (1 / cv) * cvlogT

                T = np.exp(logT)

                pv = qv * self.Rv * density * T
                logpv = logqv + self.logRv + logdensity + logT

                # calculate gradients of T and pv w.r.t. moisture concentrations
                dlogTdqv = (1 / cv) * (self.Rv * logdensity + self.Rv * logqv + self.Rv - self.c0)
                dlogTdqv += -(1 / cv) * logT * (self.cvv)

                dlogTdql = (1 / cv) * (-self.c1)
                dlogTdql += -(1 / cv) * logT * (self.cl)

                dlogTdqi = (1 / cv) * (-self.c2)
                dlogTdqi += -(1 / cv) * logT * (self.ci)

                dTdqv = dlogTdqv * T
                dTdql = dlogTdql * T
                dTdqi = dlogTdqi * T

                dpvdqv = self.Rv * density * T + qv * self.Rv * density * dTdqv
                dpvdql = qv * self.Rv * density * dTdql
                dpvdqi = qv * self.Rv * density * dTdqi

                # calculate Gibbs potentials and gradients w.r.t T and pv
                gibbs_v = -self.cpv * T * (logT - self.logT0) + self.Rv * T * (logpv - self.logp0) + self.Ls0 * (1 - T / self.T0)
                gibbs_l = -self.cl * T * (logT - self.logT0) + self.Lf0 * (1 - T / self.T0)
                gibbs_i = -self.ci * T * (logT - self.logT0)

                dgibbs_vdT = -self.cpv * (logT - self.logT0) - self.cpv + self.Rv * (logpv - self.logp0) - self.Ls0 / self.T0
                dgibbs_ldT = -self.cl * (logT - self.logT0) - self.cl - self.Lf0 / self.T0
                dgibbs_idT = -self.ci * (logT - self.logT0) - self.ci

                dgibbs_vdpv = self.Rv * T / pv

                # calculate Gibbs potentials gradients w.r.t moist concentrations
                dgibbs_vdqv = dgibbs_vdT * dTdqv + dgibbs_vdpv * dpvdqv
                dgibbs_ldqv = dgibbs_ldT * dTdqv
                dgibbs_idqv = dgibbs_idT * dTdqv

                dgibbs_vdql = dgibbs_vdT * dTdql + dgibbs_vdpv * dpvdql
                dgibbs_ldql = dgibbs_ldT * dTdql
                dgibbs_idql = dgibbs_idT * dTdql

                dgibbs_vdqi = dgibbs_vdT * dTdqi + dgibbs_vdpv * dpvdqi
                dgibbs_ldqi = dgibbs_ldT * dTdqi
                dgibbs_idqi = dgibbs_idT * dTdqi

                # update moisture species

                # if thawed (and not triple point) set qi = 0, and solve for gibbs_vapour = gibbs_liquid
                # thawed = (T > self.T0) * (1 - triple) #* (1 - all_vapourall_vapour)
                thawed = has_liquid * (1 - is_solved)
                val = (gibbs_v - gibbs_l)
                dvaldqv = (dgibbs_vdqv - dgibbs_ldqv) - (dgibbs_vdql - dgibbs_ldql)
                update = -thawed * val / dvaldqv

                # if frozen (and not triple point) set ql = 0, and solve for gibbs_vapour = gibbs_ice
                # frozen = (T < self.T0) * (1 - triple) #* (1 - all_vapour)
                frozen = (1 - has_liquid) * (1 - is_solved)
                val = (gibbs_v - gibbs_i)
                dvaldqv = (dgibbs_vdqv - dgibbs_idqv) - (dgibbs_vdqi - dgibbs_idqi)
                update = update - frozen * val / dvaldqv

                qv = qv + update

                qv = np.maximum(qv, 1e-15 + 0 * qw)

                # if triple don't update, if frozen (and not triple) set qi = qw - qv, if thawed set qi = 0
                qi = is_solved * qi + frozen * (qw - qv)
                ql = qw - (qv + qi)

                rel_update = abs(update / qv).max()
                if rel_update < tol:
                    break

            # could be issue with only vapour -- not converged?
            qv = np.minimum(qv, qw)
            qi = is_solved * qi + frozen * (qw - qv)
            ql = qw - (qv + qi)

            # if rel_update >= tol:
            #     print('Warning convergence not achieved')

            R = qv * self.Rv + qd * self.Rd
            cv = qd * self.cvd + qv * self.cvv + ql * self.cl + qi * self.ci
            logqv = np.log(qv)
            cvlogT = entropy + R * logdensity + qd * self.Rd * (logqd + self.logRd) + qv * self.Rv * logqv
            cvlogT += -qv * self.c0 - ql * self.c1 - qi * self.c2
            logT = (1 / cv) * cvlogT
            T = np.exp(logT)

            ie = cv * T + qv * self.Ls0 + ql * self.Lf0

            return qv, qi, ql, ie

        has_liquid = 0.0
        qv1, qi1, ql1, ie1 = _newton_loop(density, qw, entropy, logdensity, is_solved, has_liquid, qd, qv, ql, qi)
        has_liquid = 1.0
        qv2, qi2, ql2, ie2 = _newton_loop(density, qw, entropy, logdensity, is_solved, has_liquid, qd, qv, ql, qi)

        mask = (ie1 < ie2)

        qv[:] = mask * qv1 + (1.0 - mask) * qv2
        ql[:] = mask * ql1 + (1.0 - mask) * ql2
        qi[:] = mask * qi1 + (1.0 - mask) * qi2

        return qv, ql, qi
