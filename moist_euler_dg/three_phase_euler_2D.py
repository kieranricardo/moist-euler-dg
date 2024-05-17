import numpy as np
from imex_solvers import utils
from moist_euler_dg.two_phase_euler_2D import TwoPhaseEuler2D
import scipy
from imex_solvers.utils import block_matmat, block_matvec, broadcast_matmat
from mpi4py import MPI
import time
import os


class ThreePhaseEuler2D(TwoPhaseEuler2D):

    nvars = 5

    def __init__(self, *args, **kwargs):
        TwoPhaseEuler2D.__init__(self, *args, **kwargs)
        self.ql = None
        self.qi = None

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

        Lv0_ = 2.5e6
        Ls0_ = 2.834e6
        Lf0_ = Ls0_ - Lv0_

        self.Lv0 = Lv0_ + (self.cpv - self.cl) * self.T0
        self.Ls0 = Ls0_ + (self.cpv - self.ci) * self.T0
        self.Lf0 = self.Ls0 - self.Lv0  # = Lf0 + (self.ci - self.cl) * self.T0

        self.Lv0 = Lv0_ + (self.cpv - self.cl) * self.T0
        self.Ls0 = Ls0_ + (self.cpv - self.ci) * self.T0
        self.Lf0 = self.Ls0 - self.Lv0

        self.c0 = self.cpv + (self.Ls0 / self.T0) - self.cvv * np.log(self.T0) + self.Rv * np.log(self.rho0)
        self.c1 = self.cl + (self.Lf0 / self.T0) - self.cl * np.log(self.T0)
        self.c2 = self.ci - self.ci * np.log(self.T0)

    def entropy_vapour(self, T, qv, density, mathlib=np):
        return self.cvv * mathlib.log(T) - self.Rv * mathlib.log(qv * density) + self.c0

    def entropy_liquid(self, T, mathlib=np):
        return self.cl * mathlib.log(T) + self.c1

    def entropy_ice(self, T, mathlib=np):
        return self.ci * mathlib.log(T) + self.c2

    def entropy_air(self, T, qd, density, mathlib=np):
        return self.cvd * mathlib.log(T) - self.Rd * mathlib.log(qd * density * self.Rd)

    def gibbs_vapour(self, T, qv, density, mathlib=np):
        return -self.cvv * T * mathlib.log(T / self.T0) + self.Rv * T * mathlib.log(qv * density / self.rho0) + self.Ls0 * (1 - T / self.T0)

    def gibbs_liquid(self, T, mathlib=np):
        return -self.cl * T * mathlib.log(T / self.T0) + self.Lf0 * (1 - T / self.T0)

    def gibbs_ice(self, T, mathlib=np):
        return -self.ci * T * mathlib.log(T / self.T0)

    def get_thermodynamic_quantities(self, density, entropy, qw):

        qd = 1 - qw

        qv, ql, qi = self.solve_fractions_from_entropy(density, qw, entropy, mathlib=np, qv=self.qv, ql=self.ql, qi=self.qi)

        R = qv * self.Rv + qd * self.Rd
        cv = qd * self.cvd + qv * self.cvv + ql * self.cl + qi * self.ci

        logqv = np.log(qv)
        logqd = np.log(qd)
        logdensity = np.log(density)

        cvlogT = entropy + R * logdensity + qd * self.Rd * (logqd + np.log(self.Rd)) + qv * self.Rv * logqv
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

        return enthalpy, T, p, ie, mu

    def rh_to_qw(self, rh, p, density, mathlib=np):

        R = self.Rd
        for _ in range(100):
            T = p / (density * R)
            qv_sat = self.saturation_fraction(T, density)
            # qv_sat = pv / (density * self.Rv * T)
            qv = rh * qv_sat
            R = (1 - qv) * self.Rd + qv * self.Rv

        # qw = qv # unsaturated
        return qv

    def saturation_fraction(self, T, density, mathlib=np):
        # -self.cvv * T * mathlib.log(T / self.T0) + self.Rv * T * mathlib.log(qv * density / self.rho0) + self.Ls0 * (1 - T / self.T0)
        logqsat =  self.cvv * T * mathlib.log(T / self.T0) - self.Ls0 * (1 - T / self.T0)
        logqsat += (T <= self.T0) * self.gibbs_ice(T)
        logqsat += (T > self.T0) * self.gibbs_liquid(T)

        logqsat /= (self.Rv * T)
        return (self.rho0 / density) * mathlib.exp(logqsat)

    def solve_fractions_from_entropy(self, density, qw, entropy, mathlib=np, iters=20, qv=None, ql=None, qi=None, verbose=False, tol=1e-10):

        logdensity = mathlib.log(density)
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

        # check if all vapour
        R = qw * self.Rv + qd * self.Rd
        cv = qd * self.cvd + qw * self.cvv
        logqw = mathlib.log(qw)
        cvlogT = entropy + R * logdensity + qd * self.Rd * mathlib.log(self.Rd * qd) + qw * self.Rv * logqw
        cvlogT += -qw * self.c0
        logT = (1 / cv) * cvlogT
        T = mathlib.exp(logT)
        gv = self.gibbs_vapour(T, qw, density, mathlib=mathlib)
        all_vapour = (gv < self.gibbs_liquid(T, mathlib=mathlib)) * (gv < self.gibbs_ice(T, mathlib=mathlib)) * 1.0

        if (qv is None) or (qi is None):
            qv = qw
            qi = 0.0 * qw
            iters = 40

        qv = (1 - triple) * qv + triple * qv_
        qi = (1 - triple) * qi + triple * qi_

        qv = (1 - all_vapour) * qv + all_vapour * qw
        qi = (1 - all_vapour) * qi

        ql = qw - (qv + qi)

        is_solved = all_vapour + triple
        assert is_solved.max() <= 1.0

        has_liquid = (ql >= 0) * 1.0

        def _newton_loop(density, qw, entropy, logdensity, is_solved, has_liquid, qd, qv, ql, qi):
            for _ in range(iters):

                # solve for temperature and pv
                R = qv * self.Rv + qd * self.Rd
                cv = qd * self.cvd + qv * self.cvv + ql * self.cl + qi * self.ci

                logqv = mathlib.log(qv)

                cvlogT = entropy + R * logdensity + qd * self.Rd * mathlib.log(self.Rd * qd) + qv * self.Rv * logqv
                cvlogT += -qv * self.c0 - ql * self.c1 - qi * self.c2
                logT = (1 / cv) * cvlogT

                T = mathlib.exp(logT)

                pv = qv * self.Rv * density * T
                logpv = logqv + np.log(self.Rv) + logdensity + logT

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
                gibbs_v = -self.cpv * T * (logT - np.log(self.T0)) + self.Rv * T * (logpv - np.log(self.p0)) + self.Ls0 * (1 - T / self.T0)
                gibbs_l = -self.cl * T * (logT - np.log(self.T0)) + self.Lf0 * (1 - T / self.T0)
                gibbs_i = -self.ci * T * (logT - np.log(self.T0))

                dgibbs_vdT = -self.cpv * (logT - np.log(self.T0)) - self.cpv + self.Rv * (logpv - np.log(self.p0)) - self.Ls0 / self.T0
                dgibbs_ldT = -self.cl * (logT - np.log(self.T0)) - self.cl - self.Lf0 / self.T0
                dgibbs_idT = -self.ci * (logT - np.log(self.T0)) - self.ci

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

                qv = mathlib.maximum(qv, 1e-15 + 0 * qw)

                # if triple don't update, if frozen (and not triple) set qi = qw - qv, if thawed set qi = 0
                qi = is_solved * qi + frozen * (qw - qv)
                ql = qw - (qv + qi)

                rel_update = abs(update / qv).max()
                if rel_update < tol:
                    break

            # could be issue with only vapour -- not converged?
            qv = mathlib.minimum(qv, qw)
            qi = is_solved * qi + frozen * (qw - qv)
            ql = qw - (qv + qi)

            if rel_update >= tol:
                print('Warning convergence not achieved')

            return qv, qi, ql

        qv, qi, ql = _newton_loop(density, qw, entropy, logdensity, is_solved, has_liquid, qd, qv, ql, qi)

        R = qv * self.Rv + qd * self.Rd
        cv = qd * self.cvd + qv * self.cvv + ql * self.cl + qi * self.ci
        logqv = mathlib.log(qv)
        cvlogT = entropy + R * logdensity + qd * self.Rd * mathlib.log(self.Rd * qd) + qv * self.Rv * logqv
        cvlogT += -qv * self.c0 - ql * self.c1 - qi * self.c2
        logT = (1 / cv) * cvlogT
        T = mathlib.exp(logT)

        is_solved = is_solved + (1 - is_solved) * (T > self.T0) * has_liquid
        is_solved = is_solved + (1 - is_solved) * (T < self.T0) * (1 - has_liquid)
        assert is_solved.max() <= 1.0

        if is_solved.min() == 0.0:
            if verbose:
                print('Running loop 2')
            has_liquid = 1 - has_liquid
            qv, qi, ql = _newton_loop(density, qw, entropy, logdensity, is_solved, has_liquid, qd, qw, 0.0 * qw, 0.0 * qw)


        # TODO: faster way to solve for single phase


        # if verbose:
        #     print('T:', T)
        #     print('Rel update:', rel_update)
        #     print('qv:', qv)
        #     print('ql:', ql)
        #     print('qi:', qi)
        #     print(f'Gibbs error 1: {(gibbs_v - gibbs_l)}')
        #     print(f'Gibbs error 2: {(gibbs_v - gibbs_i)}')
        #     print(f'Gibbs v: {gibbs_v}')
        #     print('T:', T)
        #     print('rel update:', rel_update, '\n')

        return qv, ql, qi
