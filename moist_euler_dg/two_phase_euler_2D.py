import numpy as np
from imex_solvers import utils
from moist_euler_dg.euler_2D import Euler2D
import scipy
from imex_solvers.utils import block_matmat, block_matvec, broadcast_matmat
from mpi4py import MPI
import time
import os


class TwoPhaseEuler2D(Euler2D):

    nvars = 5

    def __init__(self, *args, **kwargs):
        Euler2D.__init__(self, *args, **kwargs)
        self.qv = None

        # thermodynamic constants
        self.cpd = 1_004.0
        self.Rd = 287.0
        self.cvd = self.cpd - self.Rd
        self.gamma = self.cpd / self.cvd

        self.cpv = 1_885.0
        self.Rv = 461.0
        self.cvv = self.cpv - self.Rv

        self.gammav = self.cpv / self.cvv

        self.cl = 4_186.0

        self.p0_ex = 100_000.0

        self.T0 = 273.15
        self.p0 = self.psat0 = 611.2
        self.Lv0 = 3.1285e6

        self.c0 = self.cpv + (self.Lv0 / self.T0) - self.cpv * np.log(self.T0) + self.Rv * np.log(self.p0)
        self.c1 = self.cl - self.cl * np.log(self.T0)

    def get_fluxes(self, u, w, h, s, q, idx=slice(None)):
        vel_norm = self.grad_xi_2[idx] * u ** 2 + 2 * self.grad_xi_dot_zeta[idx] * u * w + self.grad_zeta_2[idx] * w ** 2
        e, T, p, _, mu = self.get_thermodynamic_quantities(h, s, q)
        c_sound = np.sqrt(self.gamma * p / h)
        G = 0.5 * vel_norm + e - T * s
        Fx = h * (self.grad_xi_2[idx] * u + self.grad_xi_dot_zeta[idx] * w)
        Fz = h * (self.grad_xi_dot_zeta[idx] * u + self.grad_zeta_2[idx] * w)

        return G, c_sound, T, mu, Fx, Fz

    def _solve(self, state, dstatedt):

        u, w, h, s, q = self.get_vars(state)
        dudt, dwdt, dhdt, dsdt, dqdt = self.get_vars(dstatedt)

        G, c_sound, T, mu, Fx, Fz = self.get_fluxes(u, w, h, s, q)

        # density evolutions
        divF = self.ddxi(self.J * Fx) + self.ddzeta(self.J * Fz)
        divF /= self.J
        dhdt -= divF

        # velocity evolution
        vort = (self.ddxi(w) - self.ddzeta(u)) / self.J + self.f

        u_perp = (-self.drdxi_2 * w + self.dr_xi_dot_zeta * u) / self.J
        w_perp = (-self.dr_xi_dot_zeta * w + self.drdzeta_2 * u) / self.J

        dudt -= self.ddxi(G) + self.g * self.u_grav + vort * u_perp
        dwdt -= self.ddzeta(G) + self.g * self.w_grav + vort * w_perp

        # handle tracers
        for (dadt, a, b) in [(dsdt, s, T), (dqdt, q, mu)]:
            dadx = self.ddxi(a)
            dbdx = self.ddxi(b)
            dabdx = self.ddxi(a * b)

            dadz = self.ddzeta(a)
            dbdz = self.ddzeta(b)
            dabdz = self.ddzeta(a * b)

            dudt -= 0.5 * (a * dbdx + dabdx - b * dadx)
            dwdt -= 0.5 * (a * dbdz + dabdz - b * dadz)

            divA = self.ddxi(self.J * a * Fx) + self.ddzeta(self.J * a * Fz)
            divA /= self.J
            dadt -= 0.5 * (divA - a * divF + Fx * dadx + Fz * dadz) / h

        # bottom wall BCs
        ip = self.ip_vert_ext
        dhdt[ip] += (0.0 - Fz[ip]) / self.weights_z[-1]
        normal_vel = Fz[ip] / (self.norm_grad_zeta[ip] * h[ip])
        dwdt[ip] += -2 * self.a * (c_sound[ip] + np.abs(normal_vel)) * normal_vel / self.weights_z[-1]

        # top wall BCs
        if self.top_bc == 'wall':
            im = self.im_vert_ext
            dhdt[im] += -(0.0 - Fz[im]) / self.weights_z[-1]
            normal_vel = Fz[im] / (self.norm_grad_zeta[im] * h[im])
            dwdt[im] += -2 * self.a * (c_sound[im] + np.abs(normal_vel)) * normal_vel / self.weights_z[-1]
        elif self.top_bc == 'open':
            im = self.im_vert_ext
            state_m, dstatedt_m = self.get_boundary_data(state, im), self.get_boundary_data(dstatedt, im)
            dstatedt_p = np.zeros_like(dstatedt_m)
            self.solve_boundaries(self.top_boundary, state_m, dstatedt_p, dstatedt_m, 'z', idx=im)
        else:
            raise NotImplementedError

        # vertical interior boundaries
        ip = self.ip_vert_int
        im = self.im_vert_int
        state_p, dstatedt_p = self.get_boundary_data(state, ip), self.get_boundary_data(dstatedt, ip)
        state_m, dstatedt_m = self.get_boundary_data(state, im), self.get_boundary_data(dstatedt, im)
        self.solve_boundaries(state_p, state_m, dstatedt_p, dstatedt_m, 'z', idx=ip)

        # horizontal interior boundaries
        ip = self.ip_horz_int
        im = self.im_horz_int
        state_p, dstatedt_p = self.get_boundary_data(state, ip), self.get_boundary_data(dstatedt, ip)
        state_m, dstatedt_m = self.get_boundary_data(state, im), self.get_boundary_data(dstatedt, im)
        self.solve_boundaries(state_p, state_m, dstatedt_p, dstatedt_m, 'x', idx=ip)

        return dstatedt

    def solve_boundaries(self, state_p, state_m, dstatedt_p, dstatedt_m, direction, idx):

        up, wp, hp, sp, qp = (state_p[i] for i in range(self.nvars))
        um, wm, hm, sm, qm = (state_m[i] for i in range(self.nvars))

        dudtp, dwdtp, dhdtp, dsdtp, dqdtp = (dstatedt_p[i] for i in range(self.nvars))
        dudtm, dwdtm, dhdtm, dsdtm, dqdtm = (dstatedt_m[i] for i in range(self.nvars))

        # calculate fluxes
        Gp, cp, Tp, mup, Fxp, Fzp = self.get_fluxes(up, wp, hp, sp, qp, idx)
        Gm, cm, Tm, mum, Fxm, Fzm = self.get_fluxes(um, wm, hm, sm, qm, idx)

        if direction == 'z':
            norm_conrta = self.norm_grad_zeta[idx]
            norm_cov = self.norm_drdxi[idx]
            Fp, Fm = Fzp, Fzm
            dveldtp, dveldtm = dwdtp, dwdtm
            dtan_veldtp, dtan_veldtm = dudtp, dudtm
            tan_velp, tan_velm = up, um
        else:
            norm_conrta = self.norm_grad_xi[idx]
            norm_cov = self.norm_drdzeta[idx]
            Fp, Fm = Fxp, Fxm
            dveldtp, dveldtm = dudtp, dudtm
            dtan_veldtp, dtan_veldtm = dwdtp, dwdtm
            tan_velp, tan_velm = wp, wm

        J = self.J[idx]
        dr_xi_dot_zeta = self.dr_xi_dot_zeta[idx]
        drdxi_2 = self.drdxi_2[idx]
        drdzeta_2 = self.drdzeta_2[idx]

        normal_vel_p = Fp / (hp * norm_conrta)
        normal_vel_m = Fm / (hm * norm_conrta)

        c_adv = np.abs(0.5 * (normal_vel_p + normal_vel_m))
        c_snd = 0.5 * (cp + cm)

        F_num_flux = 0.5 * (Fp + Fm) - self.ah * (c_adv + c_snd) * (hp - hm) * norm_conrta

        fluxp = Gp
        fluxm = Gm
        num_flux = 0.5 * (fluxp + fluxm)
        dveldtp += (num_flux - fluxp) / self.weights_z[-1]
        dveldtm += -(num_flux - fluxm) / self.weights_z[-1]

        fluxp = Fp
        fluxm = Fm
        dhdtp += (F_num_flux - fluxp) / self.weights_z[-1]
        dhdtm += -(F_num_flux - fluxm) / self.weights_z[-1]

        # handle tracers
        tracer_vars = [(dsdtp, sp, Tp, dsdtm, sm, Tm), (dqdtp, qp, mup, dqdtm, qm, mum)]
        for (dadtp, ap, bp, dadtm, am, bm) in tracer_vars:
            if self.upwind:
                ahat = (F_num_flux >= 0) * am + (F_num_flux < 0) * ap
            else:
                ahat = 0.5 * (am + ap)
            fluxp = bp
            fluxm = bm
            num_flux = 0.5 * (fluxp + fluxm)
            dveldtp += ahat * (num_flux - fluxp) / self.weights_z[-1]
            dveldtm += -ahat * (num_flux - fluxm) / self.weights_z[-1]

            dsdtp += (F_num_flux / hp) * (ahat - ap) / self.weights_z[-1]
            dsdtm += -(F_num_flux / hm) * (ahat - am) / self.weights_z[-1]

        # dissipation from jump in normal direction
        normal_jump = normal_vel_p - normal_vel_m
        diss = -self.a * (c_adv + c_snd) * normal_jump

        dveldtp += diss / self.weights_z[-1]
        dveldtm += -diss / self.weights_z[-1]

        # dissipation from jump in tangent direction
        tang_jump = tan_velp - tan_velm
        diss = -self.a * c_adv * tang_jump

        dtan_veldtp += diss * norm_cov / (J * self.weights_z[-1])
        dtan_veldtm += -diss * norm_cov / (J * self.weights_z[-1])

        dveldtp += diss * dr_xi_dot_zeta / (J * self.weights_z[-1] * norm_cov)
        dveldtm += -diss * dr_xi_dot_zeta / (J * self.weights_z[-1] * norm_cov)

        # vorticity terms
        if direction == 'z':
            fluxp = -up
            fluxm = -um
        else:
            fluxp = wp
            fluxm = wm

        u_perp_p = (-drdxi_2 * wp + dr_xi_dot_zeta * up) / J
        w_perp_p = (-dr_xi_dot_zeta * wp + drdzeta_2 * up) / J
        u_perp_m = (-drdxi_2 * wm + dr_xi_dot_zeta * um) / J
        w_perp_m = (-dr_xi_dot_zeta * wm + drdzeta_2 * um) / J

        num_flux = 0.5 * (fluxp + fluxm)
        dudtp += u_perp_p * (num_flux - fluxp) / (J * self.weights_z[-1])
        dudtm += -u_perp_m * (num_flux - fluxm) / (J * self.weights_z[-1])
        dwdtp += w_perp_p * (num_flux - fluxp) / (J * self.weights_z[-1])
        dwdtm += -w_perp_m * (num_flux - fluxm) / (J * self.weights_z[-1])

        return 0.0

    def energy(self):
        pe = self.h * self.g * self.zs
        ke = 0.5 * self.h * (self.u ** 2 + self.w ** 2)
        ie = self.h * self.get_thermodynamic_quantities(self.h, self.s, self.q)[3]
        energy = pe + ke + ie
        return self.integrate(energy)

    @property
    def q(self):
        return self.get_vars(self.state)[4]

    @property
    def hq(self):
        return self.q * self.h

    def entropy_vapour(self, T, qv, density, mathlib=np):
        return self.cvv * mathlib.log(T) - self.Rv * mathlib.log(qv * self.Rv) - self.Rv * mathlib.log(density) + self.c0

    def entropy_liquid(self, T, mathlib=np):
        return self.cl * mathlib.log(T) + self.c1

    def entropy_air(self, T, qd, density, mathlib=np):
        return self.cvd * mathlib.log(T) - self.Rd * mathlib.log(qd * self.Rd) - self.Rd * mathlib.log(density)

    def gibbs_vapour(self, T, pv, mathlib=np):
        return -self.cpv * T * mathlib.log(T / self.T0) + self.Rv * T * mathlib.log(pv / self.p0) + self.Lv0 * (1 - T / self.T0)

    def gibbs_liquid(self, T, mathlib=np):
        return -self.cl * T * mathlib.log(T / self.T0)

    def saturation_pressure(self, T, mathlib=np):
        tmp = self.cpv * mathlib.log(T / self.T0) + (self.Lv0 / self.T0) - (self.Lv0 / T) - self.cl * mathlib.log(T / self.T0)
        logpsat = mathlib.log(self.p0) + (1 / self.Rv) * tmp
        return mathlib.exp(logpsat)

    def get_thermodynamic_quantities(self, h, s, qw):

        qd = 1 - qw

        qv = self.solve_qv_from_entropy(h, qw, s, qv=self.qv)

        qv = np.minimum(qv, qw)

        ql = qw - qv
        R = qv * self.Rv + qd * self.Rd
        cv = qd * self.cvd + qv * self.cvv + ql * self.cl

        logT = (1 / cv) * (s + R * np.log(h) + qd * self.Rd * np.log(self.Rd * qd) + qv * self.Rv * np.log(self.Rv * qv) - qv * self.c0 - ql * self.c1)
        T = np.exp(logT)
        p = h * R * T

        # print('Model T min-max:', T.min(), T.max())
        # print('Model p min-max:', p.min(), p.max())

        specific_ie = cv * T + qv * self.Lv0
        enthalpy = specific_ie + p / h
        ie = h * specific_ie

        dlogTdqv = (1 / cv) * (self.Rv * np.log(h) + self.Rv * np.log(self.Rv * qv) + self.Rv - self.c0)
        dlogTdqv += -(1 / cv) * logT * self.cvv
        dTdqv = dlogTdqv * T

        dlogTdql = (1 / cv) * (-self.c1)
        dlogTdql += -(1 / cv) * logT * self.cl
        dTdql = dlogTdql * T

        dlogTdqd = (1 / cv) * (self.Rd * np.log(h) + self.Rd * np.log(self.Rd * qd) + self.Rd)
        dlogTdqd += -(1 / cv) * logT * self.cvd
        dTdqd = dlogTdqd * T

        # these are just the Gibbs functions
        chemical_potential_d = cv * dTdqd + self.cvd * T
        chemical_potential_v = cv * dTdqv + self.cvv * T + self.Lv0
        chemical_potential_l = cv * dTdql + self.cl * T

        mu = chemical_potential_v - chemical_potential_d

        return enthalpy, T, p, ie, mu

    def solve_qv_from_entropy(self, density, qw, entropy, iters=10, qv=None, verbose=False, tol=1e-10):

        if qv is None:
            qv = 1e-3 + 0 * qw
            iters = 40

        logdensity = np.log(density)

        for _ in range(iters):
            qd = 1 - qw
            ql = qw - qv
            R = qv * self.Rv + qd * self.Rd
            cv = qd * self.cvd + qv * self.cvv + ql * self.cl

            # majority of time not from logs....? wtf
            # logh = np.log(density)
            # logqv = np.log(qv)

            logqv = np.log(qv)
            logT = (1 / cv) * (entropy + R * logdensity + qd * self.Rd * np.log(self.Rd * qd) + qv * self.Rv * (logqv + np.log(self.Rv)) - qv * self.c0 - ql * self.c1)
            dlogTdqv = (1 / cv) * (self.Rv * logdensity + self.Rv * (logqv + np.log(self.Rv)) + self.Rv - self.c0 + self.c1)
            dlogTdqv += -(1 / cv) * logT * (self.cvv - self.cl)

            # logT = (1 / cv) * (entropy + R * logdensity + qd * self.Rd * np.log(self.Rd * qd) + qv * self.Rv * np.log(self.Rv * qv) - qv * self.c0 - ql * self.c1)
            # dlogTdqv = (1 / cv) * (self.Rv * logdensity + self.Rv * np.log(self.Rv * qv) + self.Rv - self.c0 + self.c1)
            # dlogTdqv += -(1 / cv) * logT * (self.cvv - self.cl)

            T = np.exp(logT)
            p = R * density * T
            pv = qv * self.Rv * density * T
            logpv = logqv + np.log(self.Rv) + logdensity + logT

            dTdqv = dlogTdqv * T

            dpvdqv = qv * self.Rv * density * dTdqv + self.Rv * density * T

            # dgvdT = -self.cpv * np.log(T / self.T0) - self.cpv + self.Rv * np.log(pv / self.p0) - self.Lv0 / self.T0
            dgvdT = -self.cpv * (logT - np.log(self.T0)) - self.cpv + self.Rv * (logpv - np.log(self.p0)) - self.Lv0 / self.T0
            dgvdpv = self.Rv * T / pv

            dgldT = -self.cl * (logT - np.log(self.T0)) - self.cl

            dgvdqv = dgvdT * dTdqv + dgvdpv * dpvdqv
            dgldqv = dgldT * dTdqv

            grad = dgvdqv - dgldqv

            val = -self.cpv * T * (logT - np.log(self.T0)) + self.Rv * T * (logpv - np.log(self.p0)) + self.Lv0 * (1 - T / self.T0)
            val -= -self.cl * T * (logT - np.log(self.T0))

            # val = self.gibbs_vapour(T, pv, np=np) - self.gibbs_liquid(T, np=np)

            qv = qv - (val / grad)
            qv = np.maximum(qv, 1e-7 + 0 * qv)
            rel_update = abs((val / grad) / qv).max()
            if rel_update < tol:
                break

        self.gibbs_error = abs(val).max()
        if verbose:
            rel_update = abs((val / grad) / qv)
            print('Max relative last update:', rel_update.max())
            print('Max Gibbs error:', self.gibbs_error)

        return np.minimum(qv, qw)
    
    def solve_qv_from_p(self, density, qw, p, verbose=False):
        qv = 1e-3 + 0 * qw

        for _ in range(30):
            qd = 1 - qw
            ql = qw - qv
            R = qv * self.Rv + qd * self.Rd
            T = p / (density * R)
            pv = qv * self.Rv * density * T

            dTdqv = -(p / density) * (1 / R ** 2) * self.Rv
            dpvdqv = qv * self.Rv * density * dTdqv + self.Rv * density * T

            dgvdT = -self.cpv * np.log(T / self.T0) - self.cpv + self.Rv * np.log(pv / self.p0) - self.Lv0 / self.T0
            dgvdpv = self.Rv * T / pv

            dgldT = -self.cl * np.log(T / self.T0) - self.cl

            dgvdqv = dgvdT * dTdqv + dgvdpv * dpvdqv
            dgldqv = dgldT * dTdqv

            grad = dgvdqv - dgldqv
            val = self.gibbs_vapour(T, pv) - self.gibbs_liquid(T)

            qv = qv - (val / grad)
            qv = np.maximum(qv, 1e-7 + 0 * qv)

        if verbose:
            rel_update = abs((val / grad) / qv)
            print('Max relative last update:', rel_update.max())
            print('Max Gibbs error:', abs(val).max())

        return np.minimum(qv, qw)

    def solve_qv_from_enthalpy(self, enthalpy, qw, entropy, iters=20, qv=None, verbose=False):

        if qv is None:
            qv = 1e-3 + 0 * qw
            iters = 40

        for _ in range(iters):
            qd = 1 - qw
            ql = qw - qv
            R = qv * self.Rv + qd * self.Rd
            cv = qd * self.cvd + qv * self.cvv + ql * self.cl
            cp = qd * self.cpd + qv * self.cpv + ql * self.cl

            T = (enthalpy - qv * self.Lv0) / cp
            dTdqv = -(self.Lv0 / cp) - ((enthalpy - qv * self.Lv0) / cp ** 2) * (self.cpv - self.cl)

            logT = np.log(T)
            logqv = np.log(qv)

            logdensity = (1 / R) * (cv * logT - entropy - qd * self.Rd * np.log(self.Rd * qd) - qv * self.Rv * (logqv + np.log(self.Rv)) + qv * self.c0 + ql * self.c1)
            density = np.exp(logdensity)

            densitydT = (1 / R) * cv / T

            ddensitydqv = (1 / R) * ((self.cvv - self.cl) * logT - self.Rv * np.log(self.Rv * qv) - self.Rv + self.c0 - self.c1) * density
            ddensitydqv += -(1 / R) * self.Rv * logdensity * density
            ddensitydqv += densitydT * dTdqv

            # logT = (1 / cv) * (entropy + R * logdensity + qd * self.Rd * np.log(self.Rd * qd) + qv * self.Rv * np.log(self.Rv * qv) - qv * self.c0 - ql * self.c1)
            # dlogTdqv = (1 / cv) * (self.Rv * logdensity + self.Rv * np.log(self.Rv * qv) + self.Rv - self.c0 + self.c1)
            # dlogTdqv += -(1 / cv) * logT * (self.cvv - self.cl)

            p = R * density * T
            pv = qv * self.Rv * density * T
            logpv = logqv + np.log(self.Rv) + logdensity + logT

            dpvdqv = qv * self.Rv * density * dTdqv + self.Rv * density * T + qv * self.Rv * ddensitydqv * T

            # dgvdT = -self.cpv * np.log(T / self.T0) - self.cpv + self.Rv * np.log(pv / self.p0) - self.Lv0 / self.T0
            dgvdT = -self.cpv * (logT - np.log(self.T0)) - self.cpv + self.Rv * (logpv - np.log(self.p0)) - self.Lv0 / self.T0
            dgvdpv = self.Rv * T / pv

            dgldT = -self.cl * (logT - np.log(self.T0)) - self.cl

            dgvdqv = dgvdT * dTdqv + dgvdpv * dpvdqv
            dgldqv = dgldT * dTdqv

            grad = dgvdqv - dgldqv

            val = -self.cpv * T * (logT - np.log(self.T0)) + self.Rv * T * (logpv - np.log(self.p0)) + self.Lv0 * (1 - T / self.T0)
            val -= -self.cl * T * (logT - np.log(self.T0))

            # val = self.gibbs_vapour(T, pv) - self.gibbs_liquid(T)

            qv = qv - (val / grad)
            rel_update = abs((val / grad) / qv).max()
            qv = np.maximum(qv, 1e-7)
            # if rel_update < 1e-10:
            #     break

        rel_update = abs((val / grad) / qv)
        # print('Max relative last update:', rel_update.max())
        if verbose:
            rel_update = abs((val / grad) / qv)
            print('Max relative last update:', rel_update.max())
            print('Max Gibbs error:', abs(val).max())

        return np.minimum(qv, qw)