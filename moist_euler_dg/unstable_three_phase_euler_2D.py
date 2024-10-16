import numpy as np
from moist_euler_dg.fortran_three_phase_euler_2D import FortranThreePhaseEuler2D


class UnstableThreePhaseEuler2D(FortranThreePhaseEuler2D):

    def __init__(self, *args, **kwargs):
        FortranThreePhaseEuler2D.__init__(self, *args, **kwargs)
        self.var_stable = False

    def _solve(self, state, dstatedt):

        u, w, h, s, q, T, mu, p, ie = self.get_vars(state)
        dudt, dwdt, dhdt, dsdt, dqdt, *_ = self.get_vars(dstatedt)

        G, c_sound, Fx, Fz = self.get_fluxes(u, w, h, s, q, T, mu, p, ie)

        # density evolutions
        divF = self.ddxi(self.J * Fx) + self.ddzeta(self.J * Fz)
        divF /= self.J
        dhdt -= divF

        # velocity evolution
        dudt -= self.ddxi(G) + self.g * self.u_grav
        dwdt -= self.ddzeta(G) + self.g * self.w_grav

        # dudt -= vort * u_perp
        # dwdt -= vort * w_perp
        u1, u3 = Fx / h, Fz / h

        dudz = self.ddzeta(u)
        dwdx = self.ddxi(w)
        dudt -= (u3 * dudz - u3 * dwdx)
        dwdt -= (u1 * dwdx - u1 * dudz)

        # handle tracers
        for (dadt, a, b) in [(dsdt, s, T), (dqdt, q, mu)]:
            dadx = self.ddxi(a)
            dbdx = self.ddxi(b)
            dabdx = self.ddxi(a * b)

            dadz = self.ddzeta(a)
            dbdz = self.ddzeta(b)
            dabdz = self.ddzeta(a * b)

            divA = self.ddxi(self.J * a * Fx) + self.ddzeta(self.J * a * Fz)
            divA /= self.J

            if self.var_stable:
                dudt -= 0.5 * (a * dbdx + dabdx - b * dadx)
                dwdt -= 0.5 * (a * dbdz + dabdz - b * dadz)
                dadt -= 0.5 * (divA - a * divF + Fx * dadx + Fz * dadz) / h

            else:
                dudt -= a * dbdx
                dwdt -= a * dbdz
                dadt -=(divA - a * divF) / h

        # bottom wall BCs
        ip = self.ip_vert_ext
        dhdt[ip] += (0.0 - Fz[ip]) / self.weights_z[-1]
        normal_vel = Fz[ip] / (self.norm_grad_zeta[ip] * h[ip])
        diss = -2 * 0.5 * (c_sound[ip] + np.abs(normal_vel)) * normal_vel
        dwdt[ip] += diss / self.weights_z[-1]

        # energy_diss = Fz[ip] * diss / self.weights_z[-1]
        # dsdt[ip] -= energy_diss / (h[ip] * T[ip])

        # top wall BCs
        if self.top_bc == 'wall':
            im = self.im_vert_ext
            dhdt[im] += -(0.0 - Fz[im]) / self.weights_z[-1]
            normal_vel = Fz[im] / (self.norm_grad_zeta[im] * h[im])
            diss = -2 * 0.5 * (c_sound[im] + np.abs(normal_vel)) * normal_vel
            dwdt[im] += diss / self.weights_z[-1]

            # energy_diss = Fz[im] * diss / self.weights_z[-1]
            # dsdt[im] -= energy_diss / (h[im] * T[im])
        else:
            raise NotImplementedError

        # vertical interior boundaries
        ip = self.ip_vert_int
        im = self.im_vert_int
        state_p, dstatedt_p = self.get_boundary_data(state, ip), self.get_boundary_data(dstatedt, ip)
        state_m, dstatedt_m = self.get_boundary_data(state, im), self.get_boundary_data(dstatedt, im)
        self.solve_boundaries(state_p, state_m, dstatedt_p, dstatedt_m, 'z', idx=ip)

        if self.nx > 1:
            # horizontal interior boundaries
            ip = self.ip_horz_int
            im = self.im_horz_int
            state_p, dstatedt_p = self.get_boundary_data(state, ip), self.get_boundary_data(dstatedt, ip)
            state_m, dstatedt_m = self.get_boundary_data(state, im), self.get_boundary_data(dstatedt, im)
            self.solve_boundaries(state_p, state_m, dstatedt_p, dstatedt_m, 'x', idx=ip)

        if self.forcing is not None:
            self.forcing(self, state, dstatedt)

        return dstatedt

    def solve_boundaries(self, state_p, state_m, dstatedt_p, dstatedt_m, direction, idx):

        up, wp, hp, sp, qp, Tp, mup, pp, iep = (state_p[i] for i in range(self.nvars))
        um, wm, hm, sm, qm, Tm, mum, pm, iem = (state_m[i] for i in range(self.nvars))

        dudtp, dwdtp, dhdtp, dsdtp, dqdtp, *_ = (dstatedt_p[i] for i in range(self.nvars))
        dudtm, dwdtm, dhdtm, dsdtm, dqdtm, *_ = (dstatedt_m[i] for i in range(self.nvars))

        # calculate fluxes
        Gp, cp, Fxp, Fzp = self.get_fluxes(up, wp, hp, sp, qp, Tp, mup, pp, iep, idx)
        Gm, cm, Fxm, Fzm = self.get_fluxes(um, wm, hm, sm, qm, Tm, mum, pm, iem, idx)

        if direction == 'z':
            norm_contra = self.norm_grad_zeta[idx]
            norm_cov = self.norm_drdxi[idx]
            Fp, Fm = Fzp, Fzm
            dveldtp, dveldtm = dwdtp, dwdtm
            dtan_veldtp, dtan_veldtm = dudtp, dudtm
            tan_velp, tan_velm = up, um
        else:
            norm_contra = self.norm_grad_xi[idx]
            norm_cov = self.norm_drdzeta[idx]
            Fp, Fm = Fxp, Fxm
            dveldtp, dveldtm = dudtp, dudtm
            dtan_veldtp, dtan_veldtm = dwdtp, dwdtm
            tan_velp, tan_velm = wp, wm

        normal_vel_p = Fp / (hp * norm_contra)
        normal_vel_m = Fm / (hm * norm_contra)
        # normal_vel_p = Fp / (0.5 * (hp + hm) * norm_contra)
        # normal_vel_m = Fm / (0.5 * (hp + hm) * norm_contra)

        c_adv = np.abs(0.5 * (normal_vel_p + normal_vel_m))
        c_snd = 0.5 * (cp + cm)

        F_num_flux = 0.5 * (Fp + Fm) #- self.a * (c_adv + c_snd) * (hp - hm) * norm_contra

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

            dadtp += (F_num_flux / hp) * (ahat - ap) / self.weights_z[-1]
            dadtm += -(F_num_flux / hm) * (ahat - am) / self.weights_z[-1]

        # dissipation from jump in normal direction
        normal_jump = normal_vel_p - normal_vel_m
        diss = -self.a * (c_adv + c_snd) * normal_jump

        dveldtp += diss / self.weights_z[-1]
        dveldtm += -diss / self.weights_z[-1]

        # energy_diss = (Fp - Fm) * diss / self.weights_z[-1]

        # dissipation from jump in tangent direction
        # tang_jump = (hp * tan_velp - hm * tan_velm) / (0.5 * (hp + hm))
        tang_jump = tan_velp - tan_velm
        diss = -self.a * (c_adv) * tang_jump

        dtan_veldtp += diss * norm_contra / self.weights_z[-1]
        dtan_veldtm += -diss * norm_contra / self.weights_z[-1]

        # if direction == 'z':
        #     energy_diss += (Fxp - Fxm) * diss * norm_contra / self.weights_z[-1]
        # else:
        #     energy_diss += (Fzp - Fzm) * diss * norm_contra / self.weights_z[-1]

        # dsdtp -= 0.5 * energy_diss / (hp * Tp)
        # dsdtm -= 0.5 * energy_diss / (hm * Tm)

        # vorticity terms
        u1p, u1m = Fxp / hp, Fxm / hm
        u3p, u3m = Fzp / hp, Fzm / hm
        if direction == 'z':
            fluxp = up
            fluxm = um
        else:
            fluxp = -wp
            fluxm = -wm

        num_flux = 0.5 * (fluxp + fluxm)
        dudtp += u3p * (num_flux - fluxp) / self.weights_z[-1]
        dudtm += -u3m * (num_flux - fluxm) / self.weights_z[-1]

        dwdtp += -u1p * (num_flux - fluxp) / self.weights_z[-1]
        dwdtm += u1m * (num_flux - fluxm) / self.weights_z[-1]

        return 0.0