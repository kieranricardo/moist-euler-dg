import numpy as np
from moist_euler_dg.euler_2D import Euler2D


class NonEqEuler2D(Euler2D):

    nvars = 7

    def __init__(self, *args, **kwargs):
        Euler2D.__init__(self, *args, **kwargs)

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

        self.limit_water = True
        self.first_water_limit_time = None
        self.qw_min = 1e-10

    @property
    def qv(self):
        return self.get_vars(self.state)[4]

    @property
    def ql(self):
        return self.get_vars(self.state)[5]

    @property
    def qi(self):
        return self.get_vars(self.state)[6]

    def set_initial_condition(self, *vars_in):
        Euler2D.set_initial_condition(self, *vars_in)

        self.qv[:] = np.maximum(self.qw_min, self.qv)
        self.ql[:] = np.maximum(self.qw_min, self.ql)
        self.qi[:] = np.maximum(self.qw_min, self.qi)

    def time_step(self, dt=None):

        if dt is None:
            dt = self.get_dt()

        k = self.private_working_arrays[1]
        u_tmp = self.private_working_arrays[2]

        self.solve(self.state, dstatedt=k)
        if self.forcing is not None:
            self.forcing(self, self.state, k)

        u_tmp[:] = self.state + 0.5 * dt * k
        if self.limit_water:
            self.check_positivity(u_tmp)
        self.solve(u_tmp, dstatedt=k)
        if self.forcing is not None:
            self.forcing(self, u_tmp, k)

        u_tmp[:] = u_tmp[:] + 0.5 * dt * k
        if self.limit_water:
            self.check_positivity(u_tmp)
        self.solve(u_tmp, dstatedt=k)
        if self.forcing is not None:
            self.forcing(self, u_tmp, k)

        u_tmp[:] = (2 / 3) * self.state + (1 / 3) * u_tmp[:] + (1 / 6) * dt * k
        if self.limit_water:
            self.check_positivity(u_tmp)
        self.solve(u_tmp, dstatedt=k)
        if self.forcing is not None:
            self.forcing(self, u_tmp, k)

        self.state[:] = u_tmp + 0.5 * dt * k
        if self.limit_water:
            self.check_positivity(self.state)

        self.time += dt

    def forcing_only_time_step(self, dt=None):

        if dt is None:
            dt = self.get_dt()

        k = self.private_working_arrays[1]
        u_tmp = self.private_working_arrays[2]

        if self.forcing is not None:
            k[:] = 0.0
            self.forcing(self, self.state, k)

        u_tmp[:] = self.state + 0.5 * dt * k
        if self.forcing is not None:
            k[:] = 0.0
            self.forcing(self, u_tmp, k)

        u_tmp[:] = u_tmp[:] + 0.5 * dt * k
        if self.forcing is not None:
            k[:] = 0.0
            self.forcing(self, u_tmp, k)

        u_tmp[:] = (2 / 3) * self.state + (1 / 3) * u_tmp[:] + (1 / 6) * dt * k
        if self.forcing is not None:
            k[:] = 0.0
            self.forcing(self, u_tmp, k)

        self.state[:] = u_tmp + 0.5 * dt * k

        self.time += dt

    def positivity_preserving_limiter(self, in_tnsr):
        cell_means = (in_tnsr * self.weights2D[None, None] * self.J).sum(axis=(2, 3)) / (self.weights2D[None, None] * self.J).sum(axis=(2, 3))
        cell_diffs = in_tnsr - cell_means[..., None, None]

        cell_mins = in_tnsr.min(axis=-1)
        cell_mins = cell_mins.min(axis=-1)
        diff_min = cell_mins - cell_means

        new_min = np.minimum(self.qw_min, cell_means)
        new_min = np.maximum(new_min, cell_mins)
        scale = (new_min - cell_means) / diff_min

        out_tnsr = cell_means[..., None, None] + scale[..., None, None] * cell_diffs

        # cell_means1 = (out_tnsr * self.weights2D[None, None] * self.J).sum(axis=(2, 3)) / (self.weights2D[None, None] * self.J).sum(axis=(2, 3))

        return out_tnsr, cell_means

    def check_positivity(self, state):
        u, v, h, s, qv, ql, qi = self.get_vars(state)

        names = ('qv', 'ql', 'qi')
        for q, name in zip((qv, ql, qi), names):
            hq_limited, hq_cell_means = self.positivity_preserving_limiter(h * q)
            q[:] = hq_limited / h

            if (q <= 0).any():
                if self.first_water_limit_time is None:
                    self.first_water_limit_time = self.time

            if (hq_cell_means <= 0).any():
                print("Negative water cell mean detected")
                raise RuntimeError("Negative water cell mean detected")
                # print("x-coords:", self.xs[state['hqw'] <= 0], "\n")
                # print("y-coords:", self.ys[state['hqw'] <= 0], "\n")

            if (q <= 0).any():
                print(name)
                print("Negative water mass - limiting failed :( ")
                print('hqw_limited min:', hq_limited.min())
                print('h min:', h.min())
                raise RuntimeError("Negative water mass - limiting failed :( ")
                # print("x-coords:", self.xs[state['hqw'] <= 0], "\n")
                # print("y-coords:", self.ys[state['hqw'] <= 0], "\n")

    def get_fluxes(self, u, w, h, s, qv, ql, qi, T, p, ie, mu_v, mu_l, mu_i, idx=slice(None)):

        vel_norm = self.grad_xi_2[idx] * u ** 2 + 2 * self.grad_xi_dot_zeta[idx] * u * w + self.grad_zeta_2[idx] * w ** 2
        e = (ie + p) / h
        c_sound = np.sqrt(self.gamma * p / h)
        G = 0.5 * vel_norm + e - T * s - (mu_v * qv + mu_l * ql + mu_i * qi)
        Fx = h * (self.grad_xi_2[idx] * u + self.grad_xi_dot_zeta[idx] * w)
        Fz = h * (self.grad_xi_dot_zeta[idx] * u + self.grad_zeta_2[idx] * w)

        return G, c_sound, Fx, Fz

    def _solve(self, state, dstatedt):

        u, w, h, s, qv, ql, qi = self.get_vars(state)
        dudt, dwdt, dhdt, dsdt, dqvdt, dqldt, dqidt = self.get_vars(dstatedt)

        _, T, p, ie, mu_v, mu_l, mu_i = self.get_thermodynamic_quantities(h, s, qv, ql, qi)

        G, c_sound, Fx, Fz = self.get_fluxes(u, w, h, s, qv, ql, qi, T, p, ie, mu_v, mu_l, mu_i)

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
        tracers = [
            (dsdt, s, T),
            (dqvdt, qv, mu_v),
            (dqldt, ql, mu_l),
            (dqidt, qi, mu_i),
        ]

        for (dadt, a, b) in tracers:
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
        diss = -2 * self.b * (c_sound[ip] + np.abs(normal_vel)) * normal_vel
        dwdt[ip] += diss / self.weights_z[-1]

        # energy_diss = Fz[ip] * diss / self.weights_z[-1]
        # dsdt[ip] -= energy_diss / (h[ip] * T[ip])

        # top wall BCs
        if self.top_bc == 'wall':
            im = self.im_vert_ext
            dhdt[im] += -(0.0 - Fz[im]) / self.weights_z[-1]
            normal_vel = Fz[im] / (self.norm_grad_zeta[im] * h[im])
            diss = -2 * self.b * (c_sound[im] + np.abs(normal_vel)) * normal_vel
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

        up, wp, hp, sp, qvp, qlp, qip = (state_p[i] for i in range(self.nvars))
        um, wm, hm, sm, qvm, qlm, qim = (state_m[i] for i in range(self.nvars))

        _, Tp, pp, iep, mu_vp, mu_lp, mu_ip = self.get_thermodynamic_quantities(hp, sp, qvp, qlp, qip)
        _, Tm, pm, iem, mu_vm, mu_lm, mu_im = self.get_thermodynamic_quantities(hm, sm, qvm, qlm, qim)

        dudtp, dwdtp, dhdtp, dsdtp, dqvdtp, dqldtp, dqidtp = (dstatedt_p[i] for i in range(self.nvars))
        dudtm, dwdtm, dhdtm, dsdtm, dqvdtm, dqldtm, dqidtm = (dstatedt_m[i] for i in range(self.nvars))

        # calculate fluxes
        Gp, cp, Fxp, Fzp = self.get_fluxes(up, wp, hp, sp, qvp, qlp, qip, Tp, pp, iep, mu_vp, mu_lp, mu_ip, idx)
        Gm, cm, Fxm, Fzm = self.get_fluxes(um, wm, hm, sm, qvm, qlm, qim, Tm, pm, iem, mu_vm, mu_lm, mu_im, idx)

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

        F_num_flux = 0.5 * (Fp + Fm) - self.a * (c_adv + c_snd) * (hp - hm) * norm_contra

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
        tracer_vars = [
            (dsdtp, sp, Tp, dsdtm, sm, Tm),
            (dqvdtp, qvp, mu_vp, dqvdtm, qvm, mu_vm),
            (dqldtp, qlp, mu_lp, dqldtm, qlm, mu_lm),
            (dqidtp, qip, mu_ip, dqidtm, qim, mu_im)
        ]
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

    def energy(self):
        pe = self.h * self.g * self.zs
        ke = 0.5 * self.h * (self.u ** 2 + self.w ** 2)

        _, T, p, ie, mu_v, mu_l, mu_i = self.get_thermodynamic_quantities(self.h, self.s, self.qv, self.ql, self.qi)
        energy = pe + ke + ie
        return self.integrate(energy)

    def entropy_vapour(self, T, qv, density, np=np):
        # s = cvv * log(T / T0) - Rv * log(h / h0) + cpv + (Ls0 / T0)
        # c0 = cpv + (Ls0 / T0) - cvv * logT0 + Rv * log(h0)
        return self.cvv * np.log(T) - self.Rv * np.log(qv * density) + self.c0

    def entropy_liquid(self, T, np=np):
        return self.cl * np.log(T) + self.c1

    def entropy_ice(self, T, np=np):
        return self.ci * np.log(T) + self.c2

    def entropy_air(self, T, qd, density, np=np):
        return self.cvd * np.log(T) - self.Rd * np.log(qd * density * self.Rd)

    def gibbs_vapour(self, T, qv, density, np=np):
        # c0 = self.cpv + (self.Ls0 / self.T0) - self.cvv * self.logT0 + self.Rv * np.log(self.rho0)
        # s = cvv * log(T / T0) - Rv * log(h / h0) + cpv + (Ls0 / T0)
        # u = cvv * T + Ls0
        # p = h * Rv * T
        # gv = u + (p / h) - s * T
        # gv = - T * cvv * log(T / T0) + T * Rv * log(h / h0) - Ls0 * (1 - (Ls0 / T0))
        return -self.cvv * T * np.log(T / self.T0) + self.Rv * T * np.log(qv * density / self.rho0) + self.Ls0 * (1 - T / self.T0)

    def gibbs_liquid(self, T, np=np):
        # u = cl * T + Lf0
        # s = cl * log(T) + c1
        # u = exp((s - c1) / cl)
        # gl = u - s  * T
        # gl = cl * T + Lf0 - T * cl * log(T) - T * (cl + (Lf0 / T0) - cl * logT0)
        # gl = -T * cl * log(T / T0) + Lf0 * (1 - Lf0 / T0)
        return -self.cl * T * np.log(T / self.T0) + self.Lf0 * (1 - T / self.T0)

    def gibbs_ice(self, T, np=np):
        return -self.ci * T * np.log(T / self.T0)

    def gibbs_air(self, T, qd, density):
        # s = cvd * log(T) - Rd * log(h * Rd)
        # u = cvd * T
        # p = h * Rv * T
        # gd = u + (p / h) - s * T
        # gd = cpd * T - T * cvd * log(T) + Rd * log(h * Rd)
        h = qd * density
        return self.cpd * T - T * self.cvd * np.log(T) + self.Rd * T * np.log(h * self.Rd)

    def get_thermodynamic_quantities(self, density, entropy, qv, ql, qi):
        qw = qv + ql + qi
        qd = 1 - qw

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

        mu_v = chemical_potential_v - chemical_potential_d
        mu_l = chemical_potential_l - chemical_potential_d
        mu_i = chemical_potential_i - chemical_potential_d


        return enthalpy, T, p, ie, mu_v, mu_l, mu_i

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


