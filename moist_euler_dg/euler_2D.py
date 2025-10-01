import numpy as np
from moist_euler_dg import utils
import time
import os


class Euler2D():

    nvars = 4

    def __init__(self, xmap, zmap, order, nx, g, cfl=0.5, a=0, nz=None, upwind=True, nprocx=1, top_bc='wall', forcing=None):

        self.order = order
        self.g = g
        self.cfl = cfl
        self.a = a
        self.ah = 0.0
        self.f = 0.0
        self.upwind = upwind
        self.nprocx = nprocx
        self.buoyancy_relax = 1.0

        if self.nprocx > 1:
            from mpi4py import MPI

            self.comm = MPI.COMM_WORLD
            self.rank = self.comm.Get_rank()

        else:
            self.comm = None
            self.rank = 0

        self.forcing = forcing

        self.cp = 1_005.0
        self.cv = 718.0
        self.R = self.cp - self.cv
        self.p0 = 100_000.0
        self.gamma = self.cp / self.cv
        self.is_x_periodic = True
        self.top_bc = top_bc

        # interface boundary indices
        self.ip_horz_int = (slice(1, None), slice(None), 0, slice(None))
        self.im_horz_int = (slice(0, -1), slice(None), -1, slice(None))
        self.ip_vert_int = (slice(None), slice(1, None), slice(None), 0)
        self.im_vert_int = (slice(None), slice(0, -1), slice(None), -1)
        # domain boundary interfaces
        self.ip_horz_ext = (0, slice(None), 0, slice(None))
        self.im_horz_ext = (-1, slice(None), -1, slice(None))
        self.ip_vert_ext = (slice(None), 0, slice(None), 0)
        self.im_vert_ext = (slice(None), -1, slice(None), -1)

        if nz is None:
            nz = nx

        self.nx = nx
        self.nz = nz

        # function space
        xis_, self.weights_x = utils.gll(order, iterative=True)
        zetas_, self.weights_z = utils.gll(order, iterative=True)
        xis = 0 * zetas_[None, :] + xis_[:, None]
        zetas = zetas_[None, :] + 0 * xis_[:, None]

        self.D = utils.lagrange1st(order, xis_).transpose()
        self.weights2D = self.weights_z[None, :] * self.weights_x[:, None]

        if nprocx > 1:
            self.is_x_periodic = False
            rank = self.rank
            size = self.comm.Get_size()

            assert self.nx % nprocx == 0
            assert nprocx == size

            self.nx = self.nx // nprocx
            xi_start = rank / size
            xi_end = (rank + 1) / size
        else:
            xi_start = 0
            xi_end = 1

        # create cells
        xs_ = np.linspace(xi_start, xi_end, self.nx + 1)
        zs_ = np.linspace(0, 1, nz + 1)
        dx = np.diff(xs_).mean()
        dz = np.diff(zs_).mean()

        xs = 0 * zs_[None, :] + xs_[:, None]
        zs = zs_[None, :] + 0 * xs_[:, None]

        # self.xis = xs[:-1, :-1, None, None] + (xis[None, None, ...] + 1) * 0.5 * dx
        # self.zetas = zs[:-1, :-1, None, None] + (zetas[None, None, ...] + 1) * 0.5 * dz

        self.xis = xs[:-1, :-1, None, None] + (xis[None, None, :] + 1) * 0.5 * dx
        self.zetas = zs[:-1, :-1, None, None] + (zetas[None, None, :] + 1) * 0.5 * dz

        self.xs = xmap(self.xis, self.zetas)
        self.zs = zmap(self.xis, self.zetas)

        self.dx = (abs(self.xs[:, :, 1:] - self.xs[:, :, :1]).min())
        self.dz = (abs(self.zs[:, :, :, 1:] - self.zs[:, :, :, :-1]).min())

        self.cdt = self.cfl * min(self.dx, self.dz)
        self.time = 0

        self.state = np.zeros(self.nvars * self.xs.size)
        self.state_unflat = self.state.reshape((self.nvars,) + self.xs.shape)
        self.private_working_arrays = [np.zeros_like(self.state) for _ in range(3)]

        self.cell_horz_stride = self.nx
        self.horz_stride = self.order + 1

        self.right_boundary = np.zeros((self.nvars, self.nz, self.order + 1))
        self.left_boundary = np.zeros_like(self.right_boundary)
        self.right_boundary_send = np.zeros_like(self.right_boundary)
        self.left_boundary_send = np.zeros_like(self.right_boundary)

        self.top_boundary = np.zeros((self.nvars, self.nx, self.order + 1))

        self.dxdxi = self.project_H1(self.ddxi(self.xs))
        self.dxdzeta = self.project_H1(self.ddzeta(self.xs))
        self.dzdxi = self.project_H1(self.ddxi(self.zs))
        self.dzdzeta = self.project_H1(self.ddzeta(self.zs))

        if self.nprocx > 1:
            # make metric terms continuous across elements
            self.state_unflat[0] = self.dxdxi
            self.state_unflat[1] = self.dxdzeta
            self.state_unflat[2] = self.dzdxi
            self.state_unflat[3] = self.dzdzeta

            req1, req2 = self.fill_boundaries(self.state)
            req1.wait()
            req2.wait()

            self.dxdxi[self.ip_horz_ext] = 0.5 * (self.dxdxi[self.ip_horz_ext] + self.left_boundary[0])
            self.dxdzeta[self.ip_horz_ext] = 0.5 * (self.dxdzeta[self.ip_horz_ext] + self.left_boundary[1])
            self.dzdxi[self.ip_horz_ext] = 0.5 * (self.dzdxi[self.ip_horz_ext] + self.left_boundary[2])
            self.dzdzeta[self.ip_horz_ext] = 0.5 * (self.dzdzeta[self.ip_horz_ext] + self.left_boundary[3])

            self.dxdxi[self.im_horz_ext] = 0.5 * (self.dxdxi[self.im_horz_ext] + self.right_boundary[0])
            self.dxdzeta[self.im_horz_ext] = 0.5 * (self.dxdzeta[self.im_horz_ext] + self.right_boundary[1])
            self.dzdxi[self.im_horz_ext] = 0.5 * (self.dzdxi[self.im_horz_ext] + self.right_boundary[2])
            self.dzdzeta[self.im_horz_ext] = 0.5 * (self.dzdzeta[self.im_horz_ext] + self.right_boundary[3])

            self.state[:] = 0
            self.comm.Barrier()

        self.compute_metric_terms()

        self.dtype = self.state.dtype
        self.shape = (self.state.size, self.state.size)
        self.scale = 0.0


        self.mpi_send_time = 0.0
        self.mpi_recv_time = 0.0
        self.solve_time = 0.0
        self.bdry_time = 0.0
        self.matrix_assemble_time = 0.0

        self.u_grav = self.dzdxi
        self.w_grav = self.dzdzeta

    def compute_metric_terms(self):
        self.J = self.dxdxi * self.dzdzeta - self.dxdzeta * self.dzdxi

        self.dxidx = self.dzdzeta / self.J
        self.dxidz = -self.dxdzeta / self.J

        self.dzetadx = -self.dzdxi / self.J
        self.dzetadz = self.dxdxi / self.J

        self.grad_xi_2 = self.dxidx * self.dxidx + self.dxidz * self.dxidz
        self.grad_zeta_2 = self.dzetadx * self.dzetadx + self.dzetadz * self.dzetadz
        self.grad_xi_dot_zeta = self.dxidx * self.dzetadx + self.dxidz * self.dzetadz
        self.norm_grad_xi = np.sqrt(self.grad_xi_2)
        self.norm_grad_zeta = np.sqrt(self.grad_zeta_2)

        self.drdxi_2 = self.dxdxi * self.dxdxi + self.dzdxi * self.dzdxi
        self.drdzeta_2 = self.dxdzeta * self.dxdzeta + self.dzdzeta * self.dzdzeta
        self.dr_xi_dot_zeta = self.dxdxi * self.dxdzeta + self.dzdxi * self.dzdzeta

        self.norm_drdxi = np.sqrt(self.drdxi_2)
        self.norm_drdzeta = np.sqrt(self.drdzeta_2)

    def phys_to_contra(self, u_in, w_in, idx=slice(None)):
        u_out = u_in * self.dxidx[idx] + w_in * self.dxidz[idx]
        w_out = u_in * self.dzetadx[idx] + w_in * self.dzetadz[idx]

        return u_out, w_out

    def contra_to_phys(self, u_in, w_in, idx=slice(None)):
        u_out = u_in * self.dxdxi[idx] + w_in * self.dxdzeta[idx]
        w_out = u_in * self.dzdxi[idx] + w_in * self.dzdzeta[idx]

        return u_out, w_out

    def phys_to_cov(self, u_in, w_in, idx=slice(None)):
        u_out = u_in * self.dxdxi[idx] + w_in * self.dzdxi[idx]
        w_out = u_in * self.dxdzeta[idx] + w_in * self.dzdzeta[idx]

        return u_out, w_out

    def cov_to_phy(self, u_in, w_in, idx=slice(None)):
        u_out = u_in * self.dxidx[idx] + w_in * self.dzetadx[idx]
        w_out = u_in * self.dxidz[idx] + w_in * self.dzetadz[idx]

        return u_out, w_out

    def get_boundary_data(self, state, idx):
        # extract boundary data
        shape = (-1,) + self.xs.shape
        state_bdry = state.reshape(shape)[(slice(None),) + idx]
        return state_bdry

    def fill_right_boundary(self, state):
        state_m = self.get_boundary_data(state, self.im_horz_ext)
        if self.nprocx == 1:
            self.left_boundary[:] = state_m
            return None
        else:
            self.right_boundary_send[:] = state_m
            rank = self.comm.Get_rank()
            self.comm.Isend(self.right_boundary_send, dest=(rank + 1) % self.nprocx, tag=2)
            req = self.comm.Irecv(self.right_boundary, source=(rank + 1) % self.nprocx, tag=1)
            return req

    def fill_left_boundary(self, state):
        state_p = self.get_boundary_data(state, self.ip_horz_ext)
        if self.nprocx == 1:
            self.right_boundary[:] = state_p
            return None
        else:
            self.left_boundary_send[:] = state_p
            rank = self.comm.Get_rank()
            self.comm.Isend(self.left_boundary_send, dest=(rank - 1) % self.nprocx, tag=1)
            req = self.comm.Irecv(self.left_boundary, source=(rank - 1) % self.nprocx, tag=2)
            return req

    def fill_boundaries(self, state):
        return self.fill_left_boundary(state), self.fill_right_boundary(state)

    def get_fluxes(self, u, w, h, s, idx=slice(None)):
        vel_norm = self.grad_xi_2[idx] * u ** 2 + 2 * self.grad_xi_dot_zeta[idx] * u * w + self.grad_zeta_2[idx] * w ** 2
        e, T, p, _ = self.get_thermodynamic_quantities(h, h * s)
        c_sound = np.sqrt(self.gamma * p / h)
        G = 0.5 * vel_norm + e - T * s
        Fx = h * (self.grad_xi_2[idx] * u + self.grad_xi_dot_zeta[idx] * w)
        Fz = h * (self.grad_xi_dot_zeta[idx] * u + self.grad_zeta_2[idx] * w)

        return G, c_sound, T, Fx, Fz

    def solve(self, state, dstatedt=None, verbose=False):
        if self.nprocx > 1:
            self.comm.Barrier()

        t0 = time.time()
        req1, req2 = self.fill_boundaries(state)
        self.mpi_send_time += time.time() - t0

        if dstatedt is None:
            dstatedt = np.empty_like(state)
        dstatedt[:] = 0.0

        t0 = time.time()
        self._solve(state, dstatedt)
        self.solve_time += time.time() - t0

        # horizontal edge boundaries self.comm here!
        t0 = time.time()
        if req1 is not None:
            req1.wait()
        if req2 is not None:
            req2.wait()
        self.mpi_recv_time += time.time() - t0

        t0 = time.time()
        self._solve_horz_boundaries(state, dstatedt)
        self.bdry_time += time.time() - t0

        return dstatedt

    def _solve_horz_boundaries(self, state, dstatedt):


        ip = self.ip_horz_ext
        im = self.im_horz_ext
        # left boundary
        state_p, dstatedt_p = self.get_boundary_data(state, ip), self.get_boundary_data(dstatedt, ip)
        dstatedt_m = np.zeros_like(dstatedt_p)
        self.solve_boundaries(state_p, self.left_boundary, dstatedt_p, dstatedt_m, 'x', idx=ip)

        # right boundary
        state_m, dstatedt_m = self.get_boundary_data(state, im), self.get_boundary_data(dstatedt, im)
        dstatedt_p = np.zeros_like(dstatedt_m)
        self.solve_boundaries(self.right_boundary, state_m, dstatedt_p, dstatedt_m, 'x', idx=im)

    def _solve(self, state, dstatedt):

        u, w, h, s = self.get_vars(state)
        dudt, dwdt, dhdt, dsdt = self.get_vars(dstatedt)

        G, c_sound, T, Fx, Fz = self.get_fluxes(u, w, h, s)
        Sx = s * Fx
        Sz = s * Fz

        dsdx = self.ddxi(s)
        dTdx = self.ddxi(T)
        dsTdx = self.ddxi(s * T)

        dsdz = self.ddzeta(s)
        dTdz = self.ddzeta(T)
        dsTdz = self.ddzeta(s * T)

        dudt -= self.ddxi(G) + 0.5 * (s * dTdx + dsTdx - T * dsdx)
        dwdt -= self.ddzeta(G) + 0.5 * (s * dTdz + dsTdz - T * dsdz)

        divF = self.ddxi(self.J * Fx) + self.ddzeta(self.J * Fz)
        divF /= self.J
        divS = self.ddxi(self.J * Sx) + self.ddzeta(self.J * Sz)
        divS /= self.J

        dhdt -= divF
        dsdt -= 0.5 * (divS - s * divF + Fx * dsdx + Fz * dsdz) / h

        dudt -= self.g * self.u_grav
        dwdt -= self.g * self.w_grav

        # dudt -= vort * u_perp
        # dwdt -= vort * w_perp
        u1, u3 = Fx / h, Fz / h

        dudz = self.ddzeta(u)
        dwdx = self.ddxi(w)
        dudt -= (u3 * dudz - u3 * dwdx)
        dwdt -= (u1 * dwdx - u1 * dudz)

        # bottom wall BCs
        ip = self.ip_vert_ext
        dhdt[ip] += (0.0 - Fz[ip]) / self.weights_z[-1]
        normal_vel = Fz[ip] / (self.norm_grad_zeta[ip] * h[ip])
        diss = -2 * self.a * (c_sound[ip] + np.abs(normal_vel)) * normal_vel
        dwdt[ip] += diss / self.weights_z[-1]

        energy_diss = Fz[ip] * diss / self.weights_z[-1]
        dsdt[ip] -= energy_diss / (h[ip] * T[ip])

        # top wall BCs
        if self.top_bc == 'wall':
            im = self.im_vert_ext
            dhdt[im] += -(0.0 - Fz[im]) / self.weights_z[-1]
            normal_vel = Fz[im] / (self.norm_grad_zeta[im] * h[im])
            diss = -2 * self.a * (c_sound[im] + np.abs(normal_vel)) * normal_vel
            dwdt[im] += diss / self.weights_z[-1]

            energy_diss = Fz[im] * diss / self.weights_z[-1]
            dsdt[im] -= energy_diss / (h[im] * T[im])

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

        up, wp, hp, sp = (state_p[i] for i in range(self.nvars))
        um, wm, hm, sm = (state_m[i] for i in range(self.nvars))

        dudtp, dwdtp, dhdtp, dsdtp = (dstatedt_p[i] for i in range(self.nvars))
        dudtm, dwdtm, dhdtm, dsdtm = (dstatedt_m[i] for i in range(self.nvars))

        # calculate fluxes
        Gp, cp, Tp, Fxp, Fzp = self.get_fluxes(up, wp, hp, sp, idx)
        Gm, cm, Tm, Fxm, Fzm = self.get_fluxes(um, wm, hm, sm, idx)

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

        J = self.J[idx]
        dr_xi_dot_zeta = self.dr_xi_dot_zeta[idx]
        drdxi_2 = self.drdxi_2[idx]
        drdzeta_2 = self.drdzeta_2[idx]

        normal_vel_p = Fp / (hp * norm_contra)
        normal_vel_m = Fm / (hm * norm_contra)

        c_adv = np.abs(0.5 * (normal_vel_p + normal_vel_m))
        c_snd = 0.5 * (cp + cm)

        F_num_flux = 0.5 * (Fp + Fm) - self.ah * (c_adv + c_snd) * (hp - hm) * norm_contra

        if self.upwind:
            shat = (F_num_flux >= 0) * sm + (F_num_flux < 0) * sp
        else:
            shat = 0.5 * (sm + sp)

        fluxp = Gp
        fluxm = Gm
        num_flux = 0.5 * (fluxp + fluxm)
        dveldtp += (num_flux - fluxp) / self.weights_z[-1]
        dveldtm += -(num_flux - fluxm) / self.weights_z[-1]

        fluxp = Tp
        fluxm = Tm
        num_flux = 0.5 * (fluxp + fluxm)
        dveldtp += shat * (num_flux - fluxp) / self.weights_z[-1]
        dveldtm += -shat * (num_flux - fluxm) / self.weights_z[-1]

        fluxp = Fp
        fluxm = Fm
        dhdtp += (F_num_flux - fluxp) / self.weights_z[-1]
        dhdtm += -(F_num_flux - fluxm) / self.weights_z[-1]

        dsdtp += (F_num_flux / hp) * (shat - sp) / self.weights_z[-1]
        dsdtm += -(F_num_flux / hm) * (shat - sm) / self.weights_z[-1]

        # dissipation from jump in normal direction
        normal_jump = (Fp - Fm) / (0.5 * (hp + hm) * norm_contra)
        diss = -self.a * (c_adv + c_snd) * normal_jump

        dveldtp += diss / self.weights_z[-1]
        dveldtm += -diss / self.weights_z[-1]
        
        energy_diss = (Fp - Fm) * diss / self.weights_z[-1]

        # dissipation from jump in tangent direction
        # tang_jump = tan_velp - tan_velm
        # diss = -self.a * (c_adv) * tang_jump

        # dtan_veldtp += diss * norm_contra / self.weights_z[-1]
        # dtan_veldtm += -diss * norm_contra / self.weights_z[-1]

        # if direction == 'z':
        #     energy_diss += (Fxp - Fxm) * diss * norm_contra / self.weights_z[-1]
        # else:
        #     energy_diss += (Fzp - Fzm) * diss * norm_contra / self.weights_z[-1]
        
        # dsdtp -= 0.5 * energy_diss / (hp * Tp)
        # dsdtm -= 0.5 * energy_diss / (hm * Tm)

        # vorticity terms
        # dudt -= (u3 * dudz - u3 * dwdx)
        # dwdt -= (u1 * dwdx - u1 * dudz)
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
        ie = self.h * self.get_thermodynamic_quantities(self.h, self.hs)[3]
        energy = pe + ke + ie
        return self.integrate(energy)

    def get_thermodynamic_quantities(self, h, hs):
        s = hs / h

        # s = log(p) - gamma * log(h)
        # p = exp(s + gamma * log(h)) = exp(s) * h**gamma
        # T = p / (R * h)
        # u = p / (h * (gamma - 1))

        p = np.exp(s / self.cv) * h ** self.gamma
        T = p / (self.R * h)
        u = p / (h * (self.gamma - 1))
        e = u + (p / h)

        return e, T, p, u

    def entropy_2_potential_temperature(self, s):
        return np.exp((s - (self.cp * np.log(self.R) + self.R * np.log(1 / self.p0))) / self.cp)

    def potential_temperature_2_entropy_2(self, pot_temp):
        return self.cp * np.log(pot_temp) + self.cp * np.log(self.R) + self.R * np.log(1 / self.p0)

    def get_vars(self, state, reshape=True):
        assert state.size % self.nvars == 0
        sz = state.size // self.nvars

        out = tuple(state[i * sz:(i + 1) * sz] for i in range(self.nvars))
        if reshape:
            out = tuple(arr.reshape(self.xs.shape) for arr in out)

        return out

    def get_vars_bdry(self, state):
        sz = state.size // self.nvars
        shape = (-1, self.order + 1)

        out = [state[i * sz: (i + 1) * sz].reshape(shape) for i in range(self.nvars)]

        return out

    @property
    def u(self):
        u_cov, w_cov = self.get_vars(self.state)[0], self.get_vars(self.state)[1]
        u, w = self.cov_to_phy(u_cov, w_cov)
        return u

    @property
    def w(self):
        u_cov, w_cov = self.get_vars(self.state)[0], self.get_vars(self.state)[1]
        u, w = self.cov_to_phy(u_cov, w_cov)
        return w

    @property
    def h(self):
        return self.get_vars(self.state)[2]

    @property
    def s(self):
        return self.get_vars(self.state)[3]

    @property
    def hs(self):
        return self.s * self.h

    @property
    def hb(self):
        e, T, p, u = self.get_thermodynamic_quantities(self.h, self.hs)
        return self.h * T * (self.p0 / p) ** (self.R / self.cp)

    @property
    def potential_temp(self):
        _, _, p, _ = self.get_thermodynamic_quantities(self.h, self.hs)
        out = (p / self.p0) ** (self.cv / self.cp)
        out *= self.p0 / self.R
        out /= self.h
        return out

    def set_initial_condition(self, *vars_in):
        u_, w_ = vars_in[0], vars_in[1]
        u_, w_ = self.phys_to_cov(u_, w_)

        vars = self.get_vars(self.state)
        u, w = vars[0], vars[1]
        u[:] = u_
        w[:] = w_
        for i in range(2, len(vars_in)):
            vars[i][:] = vars_in[i]

    def get_dt(self):
        c = 340.0 * np.ones_like(self.h)
        return self.cdt / c.max()

    def time_step(self, dt=None):

        if dt is None:
            dt = self.get_dt()

        k = self.private_working_arrays[1]
        u_tmp = self.private_working_arrays[2]

        self.solve(self.state, dstatedt=k)

        u_tmp[:] = self.state + 0.5 * dt * k
        self.solve(u_tmp, dstatedt=k)

        u_tmp[:] = u_tmp[:] + 0.5 * dt * k
        self.solve(u_tmp, dstatedt=k)

        u_tmp[:] = (2 / 3) * self.state + (1 / 3) * u_tmp[:] + (1 / 6) * dt * k
        self.solve(u_tmp, dstatedt=k)

        self.state[:] = u_tmp + 0.5 * dt * k

        self.time += dt

    def plot_solution(self, ax, vmin=None, vmax=None, plot_func=None, dim=3, cmap='nipy_spectral', levels=1000, km=False):

        def _reshape(arr):
            return arr.swapaxes(1, 2).reshape(self.h.shape[0] * self.h.shape[2], -1)

        if plot_func is None:
            z_plot = self.h
        else:
            z_plot = plot_func(self)

        x_plot = _reshape(self.xs)
        y_plot = _reshape(self.zs)
        z_plot = _reshape(z_plot)

        if km:
            x_plot = x_plot / 1000
            y_plot = y_plot / 1000

        if dim == 3:
            return ax.plot_surface(x_plot, y_plot, z_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        elif dim == 2:
            return ax.contourf(x_plot, y_plot, z_plot, cmap=cmap, vmin=vmin, vmax=vmax, levels=levels)

    def plot_contours(self, ax, vmin=None, vmax=None, plot_func=None, levels=1000, km=False):

        def _reshape(arr):
            return arr.swapaxes(1, 2).reshape(self.h.shape[0] * self.h.shape[2], -1)

        if plot_func is None:
            z_plot = self.h
        else:
            z_plot = plot_func(self)

        x_plot = _reshape(self.xs)
        y_plot = _reshape(self.zs)
        z_plot = _reshape(z_plot)

        if km:
            x_plot = x_plot / 1000
            y_plot = y_plot / 1000

        return ax.contour(x_plot, y_plot, z_plot, vmin=vmin, vmax=vmax, levels=levels, colors='black', linewidths=0.5)

    def integrate(self, q):
        out = (self.J * self.weights2D[None, None] * q).sum()

        if self.nprocx > 1:
            from mpi4py import MPI
            out = np.array([out], 'd')
            out = self.comm.reduce(out, op=MPI.SUM)

            if out is not None:
                return out[0]
            else:
                return None
        else:
            return out

    def ddxi(self, arr):
        return np.einsum('ab,ecbd->ecad', self.D, arr)

    def ddzeta(self, arr):
        return np.einsum('ab,ecdb->ecda', self.D, arr)

    def ddz(self, arr):

        return (self.ddzeta(arr) * self.dzetadz) + (self.ddxi(arr) * self.dxidz)

    def project_H1(self, arr_in, arr_out=None):
        if arr_out is None:
            arr_out = np.empty_like(arr_in)

        arr_out[:] = arr_in

        ip = self.ip_vert_int
        im = self.im_vert_int
        arr_out[ip] = 0.5 * (arr_out[ip] + arr_out[im])
        arr_out[im] = arr_out[ip]

        ip = self.ip_horz_int
        im = self.im_horz_int
        arr_out[ip] = 0.5 * (arr_out[ip] + arr_out[im])
        arr_out[im] = arr_out[ip]

        if self.is_x_periodic:
            ip = self.ip_horz_ext
            im = self.im_horz_ext
            arr_out[ip] = 0.5 * (arr_out[ip] + arr_out[im])
            arr_out[im] = arr_out[ip]

        return arr_out

    def project_H1_vert(self, arr_in, arr_out=None):
        if arr_out is None:
            arr_out = np.empty_like(arr_in)

        arr_out[:] = arr_in
        ip = self.ip_vert_int
        im = self.im_vert_int
        arr_out[ip] = 0.5 * (arr_out[ip] + arr_out[im])
        arr_out[im] = arr_out[ip]

        return arr_out

    def get_filepath(self, data_dir, experiment_name, proc=None, time=None, nprocx=None, ext='npy'):

        if proc is None:
            proc = self.rank

        if time is None:
            time = self.time

        if nprocx is None:
            nprocx = self.nprocx  # might be more convenient to use size

        time = int(time)
        time_str = f'{(time // 3600)}H{(time % 3600) // 60}m{time % 60}s'

        if self.rank == 0:
            if not os.path.exists(data_dir):
                os.makedirs(data_dir)

        if nprocx > 1:
            fn = f"{data_dir}/{experiment_name}_nx_{self.nx * self.nprocx}_nz_{self.nz}_p{self.order}_upwind_{self.upwind}_part_{proc + 1}_of_{nprocx}_time_{time_str}.{ext}"
        else:
            fn = f"{data_dir}/{experiment_name}_nx_{self.nx * self.nprocx}_nz_{self.nz}_p{self.order}_upwind_{self.upwind}_time_{time_str}.{ext}"

        return fn

    def save(self, fn):
        np.save(fn, self.state)

        if self.nprocx > 1:
            self.comm.Barrier()

    def load(self, filepaths):
        if type(filepaths) is str:
            filepaths = [filepaths]

        assert self.nx % len(filepaths) == 0

        dnx = self.nx // len(filepaths)
        vars = self.get_vars(self.state)

        for i, filepath in enumerate(filepaths):
            i_start, i_stop = i * dnx, (i + 1) * dnx
            state_in = np.load(filepath)
            vars_in = self.get_vars(state_in, reshape=False)
            for var, var_in in zip(vars, vars_in):
                var[i_start:i_stop].ravel()[:] = var_in
