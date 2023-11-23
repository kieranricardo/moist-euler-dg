import meshzoo
import torch
import numpy as np
from .utils import gll, lagrange1st
from matplotlib import pyplot as plt
from scipy.interpolate import lagrange


class DryEuler2D:

    def __init__(
            self, xrange, yrange, poly_order, nx, ny,
            g, eps, device='cpu', solution=None, a=0.0, dtype=np.float32,
            xperiodic=True, yperiodic=True, tau=0, angle=0.0, tau_func=lambda t, dt: t, passive=False, **kwargs,
    ):
        self.time = 0
        self.poly_order = poly_order
        self.u = None
        self.v = None
        self.h = None
        self.hs = None
        self.hb = None
        self.g = g
        self.f = 0
        self.eps = eps
        self.a = a
        self.tau = tau
        self.solution = solution
        self.dtype = dtype
        self.xperiodic = xperiodic
        self.yperiodic = yperiodic
        self.tau_func = tau_func
        self.passive = passive

        self.R = 287.0
        self.cp = 1_005.0
        self.cv = 718.0
        self.p0 = 100_000.0
        self.gamma = self.cp / self.cv

        [xs_1d, w_x] = gll(poly_order, iterative=True)
        [y_1d, w_y] = gll(poly_order, iterative=True)

        xs = np.linspace(xrange[0], xrange[1], nx)
        ys = np.linspace(yrange[0], yrange[1], ny)

        lx = np.mean(np.diff(xs))
        ly = np.mean(np.diff(ys))

        self.cdt = eps * min(lx, ly) / (2 * poly_order + 1)
        # self.cdt = eps * min(lx, ly) / poly_order

        points, cells = meshzoo.rectangle_quad(
            ys,
            xs,
        )

        cells = cells.reshape(len(ys) - 1, len(xs) - 1, 4)

        w_x, w_y = np.meshgrid(w_x, w_y)
        self.w_x = w_x[0][None, None, ...]
        self.w = w_x * w_y

        xs, ys = np.meshgrid(xs_1d, y_1d)

        xs = (1 + xs) * lx / 2
        ys = (1 + ys) * ly / 2

        self.xs = xs[None, None, ...] * np.ones(cells.shape[:2] + (1, 1)) + points[cells[..., 0]][..., 1][..., None, None]
        self.ys = ys[None, None, ...] * np.ones(cells.shape[:2] + (1, 1)) + points[cells[..., 0]][..., 0][..., None, None]

        self.xs = self.xs + np.sin(angle) * self.ys
        self.ys = np.cos(angle) * self.ys

        self.l1d = lagrange1st(poly_order, xs_1d)

        n = poly_order + 1

        # self.K = self.K.reshape((-1, n * n * 2)).transpose()
        self.device = torch.device(device)
        self.n = n
        self.w = torch.from_numpy(self.w.astype(self.dtype)).to(self.device)
        self.w_x = torch.from_numpy(self.w_x.astype(self.dtype)).to(self.device)
        self.nx = nx - 1
        self.ny = ny - 1

        self.ddxi = torch.from_numpy(np.zeros((n, n, n, n), dtype=self.dtype)).to(self.device)
        self.ddeta = torch.zeros((n, n, n, n), dtype=self.ddxi.dtype, device=self.device)

        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        self.ddxi[i, j, k, l] = self.l1d[l, j] * (k == i)
                        self.ddeta[i, j, k, l] = self.l1d[k, i] * (l == j)

        x_tnsr = torch.from_numpy(self.xs.astype(self.dtype)).to(self.device)
        self.y_tnsr = torch.from_numpy(self.ys.astype(self.dtype)).to(self.device)

        self.dxdxi = torch.einsum('fgcd,abcd->fgab', x_tnsr, self.ddxi)
        self.dxdeta = torch.einsum('fgcd,abcd->fgab', x_tnsr, self.ddeta)
        self.dydxi = torch.einsum('fgcd,abcd->fgab', self.y_tnsr, self.ddxi)
        self.dydeta = torch.einsum('fgcd,abcd->fgab', self.y_tnsr, self.ddeta)

        self.J = self.dxdxi * self.dydeta - self.dxdeta * self.dydxi
        self.Jx = torch.sqrt(self.dxdxi ** 2 + self.dydxi ** 2)
        self.Jy = torch.sqrt(self.dxdeta ** 2 + self.dydeta ** 2)

        self.dxidx = self.dydeta / self.J
        self.dxidy = -self.dxdeta / self.J

        self.detadx = -self.dydxi / self.J
        self.detady = self.dxdxi / self.J

        self.J_xi = torch.sqrt(self.dxidx ** 2 + self.dxidy ** 2)
        self.J_eta = torch.sqrt(self.detadx ** 2 + self.detady ** 2)

        self.eta_x = self.detadx
        self.eta_y = self.detady

        self.xi_x = self.dxidx
        self.xi_y = self.dxidy

        self.xi_x_left = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.ddxi.dtype, device=self.device)
        self.xi_x_right = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.ddxi.dtype, device=self.device)
        self.eta_x_up = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.ddxi.dtype, device=self.device)
        self.eta_x_down = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.ddxi.dtype, device=self.device)

        self.xi_y_left = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.ddxi.dtype, device=self.device)
        self.xi_y_right = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.ddxi.dtype, device=self.device)
        self.eta_y_up = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.ddxi.dtype, device=self.device)
        self.eta_y_down = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.ddxi.dtype, device=self.device)

        self.eta_x_up[:-1] = self.eta_x[:, :, 0, :] / self.J_eta[:, :, 0, :]
        self.eta_x_up[-1] = self.eta_x_up[0]

        self.eta_x_down[1:] = self.eta_x[:, :, -1, :] / self.J_eta[:, :, -1, :]
        self.eta_x_down[0] = self.eta_x_down[-1]

        self.eta_y_up[:-1] = self.eta_y[:, :, 0, :] / self.J_eta[:, :, 0, :]
        self.eta_y_up[-1] = self.eta_y_up[0]

        self.eta_y_down[1:] = self.eta_y[:, :, -1, :] / self.J_eta[:, :, -1, :]
        self.eta_y_down[0] =  self.eta_y_down[-1]

        self.xi_x_right[:, :-1] = self.xi_x[:, :, :, 0] / self.J_xi[:, :, :, 0]
        self.xi_x_right[:, -1] = self.xi_x_right[:, 0]

        self.xi_x_left[:, 1:] = self.xi_x[:, :, :, -1] / self.J_xi[:, :, :, -1]
        self.xi_x_left[:, 0] = self.xi_x_left[:, -1]

        self.xi_y_right[:, :-1] = self.xi_y[:, :, :, 0] / self.J_xi[:, :, :, 0]
        self.xi_y_right[:, -1] =  self.xi_y_right[:, 0]

        self.xi_y_left[:, 1:] = self.xi_y[:, :, :, -1] / self.J_xi[:, :, :, -1]
        self.xi_y_left[:, 0] = self.xi_y_left[:, -1]

        base_K_1 = torch.zeros((1, 1, n, n, n, n), dtype=self.ddxi.dtype, device=self.device)
        base_K_2 = torch.zeros((1, 1, n, n, n, n), dtype=self.ddxi.dtype, device=self.device)
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    for l in range(n):
                        base_K_1[0, 0, i, j, k, l] = self.w[k, l] * self.l1d[i, k] * (j == l)
                        base_K_2[0, 0, i, j, k, l] = self.w[k, l] * self.l1d[j, l] * (k == i)

        self.Ky = base_K_1 * self.detady[:, :, None, None, :, :] + base_K_2 * self.dxidy[:, :, None, None, :, :]
        self.Ky *= self.J[:, :, None, None, :, :]
        self.Kx = base_K_1 * self.detadx[:, :, None, None, :, :] + base_K_2 * self.dxidx[:, :, None, None, :, :]
        self.Kx *= self.J[:, :, None, None, :, :]

    def boundaries(self, u, v, h, b, t):

        self.u_up[:-1] = u[:, :, 0, :]
        self.u_down[1:] = u[:, :, -1, :]
        self.u_right[:, :-1] = u[:, :, :, 0]
        self.u_left[:, 1:] = u[:, :, :, -1]


        self.v_up[:-1] = v[:, :, 0, :]
        self.v_down[1:] = v[:, :, -1, :]
        self.v_right[:, :-1] = v[:, :, :, 0]
        self.v_left[:, 1:] = v[:, :, :, -1]

        self.h_up[:-1] = h[:, :, 0, :]
        self.h_down[1:] = h[:, :, -1, :]
        self.h_right[:, :-1] = h[:, :, :, 0]
        self.h_left[:, 1:] = h[:, :, :, -1]

        self.hs_up[:-1] = b[:, :, 0, :]
        self.hs_down[1:] = b[:, :, -1, :]
        self.hs_right[:, :-1] = b[:, :, :, 0]
        self.hs_left[:, 1:] = b[:, :, :, -1]

        # wall BC
        self.u_up[-1] = u[-1, :, -1, :]
        self.u_down[0] = u[0, :, 0, :]
        self.v_up[-1] = 0
        self.v_down[0] = 0
        self.h_up[-1] = h[-1, :, -1, :]
        self.h_down[0] = h[0, :, 0, :]
        self.hs_up[-1] = b[-1, :, -1, :]
        self.hs_down[0] = b[0, :, 0, :]

        # self.u_up[-1] = u[0, :, 0, :]
        # self.u_down[0] = u[-1, :, -1, :]
        # self.v_up[-1] = v[0, :, 0, :]
        # self.v_down[0] = v[-1, :, -1, :]
        # self.h_up[-1] = h[0, :, 0, :]
        # self.h_down[0] = h[-1, :, -1, :]
        # self.hs_up[-1] = b[0, :, 0, :]
        # self.hs_down[0] = b[-1, :, -1, :]

        self.u_right[:, -1] = u[:, 0, :, 0]
        self.u_left[:, 0] = u[:, -1, :, -1]
        self.v_right[:, -1] = v[:, 0, :, 0]
        self.v_left[:, 0] = v[:, -1, :, -1]
        self.h_right[:, -1] = h[:, 0, :, 0]
        self.h_left[:, 0] = h[:, -1, :, -1]
        self.hs_right[:, -1] = b[:, 0, :, 0]
        self.hs_left[:, 0] = b[:, -1, :, -1]

    def time_step(self, dt=None, order=3, forcing=None):
        if dt is None:
            speed = self.wave_speed(self.u, self.v, self.h, self.hs)
            dt = self.cdt / torch.max(speed).cpu().numpy()

        if order == 3:
            uk_1, vk_1, hk_1, hsk_1 = self.solve(self.u, self.v, self.h, self.hs, self.time, dt)

            # SSPRK3
            u_1 = self.u + dt * uk_1
            v_1 = self.v + dt * vk_1
            h_1 = self.h + dt * hk_1
            hs_1 = self.hs + dt * hsk_1
            if (hs_1 < 0).any() and not self.passive:
                print(
                    'Entropy-enstrophy:',
                    self.integrate(self.energy()).numpy(),
                    self.integrate(self.entropy()).numpy()
                )
                print('b1 crash!', self.time, self.h.min(), (self.hs / self.h).min())
                raise ValueError

            uk_2, vk_2, hk_2, hsk_2 = self.solve(u_1, v_1, h_1, hs_1, self.time + dt, dt)

            u_2 = 0.75 * self.u + 0.25 * (u_1 + uk_2 * dt)
            v_2 = 0.75 * self.v + 0.25 * (v_1 + vk_2 * dt)
            h_2 = 0.75 * self.h + 0.25 * (h_1 + hk_2 * dt)
            hs_2 = 0.75 * self.hs + 0.25 * (hs_1 + hsk_2 * dt)

            uk_3, vk_3, hk_3, hsk_3 = self.solve(u_2, v_2, h_2, hs_2, self.time + 0.5 * dt, dt)

            self.u = (self.u + 2 * (u_2 + dt * uk_3)) / 3
            self.v = (self.v + 2 * (v_2 + dt * vk_3)) / 3
            self.h = (self.h + 2 * (h_2 + dt * hk_3)) / 3
            self.hs = (self.hs + 2 * (hs_2 + dt * hsk_3)) / 3

        else:
            raise ValueError(f"order: expected one of [3], found {order}.")

        self.time += dt
        e, T, p, u = self.get_thermodynamics_quantities(self.h, self.hs)
        self.hb = self.h * T * (self.p0 / p) ** (self.R / self.cp)

    def set_initial_condition(self, u, v, h, hs):

        self.u = torch.from_numpy(u.astype(self.dtype)).to(self.device)
        self.v = torch.from_numpy(v.astype(self.dtype)).to(self.device)
        self.h = torch.from_numpy(h.astype(self.dtype)).to(self.device)
        self.hs = torch.from_numpy(hs.astype(self.dtype)).to(self.device)

        e, T, p, u = self.get_thermodynamics_quantities(self.h, self.hs)
        self.hb = self.h * T * (self.p0 / p) ** (self.R / self.cp)

        self.tmp1 = torch.zeros_like(self.u).to(self.device)
        self.tmp2 = torch.zeros_like(self.u).to(self.device)

        self.u_left = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype).to(self.device)
        self.u_right = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype).to(self.device)
        self.u_up = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype).to(self.device)
        self.u_down = torch.zeros((self.ny +1, self.nx, self.n), dtype=self.tmp1.dtype).to(self.device)

        self.v_left = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype).to(self.device)
        self.v_right = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype).to(self.device)
        self.v_up = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype).to(self.device)
        self.v_down = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype).to(self.device)

        self.h_left = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype).to(self.device)
        self.h_right = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype).to(self.device)
        self.h_up = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype).to(self.device)
        self.h_down = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype).to(self.device)

        self.hs_left = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype).to(self.device)
        self.hs_right = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.tmp1.dtype).to(self.device)
        self.hs_up = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype).to(self.device)
        self.hs_down = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.tmp1.dtype).to(self.device)

        self.boundaries(self.u, self.v, self.h, self.hs, 0)

    def integrate(self, q):
        return (q * self.w * self.J).sum()

    def plot_solution(self, ax, vmin=None, vmax=None, plot_func=None, dim=3, cmap='nipy_spectral'):
        x_plot = self.xs.swapaxes(1, 2).reshape(self.h.shape[0] * self.h.shape[2], -1)
        y_plot = self.ys.swapaxes(1, 2).reshape(self.h.shape[0] * self.h.shape[2], -1)

        if plot_func is None:
            z_plot = self.h.swapaxes(1, 2).reshape(self.h.shape[0] * self.h.shape[2], -1)
        else:
            out = plot_func(self)
            z_plot = out.swapaxes(1, 2).reshape(out.shape[0] * out.shape[2], -1)

        if dim == 3:
            return ax.plot_surface(x_plot, y_plot, z_plot, cmap=cmap, vmin=vmin, vmax=vmax)
        elif dim == 2:
            return ax.contourf(x_plot, y_plot, z_plot, cmap=cmap, vmin=vmin, vmax=vmax, levels=1000)
            #return ax.imshow(z_plot, cmap=cmap, vmin=vmin, vmax=vmax)

    def hflux(self, u, v, h, hs):
        yflux = v * h
        xflux = u * h
        return yflux, xflux

    def bflux(self, u, v, h, hs):
        yflux = v * hs
        xflux = u * hs
        return yflux, xflux

    def uv_flux(self, u, v, h, hs, e, T):
        return 0.5 * (u ** 2 + v ** 2) + e - T * hs / h

    def wave_speed(self, u, v, h, hs):
        # TODO: implement actual wave speed
        return 400.0 + u - u

    def solve(self, u, v, h, hs, t, dt, verbose=False):

        # copy the boundaries across
        self.boundaries(u, v, h, hs, t)
        s = hs / h
        s_up = self.hs_up / self.h_up
        s_down = self.hs_down / self.h_down
        s_right = self.hs_right / self.h_right
        s_left = self.hs_left / self.h_left

        c_up = self.wave_speed(self.u_up, self.v_up, self.h_up, self.hs_up)
        c_down = self.wave_speed(self.u_down, self.v_down, self.h_down, self.hs_down)
        c_right = self.wave_speed(self.u_right, self.v_right, self.h_right, self.hs_right)
        c_left = self.wave_speed(self.u_left, self.v_left, self.h_left, self.hs_left)
        c_ho = 0.5 * (c_right + c_left)
        c_ve = 0.5 * (c_up + c_down)

        h_ho = 0.5 * (self.h_right + self.h_left)
        h_ve = 0.5 * (self.h_up + self.h_down)

        e, T, _, _ = self.get_thermodynamics_quantities(h, hs)
        e_left, T_left, _, _ = self.get_thermodynamics_quantities(self.h_left, self.hs_left)
        e_right, T_right, _, _ = self.get_thermodynamics_quantities(self.h_right, self.hs_right)
        e_up, T_up, _, _ = self.get_thermodynamics_quantities(self.h_up, self.hs_up)
        e_down, T_down, _, _ = self.get_thermodynamics_quantities(self.h_down, self.hs_down)

        ### boundary fluxes

        h_up_flux_y, h_up_flux_x = self.hflux(self.u_up, self.v_up, self.h_up, self.hs_up)
        h_down_flux_y, h_down_flux_x = self.hflux(self.u_down, self.v_down, self.h_down, self.hs_down)
        h_right_flux_y, h_right_flux_x = self.hflux(self.u_right, self.v_right, self.h_right, self.hs_right)
        h_left_flux_y, h_left_flux_x = self.hflux(self.u_left, self.v_left, self.h_left, self.hs_left)

        h_up_flux = h_up_flux_y * self.eta_y_up + h_up_flux_x * self.eta_x_up
        h_down_flux = h_down_flux_y * self.eta_y_down + h_down_flux_x * self.eta_x_down
        h_right_flux = h_right_flux_y * self.xi_y_right + h_right_flux_x * self.xi_x_right
        h_left_flux = h_left_flux_y * self.xi_y_left + h_left_flux_x * self.xi_x_left

        uv_up_flux = self.uv_flux(self.u_up, self.v_up, self.h_up, self.hs_up, e_up, T_up)
        uv_down_flux = self.uv_flux(self.u_down, self.v_down, self.h_down, self.hs_down, e_down, T_down)
        uv_right_flux = self.uv_flux(self.u_right, self.v_right, self.h_right, self.hs_right, e_right, T_right)
        uv_left_flux = self.uv_flux(self.u_left, self.v_left, self.h_left, self.hs_left, e_left, T_left)

        s_up_flux_y, s_up_flux_x = self.bflux(self.u_up, self.v_up, self.h_up, self.hs_up)
        s_down_flux_y, s_down_flux_x = self.bflux(self.u_down, self.v_down, self.h_down, self.hs_down)
        s_right_flux_y, s_right_flux_x = self.bflux(self.u_right, self.v_right, self.h_right, self.hs_right)
        s_left_flux_y, s_left_flux_x = self.bflux(self.u_left, self.v_left, self.h_left, self.hs_left)

        s_up_flux = s_up_flux_y * self.eta_y_up + s_up_flux_x * self.eta_x_up
        s_down_flux = s_down_flux_y * self.eta_y_down + s_down_flux_x * self.eta_x_down
        s_right_flux = s_right_flux_y * self.xi_y_right + s_right_flux_x * self.xi_x_right
        s_left_flux = s_left_flux_y * self.xi_y_left + s_left_flux_x * self.xi_x_left

        ###  handle h
        h_yflux, h_xflux = self.hflux(u, v, h, hs)

        div = torch.einsum('fgcd,abcd->fgab', h_xflux, self.ddxi) * self.dxidx
        div += torch.einsum('fgcd,abcd->fgab', h_xflux, self.ddeta) * self.detadx
        div += torch.einsum('fgcd,abcd->fgab', h_yflux, self.ddxi) * self.dxidy
        div += torch.einsum('fgcd,abcd->fgab', h_yflux, self.ddeta) * self.detady
        out = -self.w * self.J * div

        h_flux_vert = 0.5 * (h_up_flux + h_down_flux) #- self.a * c_ve * (self.h_up - self.h_down)
        h_flux_horz = 0.5 * (h_right_flux + h_left_flux) #- self.a * c_ho * (self.h_right - self.h_left)
        h_flux_vert[0, ...] = 0.0
        h_flux_vert[-1, ...] = 0.0

        self.tmp1[:, :, -1] = (h_flux_vert[1:] - h_down_flux[1:]) * (self.w_x * self.Jx[:, :, -1])
        self.tmp1[:, :, 0] = -(h_flux_vert[:-1] - h_up_flux[:-1]) * (self.w_x * self.Jx[:, :, 0])
        self.tmp2[:, :, :, -1] = (h_flux_horz[:, 1:] - h_left_flux[:, 1:]) * (self.w_x * self.Jy[:, :, :, -1])
        self.tmp2[:, :, :, 0] = -(h_flux_horz[:, :-1] - h_right_flux[:, :-1]) * (self.w_x * self.Jy[:, :, :, 0])
        out -= (self.tmp1 + self.tmp2)

        h_k = out / (self.J * self.w)

        # handle s

        hs_yflux, hs_xflux = self.bflux(u, v, h, hs)
        sdiv = torch.einsum('fgcd,abcd->fgab', hs_xflux, self.ddxi) * self.dxidx
        sdiv += torch.einsum('fgcd,abcd->fgab', hs_xflux, self.ddeta) * self.detadx
        sdiv += torch.einsum('fgcd,abcd->fgab', hs_yflux, self.ddxi) * self.dxidy
        sdiv += torch.einsum('fgcd,abcd->fgab', hs_yflux, self.ddeta) * self.detady

        dsdx = torch.einsum('fgcd,abcd->fgab', s, self.ddxi) * self.dxidx
        dsdx += torch.einsum('fgcd,abcd->fgab', s, self.ddeta) * self.detadx
        dsdy = torch.einsum('fgcd,abcd->fgab', s, self.ddxi) * self.dxidy
        dsdy += torch.einsum('fgcd,abcd->fgab', s, self.ddeta) * self.detady

        out = -self.w * self.J * 0.5 * (sdiv + s * div + h_xflux * dsdx + h_yflux * dsdy)

        s_flux_horz = 0.5 * (s_right + s_left) * h_flux_horz - self.a * abs(h_flux_horz) * (s_right - s_left)

        s_flux_vert = 0.5 * (s_up + s_down) * h_flux_vert - self.a * abs(h_flux_vert) * (s_up - s_down)

        self.tmp1[:, :, -1] = (s_flux_vert[1:] - s_down_flux[1:]) * (self.w_x * self.Jx[:, :, -1])
        self.tmp1[:, :, 0] = -(s_flux_vert[:-1] - s_up_flux[:-1]) * (self.w_x * self.Jx[:, :, 0])
        self.tmp2[:, :, :, -1] = (s_flux_horz[:, 1:] - s_left_flux[:, 1:]) * (self.w_x * self.Jy[:, :, :, -1])
        self.tmp2[:, :, :, 0] = -(s_flux_horz[:, :-1] - s_right_flux[:, :-1]) * (self.w_x * self.Jy[:, :, :, 0])
        out -= (self.tmp1 + self.tmp2)

        s_k = out / (self.J * self.w)

        # u and v fluxes
        ########
        #######

        uv_flux = self.uv_flux(u, v, h, hs, e, T)

        dTdx = self.ddx(T)
        dsTdx = self.ddx(s * T)
        dTdy = self.ddy(T)
        dsTdy = self.ddy(s * T)

        # upwinded
        s_hat_left = 0.5 * (0.5 * (s_right + s_left) - self.a * torch.sign(h_flux_horz) * (s_right - s_left))
        s_hat_right = s_hat_left
        s_hat_down = 0.5 * (0.5 * (s_up + s_down) - self.a * torch.sign(h_flux_vert) * (s_up - s_down))
        s_hat_up = s_hat_down

        # vorticity

        vort = self.ddx(v) - self.ddy(u)
        vort += self.f

        # handle u
        #######
        ###
        duv_fluxdxi = torch.einsum('fgcd,abcd->fgab', uv_flux, self.ddxi)
        duv_fluxdeta = torch.einsum('fgcd,abcd->fgab', uv_flux, self.ddeta)

        out = -(duv_fluxdxi * self.dxidx + duv_fluxdeta * self.detadx) * self.J * self.w
        out -= -vort * v * self.J * self.w

        diff = h_right_flux - h_left_flux
        uv_flux_horz = 0.5 * (uv_right_flux + uv_left_flux) - self.a * (c_ho / h_ho) * diff

        diff = h_up_flux - h_down_flux
        uv_flux_vert = 0.5 * (uv_up_flux + uv_down_flux) - self.a * (c_ve / h_ve) * diff
        uv_flux_vert[-1, ...] = (uv_down_flux - self.a * (c_ve / h_ve) * diff)[-1, ...]
        uv_flux_vert[0, ...] = (uv_up_flux - self.a * (c_ve / h_ve) * diff)[0, ...]

        self.tmp1[:, :, -1] = (uv_flux_vert - uv_down_flux)[1:] * (self.w_x * self.Jx[:, :, -1]) * self.eta_x_down[1:]
        self.tmp1[:, :, 0] = -(uv_flux_vert - uv_up_flux)[:-1] * (self.w_x * self.Jx[:, :, 0]) * self.eta_x_up[:-1]
        self.tmp2[:, :, :, -1] = (uv_flux_horz - uv_left_flux)[:, 1:] * (self.w_x * self.Jy[:, :, :, -1]) * self.xi_x_left[:, 1:]
        self.tmp2[:, :, :, 0] = -(uv_flux_horz - uv_right_flux)[:, :-1] * (self.w_x * self.Jy[:, :, :, 0]) * self.xi_x_right[:, :-1]

        # if not self.passive:
        out -= 0.5  * self.J * self.w * (s * dTdx + dsTdx - T * dsdx)

        # todo: which way do the signs go?
        self.tmp1[:, :, -1] += (s_hat_down * (T_up - T_down))[1:] * (self.w_x * self.Jx[:, :, -1]) * self.eta_x_down[1:]
        self.tmp1[:, :, 0] += - (s_hat_up * (T_down - T_up))[:-1] * (self.w_x * self.Jx[:, :, 0]) * self.eta_x_up[:-1]
        self.tmp2[:, :, :, -1] +=  (s_hat_left * (T_right - T_left))[:, 1:] * (self.w_x * self.Jy[:, :, :, -1]) * self.xi_x_left[:, 1:]
        self.tmp2[:, :, :, 0] += - (s_hat_right * (T_left - T_right))[:, :-1] * (self.w_x * self.Jy[:, :, :, 0]) * self.xi_x_right[:, :-1]

        # vorticity
        self.tmp1[:, :, -1] += 0.5 * (self.v_down * (self.u_up - self.u_down))[1:] * (self.w_x * self.Jx[:, :, -1])
        self.tmp1[:, :, 0] += 0.5 * (self.v_up * (self.u_up - self.u_down))[:-1] * (self.w_x * self.Jx[:, :, 0])
        self.tmp2[:, :, :, -1] += -0.5 * (self.v_left * (self.v_right - self.v_left))[:, 1:] * (self.w_x * self.Jy[:, :, :, -1])
        self.tmp2[:, :, :, 0] += -0.5 * (self.v_right * (self.v_right - self.v_left))[:, :-1] * (self.w_x * self.Jy[:, :, :, 0])

        out -= (self.tmp1 + self.tmp2)

        u_k = out / (self.J * self.w)

        # handle v
        #######
        ###

        out = -(duv_fluxdxi * self.dxidy + duv_fluxdeta * self.detady) * self.J * self.w
        out -= vort * u * self.J * self.w

        self.tmp1[:, :, -1] = (uv_flux_vert - uv_down_flux)[1:] * (self.w_x * self.Jx[:, :, -1]) * self.eta_y_down[1:]
        self.tmp1[:, :, 0] = -(uv_flux_vert - uv_up_flux)[:-1] * (self.w_x * self.Jx[:, :, 0]) * self.eta_y_up[:-1]
        self.tmp2[:, :, :, -1] = (uv_flux_horz - uv_left_flux)[:, 1:] * (self.w_x * self.Jy[:, :, :, -1]) * self.xi_y_left[:, 1:]
        self.tmp2[:, :, :, 0] = -(uv_flux_horz - uv_right_flux)[:, :-1] * (self.w_x * self.Jy[:, :, :, 0]) * self.xi_y_right[:, :-1]

        out -= 0.5 * self.J * self.w * (s * dTdy + dsTdy - T * dsdy)

        self.tmp1[:, :, -1] += (s_hat_down * (T_up - T_down))[1:] * (self.w_x * self.Jx[:, :, -1]) * self.eta_y_down[1:]
        self.tmp1[:, :, 0] += - (s_hat_up * (T_down - T_up))[:-1] * (self.w_x * self.Jx[:, :, 0]) * self.eta_y_up[:-1]
        self.tmp2[:, :, :, -1] += (s_hat_left * (T_right - T_left))[:, 1:] * (self.w_x * self.Jy[:, :, :, -1]) * self.xi_y_left[:, 1:]
        self.tmp2[:, :, :, 0] += - (s_hat_right * (T_left - T_right))[:, :-1] * (self.w_x * self.Jy[:, :, :, 0]) * self.xi_y_right[:, :-1]

        # vorticity boundary terms
        self.tmp1[:, :, -1] += -0.5 * (self.u_down * (self.u_up - self.u_down))[1:] * (self.w_x * self.Jx[:, :, -1])
        self.tmp1[:, :, 0] += -0.5 * (self.u_up * (self.u_up - self.u_down))[:-1] * (self.w_x * self.Jx[:, :, 0])
        self.tmp2[:, :, :, -1] += 0.5 * (self.u_left * (self.v_right - self.v_left))[:, 1:] * (self.w_x * self.Jy[:, :, :, -1])
        self.tmp2[:, :, :, 0] += 0.5 * (self.u_right * (self.v_right - self.v_left))[:, :-1] * (self.w_x * self.Jy[:, :, :, 0])

        out -= (self.tmp1 + self.tmp2)
        v_k = out / (self.J * self.w)

        v_k -= self.g

        if verbose:
            time_deriv = {'h': h_k, 'u': u_k, 'v': v_k, 'hs': s_k}
            for name in time_deriv.keys():
                print(f'd{name}/dt abs max: {abs(time_deriv[name]).max()}.')

        return u_k, v_k, h_k, s_k

    def ddx(self, q):
        dqdx = torch.einsum('fgcd,abcd->fgab', q, self.ddxi) * self.dxidx
        dqdx += torch.einsum('fgcd,abcd->fgab', q, self.ddeta) * self.detadx
        return dqdx

    def ddy(self, q):
        dqdy = torch.einsum('fgcd,abcd->fgab', q, self.ddxi) * self.dxidy
        dqdy += torch.einsum('fgcd,abcd->fgab', q, self.ddeta) * self.detady
        return dqdy

    def energy(self):
        return self.ke() + self.pe() + self.ie()

    def ke(self):
        return 0.5 * self.h * (self.u ** 2 + self.v ** 2)

    def pe(self):
        return self.h * self.g * self.y_tnsr

    def ie(self):
        _, _, _, u = self.get_thermodynamics_quantities(self.h, self.hs)
        return self.h * u

    def get_thermodynamics_quantities(self, h, hs):
        s = hs / h

        # s = log(p) - gamma * log(h)
        # p = exp(s + gamma * log(h)) = exp(s) * h**gamma
        # T = p / (R * h)
        # u = p / (h * (gamma - 1))

        p = torch.exp(s / self.cv) * h**self.gamma
        T = p / (self.R * h)
        u = p / (h * (self.gamma - 1))
        e = u + (p / h)

        return e, T, p, u
