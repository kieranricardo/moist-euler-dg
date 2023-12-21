import meshzoo
import torch
import numpy as np
from .utils import gll, lagrange1st
from matplotlib import pyplot as plt
from scipy.interpolate import lagrange


class EquilibriumEuler2D:

    def __init__(
            self, xrange, yrange, poly_order, nx, ny,
            g, eps, device='cpu', solution=None, a=0.0, dtype=np.float64,
            angle=0.0, upwind=True, a_bdry=0.0, strong_bcs=False, **kwargs,
    ):
        self.time = 0
        self.poly_order = poly_order

        self.left_state = dict()
        self.right_state = dict()
        self.down_state = dict()
        self.up_state = dict()
        self.state = dict()
        self.potential_temperature = self.tmp1 = self.tmp2 = self.qv = self.p = None
        self.moist_pt = None
        self.gibbs_error = None
        self.a_bdry = a_bdry
        self.strong_bcs = strong_bcs

        self.diagnostics = dict((name, []) for name in (
            'energy', 'entropy', 'mass', 'water', 'vapour',
            'entropy_variance', 'water_variance', 'gibbs_error',
            'dEdt', 'time',
            )
        )

        self.g = g
        self.f = 0
        self.eps = eps
        self.a = a
        self.solution = solution
        self.dtype = dtype
        self.xperiodic = True
        self.yperiodic = False
        self.upwind = upwind

        # dry quantities
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

    def boundaries(self, state):

        for name in self.state.keys():
            self.up_state[name][:-1] = state[name][:, :, 0, :]
            self.down_state[name][1:] = state[name][:, :, -1, :]
            self.right_state[name][:, :-1] = state[name][:, :, :, 0]
            self.left_state[name][:, 1:] = state[name][:, :, :, -1]

        # vertical wall BC
        for name in self.state.keys():
            if name != 'v':
                self.up_state[name][-1] = state[name][-1, :, -1, :]
                self.down_state[name][0] = state[name][0, :, 0, :]

        self.up_state['v'][-1] = 0.0
        self.down_state['v'][0] = 0.0

        # horizontal periodic BC
        # vertical wall BC
        for name in self.state.keys():
            self.right_state[name][:, -1] = state[name][:, 0, :, 0]
            self.left_state[name][:, 0] = state[name][:, -1, :, -1]

    def time_step(self, dt=None, order=3, forcing=None):


        self.diagnostics['energy'].append(self.integrate(self.energy()))
        self.diagnostics['mass'].append(self.state['h'])
        self.diagnostics['water'].append(self.state['hqw'])
        self.diagnostics['entropy'].append(self.state['hs'])
        self.diagnostics['entropy_variance'].append(self.integrate(self.variance(self.state['hs'])))
        self.diagnostics['water_variance'].append(self.integrate(self.variance(self.state['hqw'])))
        self.diagnostics['vapour'].append(self.integrate(self.state['h'] * self.qv))
        self.diagnostics['gibbs_error'].append(self.gibbs_error)
        self.diagnostics['dEdt'].append(self.integrate(self.get_dEdt()))
        self.diagnostics['time'].append(self.time)

        if dt is None:
            speed = self.wave_speed(self.state)
            dt = self.cdt / torch.max(speed).cpu().numpy()

        if order == 3:
            k1 = self.solve(self.state)
            state1 = {name: self.state[name] + dt * k1[name] for name in self.state.keys()}

            k2 = self.solve(state1)
            state2 = {name: 0.75 * self.state[name] + 0.25 * (state1[name] + k2[name] * dt) for name in self.state.keys()}

            k3 = self.solve(state2)
            self.state = {name: (self.state[name] + 2 * (state2[name] + dt * k3[name])) / 3 for name in self.state.keys()}

        else:
            raise ValueError(f"order: expected one of [3], found {order}.")

        self.time += dt
        ie, die_d, p, qv = self.get_thermodynamics_quantities(self.state)
        T = die_d['hs']
        # self.potential_temperature = T * (self.p0_ex / p) ** (self.Rd / self.cpd)
        ex = (p / self.p0_ex)**(self.Rd / self.cpd)
        self.potential_temperature = p / (self.state['h'] * self.Rd * ex)
        self.p = p

        self.qv = qv

    def set_initial_condition(self, u, v, h, hs, hqw):

        self.state = {
            'u': torch.from_numpy(u.astype(self.dtype)).to(self.device),
            'v': torch.from_numpy(v.astype(self.dtype)).to(self.device),
            'h': torch.from_numpy(h.astype(self.dtype)).to(self.device),
            'hs': torch.from_numpy(hs.astype(self.dtype)).to(self.device),
            'hqw': torch.from_numpy(hqw.astype(self.dtype)).to(self.device),
        }

        for name in self.state.keys():
            self.left_state[name] = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.state['u'].dtype).to(self.device)
            self.right_state[name] = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.state['u'].dtype).to(self.device)
            self.down_state[name] = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.state['u'].dtype).to(self.device)
            self.up_state[name] = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.state['u'].dtype).to(self.device)

        ie, die_d, p, qv = self.get_thermodynamics_quantities(self.state)
        T = die_d['hs']
        ex = (p / self.p0_ex) ** (self.Rd / self.cpd)
        self.potential_temperature = p / (self.state['h'] * self.Rd * ex)
        self.p = p
        self.qv = qv

        self.tmp1 = torch.zeros_like(self.state['h']).to(self.device)
        self.tmp2 = torch.zeros_like(self.state['h']).to(self.device)

        self.boundaries(self.state)

    def integrate(self, q):
        return (q * self.w * self.J).sum()

    def plot_solution(self, ax, vmin=None, vmax=None, plot_func=None, dim=3, cmap='nipy_spectral'):
        x_plot = self.xs.swapaxes(1, 2).reshape(self.state['h'].shape[0] * self.state['h'].shape[2], -1)
        y_plot = self.ys.swapaxes(1, 2).reshape(self.state['h'].shape[0] * self.state['h'].shape[2], -1)

        if plot_func is None:
            z_plot = self.state['h'].swapaxes(1, 2).reshape(self.state['h'].shape[0] * self.state['h'].shape[2], -1)
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

    def uv_flux(self, u, v, h, hs, die_d):
        return 0.5 * (u ** 2 + v ** 2) + die_d['h']

    def wave_speed(self, state):
        # TODO: implement actual wave speed
        return 400.0 + state['u'] * 0.0

    def project_H1(self, u):
        u_sum = torch.zeros_like(u) + u
        count = torch.ones_like(u)

        for tnsr in [u_sum, count]:
            tnsr[:, 1:, :, 0] = tnsr[:, 1:, :, 0] + tnsr[:, :-1, :, -1]
            tnsr[:, :-1, :, -1] = tnsr[:, 1:, :, 0]

            tnsr[1:, :, 0] = tnsr[1:, :, 0] + tnsr[:-1, :, -1]
            tnsr[:-1, :, -1] = tnsr[1:, :, 0]

            if self.xperiodic:
                tnsr[:, 0, :, 0] = tnsr[:, 0, :, 0] + tnsr[:, -1, :, -1]
                tnsr[:, -1, :, -1] = tnsr[:, 0, :, 0]

            if self.yperiodic:
                tnsr[0, :, 0] = tnsr[0, :, 0] + tnsr[-1, :, -1]
                tnsr[-1, :, -1] = tnsr[0, :, 0]

        return u_sum / count

    def solve(self, state, verbose=False):

        # copy the boundaries across
        self.boundaries(state)
        time_deriv = dict()

        # pull out variables from states
        u, v, h, hs, hqw = state['u'], state['v'], state['h'], state['hs'], state['hqw']
        u_left, v_left, h_left, hs_left, hqw_left = self.left_state['u'], self.left_state['v'], self.left_state['h'], self.left_state['hs'], self.left_state['hqw']
        u_right, v_right, h_right, hs_right, hqw_right = self.right_state['u'], self.right_state['v'], self.right_state['h'], self.right_state['hs'], self.right_state['hqw']
        u_down, v_down, h_down, hs_down, hqw_down = self.down_state['u'], self.down_state['v'], self.down_state['h'], self.down_state['hs'], self.down_state['hqw']
        u_up, v_up, h_up, hs_up, hqw_up = self.up_state['u'], self.up_state['v'], self.up_state['h'], self.up_state['hs'], self.up_state['hqw']

        c_up = self.wave_speed(self.up_state)
        c_down = self.wave_speed(self.down_state)
        c_right = self.wave_speed(self.right_state)
        c_left = self.wave_speed(self.left_state)
        c_ho = 0.5 * (c_right + c_left)
        c_ve = 0.5 * (c_up + c_down)

        h_ho = 0.5 * (h_right + h_left)
        h_ve = 0.5 * (h_up + h_down)

        _, die_d, *_ = self.get_thermodynamics_quantities(state)

        up_die_d = dict()
        down_die_d = dict()
        right_die_d = dict()
        left_die_d = dict()

        for name in die_d.keys():
            up_die_d[name] = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.state['u'].dtype).to(self.device)
            down_die_d[name] = torch.zeros((self.ny + 1, self.nx, self.n), dtype=self.state['u'].dtype).to(self.device)
            right_die_d[name] = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.state['u'].dtype).to(self.device)
            left_die_d[name] = torch.zeros((self.ny, self.nx + 1, self.n), dtype=self.state['u'].dtype).to(self.device)

            up_die_d[name][:-1] = die_d[name][:, :, 0, :]
            down_die_d[name][1:] = die_d[name][:, :, -1, :]
            right_die_d[name][:, :-1] = die_d[name][:, :, :, 0]
            left_die_d[name][:, 1:] = die_d[name][:, :, :, -1]

            # horizontal periodic
            right_die_d[name][:, -1] = die_d[name][:, 0, :, 0]
            left_die_d[name][:, 0] = die_d[name][:, -1, :, -1]
            # vertical wall
            up_die_d[name][-1] = die_d[name][-1, :, -1, :]
            down_die_d[name][0] = die_d[name][0, :, 0, :]

        ### boundary fluxes

        h_up_flux_y, h_up_flux_x = self.hflux(u_up, v_up, h_up, hs_up)
        h_down_flux_y, h_down_flux_x = self.hflux(u_down, v_down, h_down, hs_down)
        h_right_flux_y, h_right_flux_x = self.hflux(u_right, v_right, h_right, hs_right)
        h_left_flux_y, h_left_flux_x = self.hflux(u_left, v_left, h_left, hs_left)

        h_up_flux = h_up_flux_y * self.eta_y_up + h_up_flux_x * self.eta_x_up
        h_down_flux = h_down_flux_y * self.eta_y_down + h_down_flux_x * self.eta_x_down
        h_right_flux = h_right_flux_y * self.xi_y_right + h_right_flux_x * self.xi_x_right
        h_left_flux = h_left_flux_y * self.xi_y_left + h_left_flux_x * self.xi_x_left

        uv_up_flux = self.uv_flux(u_up, v_up, h_up, hs_up, up_die_d)
        uv_down_flux = self.uv_flux(u_down, v_down, h_down, hs_down, down_die_d)
        uv_right_flux = self.uv_flux(u_right, v_right, h_right, hs_right, right_die_d)
        uv_left_flux = self.uv_flux(u_left, v_left, h_left, hs_left, left_die_d)

        ###  handle h
        h_yflux, h_xflux = self.hflux(u, v, h, hs)

        div = self.ddx(h_xflux) + self.ddy(h_yflux)
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

        time_deriv['h'] = out / (self.J * self.w)

        # u and v fluxes
        ########
        #######

        ### velocity joint terms
        uv_flux = self.uv_flux(u, v, h, hs, die_d)

        vort = self.ddx(v) - self.ddy(u)
        vort += self.f

        diff = h_right_flux - h_left_flux
        uv_flux_horz = 0.5 * (uv_right_flux + uv_left_flux) - self.a * (c_ho / h_ho) * diff

        diff = h_up_flux - h_down_flux
        uv_flux_vert = 0.5 * (uv_up_flux + uv_down_flux) - self.a * (c_ve / h_ve) * diff

        # wall boundaries
        uv_flux_vert[-1, ...] = (uv_down_flux - self.a_bdry * (c_ve / h_ve) * diff)[-1, ...]
        uv_flux_vert[0, ...] = (uv_up_flux - self.a_bdry * (c_ve / h_ve) * diff)[0, ...]

        # handle u
        #######
        ###

        out = -self.ddx(uv_flux) * self.J * self.w
        out -= -vort * v * self.J * self.w

        self.tmp1[:, :, -1] = (uv_flux_vert - uv_down_flux)[1:] * (self.w_x * self.Jx[:, :, -1]) * self.eta_x_down[1:]
        self.tmp1[:, :, 0] = -(uv_flux_vert - uv_up_flux)[:-1] * (self.w_x * self.Jx[:, :, 0]) * self.eta_x_up[:-1]
        self.tmp2[:, :, :, -1] = (uv_flux_horz - uv_left_flux)[:, 1:] * (self.w_x * self.Jy[:, :, :, -1]) * self.xi_x_left[:, 1:]
        self.tmp2[:, :, :, 0] = -(uv_flux_horz - uv_right_flux)[:, :-1] * (self.w_x * self.Jy[:, :, :, 0]) * self.xi_x_right[:, :-1]

        # vorticity
        self.tmp1[:-1, :, -1] += 0.5 * (v_down * (u_up - u_down))[1:-1] * (self.w_x * self.Jx[:-1, :, -1])
        self.tmp1[1:, :, 0] += 0.5 * (v_up * (u_up - u_down))[1:-1] * (self.w_x * self.Jx[1:, :, 0])
        self.tmp2[:, :, :, -1] += -0.5 * (v_left * (v_right - v_left))[:, 1:] * (self.w_x * self.Jy[:, :, :, -1])
        self.tmp2[:, :, :, 0] += -0.5 * (v_right * (v_right - v_left))[:, :-1] * (self.w_x * self.Jy[:, :, :, 0])

        self.tmp1[-1, :, -1] += 0.5 * (v_down * (u_up - u_down))[-1] * (self.w_x * self.Jx[-1, :, -1])[0]
        self.tmp1[0, :, 0] += 0.5 * (v_up * (u_up - u_down))[0] * (self.w_x * self.Jx[0, :, 0])[0]

        # fake tangent BCs - 0.5 = linear neutral, 1 = linear dissipative
        # self.tmp1[-1, :, -1] += -0.5 * ((v_down < 0) * v_down * u_down)[-1] * (self.w_x * self.Jx[-1, :, -1])[0]
        # self.tmp1[0, :, 0] += 0.5 * ((v_up > 0) * v_up * u_up)[0] * (self.w_x * self.Jx[0, :, 0])[0]

        out -= (self.tmp1 + self.tmp2)
        time_deriv['u'] = out / (self.J * self.w)

        # handle v
        #######
        ###

        out = -self.ddy(uv_flux) * self.J * self.w
        out -= vort * u * self.J * self.w

        self.tmp1[:, :, -1] = (uv_flux_vert - uv_down_flux)[1:] * (self.w_x * self.Jx[:, :, -1]) * self.eta_y_down[1:]
        self.tmp1[:, :, 0] = -(uv_flux_vert - uv_up_flux)[:-1] * (self.w_x * self.Jx[:, :, 0]) * self.eta_y_up[:-1]
        self.tmp2[:, :, :, -1] = (uv_flux_horz - uv_left_flux)[:, 1:] * (self.w_x * self.Jy[:, :, :, -1]) * self.xi_y_left[:, 1:]
        self.tmp2[:, :, :, 0] = -(uv_flux_horz - uv_right_flux)[:, :-1] * (self.w_x * self.Jy[:, :, :, 0]) * self.xi_y_right[:, :-1]

        # vorticity boundary terms
        self.tmp1[:-1, :, -1] += -0.5 * (u_down * (u_up - u_down))[1:-1] * (self.w_x * self.Jx[:-1, :, -1])
        self.tmp1[1:, :, 0] += -0.5 * (u_up * (u_up - u_down))[1:-1] * (self.w_x * self.Jx[1:, :, 0])
        self.tmp2[:, :, :, -1] += 0.5 * (u_left * (v_right - v_left))[:, 1:] * (self.w_x * self.Jy[:, :, :, -1])
        self.tmp2[:, :, :, 0] += 0.5 * (u_right * (v_right - v_left))[:, :-1] * (self.w_x * self.Jy[:, :, :, 0])

        self.tmp1[-1, :, -1] += -0.5 * (u_down * (u_up - u_down))[-1] * (self.w_x * self.Jx[-1, :, -1])[0]
        self.tmp1[0, :, 0] += -0.5 * (u_up * (u_up - u_down))[0] * (self.w_x * self.Jx[0, :, 0])[0]

        # self.tmp1[-1, :, -1] += ((v_down < 0) * u_down * u_down)[-1] * (self.w_x * self.Jx[-1, :, -1])[0]
        # self.tmp1[0, :, 0] += -((v_up > 0) * u_up * u_up)[0] * (self.w_x * self.Jx[0, :, 0])[0]

        out -= (self.tmp1 + self.tmp2)
        time_deriv['v'] = (out / (self.J * self.w)) - self.g

        # tracer time
        tracer_names = ['hs', 'hqw']

        for name in tracer_names:
            hq = state[name]
            q = hq / h

            q_up, q_down, q_right, q_left = self.up_state[name] / h_up, self.down_state[name] / h_down, \
                                            self.right_state[name] / h_right, self.left_state[name] / h_left
            # calculate derivatives and fluxes
            qdiv = torch.einsum('fgcd,abcd->fgab', q * h_xflux, self.ddxi) * self.dxidx
            qdiv += torch.einsum('fgcd,abcd->fgab', q * h_xflux, self.ddeta) * self.detadx
            qdiv += torch.einsum('fgcd,abcd->fgab', q * h_yflux, self.ddxi) * self.dxidy
            qdiv += torch.einsum('fgcd,abcd->fgab', q * h_yflux, self.ddeta) * self.detady

            dqdx = self.ddx(q)
            dqdy = self.ddy(q)

            # upwinded
            if self.upwind:
                q_hat_horz = 0.5 * (q_right + q_left) - 0.5 * torch.sign(h_flux_horz) * (q_right - q_left)
                q_hat_vert = 0.5 * (q_up + q_down) - 0.5 * torch.sign(h_flux_vert) * (q_up - q_down)
            else:
                q_hat_horz = 0.5 * (q_right + q_left)
                q_hat_vert = 0.5 * (q_up + q_down)

            # advance tracer
            out = -self.w * self.J * 0.5 * (qdiv + q * div + h_xflux * dqdx + h_yflux * dqdy)
            self.tmp1[:, :, -1] = (q_hat_vert * h_flux_vert - q_down * h_down_flux)[1:] * (self.w_x * self.Jx[:, :, -1])
            self.tmp1[:, :, 0] = -(q_hat_vert * h_flux_vert - q_up * h_up_flux)[:-1] * (self.w_x * self.Jx[:, :, 0])
            self.tmp2[:, :, :, -1] = (q_hat_horz * h_flux_horz - q_left * h_left_flux)[:, 1:] * (self.w_x * self.Jy[:, :, :, -1])
            self.tmp2[:, :, :, 0] = -(q_hat_horz * h_flux_horz - q_right * h_right_flux)[:, :-1] * (self.w_x * self.Jy[:, :, :, 0])
            out -= (self.tmp1 + self.tmp2)
            time_deriv[name] = out / (self.J * self.w)

            # update u
            out = -0.5 * self.J * self.w * (q * self.ddx(die_d[name]) + self.ddx(die_d[name] * q) - die_d[name] * dqdx)
            self.tmp1[:, :, -1] = (0.5 * q_hat_vert * (up_die_d[name] - down_die_d[name]))[1:] * (self.w_x * self.Jx[:, :, -1]) * self.eta_x_down[1:]
            self.tmp1[:, :, 0] = -(0.5 * q_hat_vert * (down_die_d[name] - up_die_d[name]))[:-1] * (self.w_x * self.Jx[:, :, 0]) * self.eta_x_up[:-1]
            self.tmp2[:, :, :, -1] = (0.5 * q_hat_horz * (right_die_d[name] - left_die_d[name]))[:, 1:] * (self.w_x * self.Jy[:, :, :, -1]) * self.xi_x_left[:, 1:]
            self.tmp2[:, :, :, 0] = -(0.5 * q_hat_horz * (left_die_d[name] - right_die_d[name]))[:, :-1] * (self.w_x * self.Jy[:, :, :, 0]) * self.xi_x_right[:, :-1]
            out -= (self.tmp1 + self.tmp2)
            time_deriv['u'] += out / (self.J * self.w)

            # update v
            out = -0.5 * self.J * self.w * (q * self.ddy(die_d[name]) + self.ddy(die_d[name] * q) - die_d[name] * dqdy)
            self.tmp1[:, :, -1] = (0.5 * q_hat_vert * (up_die_d[name] - down_die_d[name]))[1:] * (self.w_x * self.Jx[:, :, -1]) * self.eta_y_down[1:]
            self.tmp1[:, :, 0] = -(0.5 * q_hat_vert * (down_die_d[name] - up_die_d[name]))[:-1] * (self.w_x * self.Jx[:, :, 0]) * self.eta_y_up[:-1]
            self.tmp2[:, :, :, -1] = (0.5 * q_hat_horz * (right_die_d[name] - left_die_d[name]))[:, 1:] * (self.w_x * self.Jy[:, :, :, -1]) * self.xi_y_left[:, 1:]
            self.tmp2[:, :, :, 0] = -(0.5 * q_hat_horz * (left_die_d[name] - right_die_d[name]))[:, :-1] * (self.w_x * self.Jy[:, :, :, 0]) * self.xi_y_right[:, :-1]
            out -= (self.tmp1 + self.tmp2)
            time_deriv['v'] += out / (self.J * self.w)

        if verbose:
            for name in time_deriv.keys():
                print(f'd{name}/dt abs max: {abs(time_deriv[name]).max()}.')

        if self.strong_bcs:
            time_deriv['v'][-1, :, -1] = 0.0
            time_deriv['v'][0, :, 0] = 0.0
            time_deriv['u'][-1, :, -1] = 0.0
            time_deriv['u'][0, :, 0] = 0.0

        return time_deriv

    def ddx(self, q):
        dqdx = torch.einsum('fgcd,abcd->fgab', q, self.ddxi) * self.dxidx
        dqdx += torch.einsum('fgcd,abcd->fgab', q, self.ddeta) * self.detadx
        return dqdx

    def ddy(self, q):
        dqdy = torch.einsum('fgcd,abcd->fgab', q, self.ddxi) * self.dxidy
        dqdy += torch.einsum('fgcd,abcd->fgab', q, self.ddeta) * self.detady
        return dqdy

    def energy(self):
        return self.get_ke() + self.get_pe() + self.get_ie()

    def variance(self, hq):
        return (hq**2) / self.state['h']

    def get_ke(self):

        return 0.5 * self.state['h'] * (self.state['u'] ** 2 + self.state['v'] ** 2)

    def get_pe(self):
        return self.state['h'] * self.g * self.y_tnsr

    def get_ie(self):
        return self.get_thermodynamics_quantities(self.state)[0]

    def get_dEdt(self):
        ddt = self.solve(self.state)
        dkedt = 0.5 * ddt['h'] * (self.state['u'] ** 2 + self.state['v'] ** 2)
        dkedt += self.state['h'] * (self.state['u'] * ddt['u'] + self.state['v'] * ddt['v'])

        dpedt = ddt['h'] * self.g * self.y_tnsr

        ie, die_d, p, qv = self.get_thermodynamics_quantities(self.state)
        diedt = die_d['h'] * ddt['h'] + die_d['hs'] * ddt['hs'] + die_d['hqw'] * ddt['hqw']

        dEdt = dkedt + dpedt + diedt
        return dEdt

    def get_thermodynamics_quantities(self, state):
        h, hs, hqw = state['h'], state['hs'], state['hqw']
        s = hs / h
        qw = hqw / h
        qd = 1 - qw

        qv = self.solve_qv_from_entropy(h, qw, s, mathlib=torch, qv=self.qv)

        qv = torch.minimum(qv, qw)

        ql = qw - qv
        R = qv * self.Rv + qd * self.Rd
        cv = qd * self.cvd + qv * self.cvv + ql * self.cl

        logT = (1 / cv) * (s + R * torch.log(h) + qd * self.Rd * torch.log(self.Rd * qd) + qv * self.Rv * torch.log(self.Rv * qv) - qv * self.c0 - ql * self.c1)
        T = torch.exp(logT)
        p = h * R * T

        # print('Model T min-max:', T.min(), T.max())
        # print('Model p min-max:', p.min(), p.max())

        specific_ie = cv * T + qv * self.Lv0
        enthalpy = specific_ie + p / h
        ie = h * specific_ie

        dlogTdqv = (1 / cv) * (self.Rv * torch.log(h) + self.Rv * torch.log(self.Rv * qv) + self.Rv - self.c0)
        dlogTdqv += -(1 / cv) * logT * self.cvv
        dTdqv = dlogTdqv * T

        dlogTdql = (1 / cv) * (-self.c1)
        dlogTdql += -(1 / cv) * logT * self.cl
        dTdql = dlogTdql * T

        dlogTdqd = (1 / cv) * (self.Rd * torch.log(h) + self.Rd * torch.log(self.Rd * qd) + self.Rd)
        dlogTdqd += -(1 / cv) * logT * self.cvd
        dTdqd = dlogTdqd * T

        # these are just the Gibbs functions
        chemical_potential_d = cv * dTdqd + self.cvd * T
        chemical_potential_v = cv * dTdqv + self.cvv * T + self.Lv0
        chemical_potential_l = cv * dTdql + self.cl * T

        # chemical_potential_i = self.ci * T
        die_d = dict()
        die_d['hs'] = T
        die_d['hqw'] = chemical_potential_v - chemical_potential_d
        die_d['h'] = enthalpy - sum(die_d[name] * state[name] / state['h'] for name in die_d.keys())

        return ie, die_d, p, qv

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

    def solve_qv_from_p(self, density, qw, p, mathlib=np, verbose=False):
        qv = 1e-3 + 0 * qw

        for _ in range(30):
            qd = 1 - qw
            ql = qw - qv
            R = qv * self.Rv + qd * self.Rd
            T = p / (density * R)
            pv = qv * self.Rv * density * T

            dTdqv = -(p / density) * (1 / R ** 2) * self.Rv
            dpvdqv = qv * self.Rv * density * dTdqv + self.Rv * density * T

            dgvdT = -self.cpv * mathlib.log(T / self.T0) - self.cpv + self.Rv * mathlib.log(pv / self.p0) - self.Lv0 / self.T0
            dgvdpv = self.Rv * T / pv

            dgldT = -self.cl * mathlib.log(T / self.T0) - self.cl

            dgvdqv = dgvdT * dTdqv + dgvdpv * dpvdqv
            dgldqv = dgldT * dTdqv

            grad = dgvdqv - dgldqv
            val = self.gibbs_vapour(T, pv, mathlib=mathlib) - self.gibbs_liquid(T, mathlib=mathlib)

            qv = qv - (val / grad)
            qv = mathlib.maximum(qv, 1e-7 + 0 * qv)

        if verbose:
            rel_update = abs((val / grad) / qv)
            print('Max relative last update:', rel_update.max())
            print('Max Gibbs error:', abs(val).max())

        return mathlib.minimum(qv, qw)

    def solve_qv_from_entropy(self, density, qw, entropy, mathlib=np, iters=10, qv=None, verbose=False, tol=1e-10):

        if qv is None:
            qv = 1e-3 + 0 * qw
            iters = 40

        logdensity = mathlib.log(density)

        for _ in range(iters):
            qd = 1 - qw
            ql = qw - qv
            R = qv * self.Rv + qd * self.Rd
            cv = qd * self.cvd + qv * self.cvv + ql * self.cl

            # majority of time not from logs....? wtf
            # logh = mathlib.log(density)
            # logqv = mathlib.log(qv)

            logqv = mathlib.log(qv)
            logT = (1 / cv) * (entropy + R * logdensity + qd * self.Rd * mathlib.log(self.Rd * qd) + qv * self.Rv * (logqv + np.log(self.Rv)) - qv * self.c0 - ql * self.c1)
            dlogTdqv = (1 / cv) * (self.Rv * logdensity + self.Rv * (logqv + np.log(self.Rv)) + self.Rv - self.c0 + self.c1)
            dlogTdqv += -(1 / cv) * logT * (self.cvv - self.cl)

            # logT = (1 / cv) * (entropy + R * logdensity + qd * self.Rd * mathlib.log(self.Rd * qd) + qv * self.Rv * mathlib.log(self.Rv * qv) - qv * self.c0 - ql * self.c1)
            # dlogTdqv = (1 / cv) * (self.Rv * logdensity + self.Rv * mathlib.log(self.Rv * qv) + self.Rv - self.c0 + self.c1)
            # dlogTdqv += -(1 / cv) * logT * (self.cvv - self.cl)

            T = mathlib.exp(logT)
            p = R * density * T
            pv = qv * self.Rv * density * T
            logpv = logqv + np.log(self.Rv) + logdensity + logT

            dTdqv = dlogTdqv * T

            dpvdqv = qv * self.Rv * density * dTdqv + self.Rv * density * T

            # dgvdT = -self.cpv * mathlib.log(T / self.T0) - self.cpv + self.Rv * mathlib.log(pv / self.p0) - self.Lv0 / self.T0
            dgvdT = -self.cpv * (logT - np.log(self.T0)) - self.cpv + self.Rv * (logpv - np.log(self.p0)) - self.Lv0 / self.T0
            dgvdpv = self.Rv * T / pv

            dgldT = -self.cl * (logT - np.log(self.T0)) - self.cl

            dgvdqv = dgvdT * dTdqv + dgvdpv * dpvdqv
            dgldqv = dgldT * dTdqv

            grad = dgvdqv - dgldqv

            val = -self.cpv * T * (logT - np.log(self.T0)) + self.Rv * T * (logpv - np.log(self.p0)) + self.Lv0 * (1 - T / self.T0)
            val -= -self.cl * T * (logT - np.log(self.T0))

            # val = self.gibbs_vapour(T, pv, mathlib=mathlib) - self.gibbs_liquid(T, mathlib=mathlib)

            qv = qv - (val / grad)
            qv = mathlib.maximum(qv, 1e-7 + 0 * qv)
            rel_update = abs((val / grad) / qv).max()
            if rel_update < tol:
                break

        self.gibbs_error = abs(val).max()
        if verbose:
            rel_update = abs((val / grad) / qv)
            print('Max relative last update:', rel_update.max())
            print('Max Gibbs error:', self.gibbs_error)

        return mathlib.minimum(qv, qw)

    def solve_qv_from_enthalpy(self, enthalpy, qw, entropy, mathlib=np, iters=20, qv=None, verbose=False):

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

            logT = mathlib.log(T)
            logqv = mathlib.log(qv)

            logdensity = (1 / R) * (cv * logT - entropy - qd * self.Rd * mathlib.log(self.Rd * qd) - qv * self.Rv * (logqv + np.log(self.Rv)) + qv * self.c0 + ql * self.c1)
            density = mathlib.exp(logdensity)

            densitydT = (1 / R) * cv / T

            ddensitydqv = (1 / R) * ((self.cvv - self.cl) * logT - self.Rv * mathlib.log(self.Rv * qv) - self.Rv + self.c0 - self.c1) * density
            ddensitydqv += -(1 / R) * self.Rv * logdensity * density
            ddensitydqv += densitydT * dTdqv

            # logT = (1 / cv) * (entropy + R * logdensity + qd * self.Rd * mathlib.log(self.Rd * qd) + qv * self.Rv * mathlib.log(self.Rv * qv) - qv * self.c0 - ql * self.c1)
            # dlogTdqv = (1 / cv) * (self.Rv * logdensity + self.Rv * mathlib.log(self.Rv * qv) + self.Rv - self.c0 + self.c1)
            # dlogTdqv += -(1 / cv) * logT * (self.cvv - self.cl)

            p = R * density * T
            pv = qv * self.Rv * density * T
            logpv = logqv + np.log(self.Rv) + logdensity + logT

            dpvdqv = qv * self.Rv * density * dTdqv + self.Rv * density * T + qv * self.Rv * ddensitydqv * T

            # dgvdT = -self.cpv * mathlib.log(T / self.T0) - self.cpv + self.Rv * mathlib.log(pv / self.p0) - self.Lv0 / self.T0
            dgvdT = -self.cpv * (logT - np.log(self.T0)) - self.cpv + self.Rv * (logpv - np.log(self.p0)) - self.Lv0 / self.T0
            dgvdpv = self.Rv * T / pv

            dgldT = -self.cl * (logT - np.log(self.T0)) - self.cl

            dgvdqv = dgvdT * dTdqv + dgvdpv * dpvdqv
            dgldqv = dgldT * dTdqv

            grad = dgvdqv - dgldqv

            val = -self.cpv * T * (logT - np.log(self.T0)) + self.Rv * T * (logpv - np.log(self.p0)) + self.Lv0 * (1 - T / self.T0)
            val -= -self.cl * T * (logT - np.log(self.T0))

            # val = self.gibbs_vapour(T, pv, mathlib=mathlib) - self.gibbs_liquid(T, mathlib=mathlib)

            qv = qv - (val / grad)
            rel_update = abs((val / grad) / qv).max()
            qv = mathlib.maximum(qv, 1e-7)
            # if rel_update < 1e-10:
            #     break

        rel_update = abs((val / grad) / qv)
        # print('Max relative last update:', rel_update.max())
        if verbose:
            rel_update = abs((val / grad) / qv)
            print('Max relative last update:', rel_update.max())
            print('Max Gibbs error:', abs(val).max())

        return mathlib.minimum(qv, qw)

    def get_moist_pt(self, s=None, qw=None, mathlib=np):
        if s is None:
            s = self.state['hs'] / self.state['h']
        if qw is None:
            qw = self.state['hqw'] / self.state['h']

        qd = 1 - qw

        tmp = s + qd * self.Rd * mathlib.log(self.p0_ex) - qw * self.c1
        logpt = tmp / (qd * self.cpd + qw * self.cl)
        return mathlib.exp(logpt)

    def moist_pt2entropy(self, pt, qw, mathlib=np):
        qd = 1 - qw
        logpt = mathlib.log(pt)
        tmp = logpt * (qd * self.cpd + qw * self.cl)
        s = tmp - qd * self.Rd * mathlib.log(self.p0_ex) + qw * self.c1

        return s

    def save_restarts(self, fp_prefix):
        for name, tnsr in self.state.items():
            np.save(f"{fp_prefix}_{name}.npy", tnsr.numpy())

        for name, vals in self.diagnostics.items():
            arr = np.array(vals)
            np.save(f"{fp_prefix}_{name}.npy", arr)

    def load_restarts(self, fp_prefix):
        names = ['u', 'v', 'h', 'hs', 'hqw']
        data = (np.load(f"{fp_prefix}_{name}.npy") for name in names)
        self.set_initial_condition(*data)

        for name in self.diagnostics.keys():
            arr = np.load(f"{fp_prefix}_{name}.npy")
            self.diagnostics[name] = list(arr)
