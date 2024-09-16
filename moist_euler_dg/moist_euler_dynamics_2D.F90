module fmoist_euler_2D_dynamics

implicit none

contains

! apply jacobian
subroutine solve(&
        u, w, h, s, q, T, mu, p, ie, &
        dudt, dwdt, dhdt, dsdt, dqdt, &
        D, wz, Ja, &
        grad_xi_2, grad_xi_dot_zeta, grad_zeta_2, &
        nx, nz, n, &
        a, upwind_flag, gamma &
    )
    real(8), intent(in) :: u(:), w(:), h(:), s(:), q(:), T(:), mu(:), p(:), ie(:)
    real(8), intent(inout) :: dudt(:), dwdt(:), dhdt(:), dsdt(:), dqdt(:)
    real(8), intent(in) :: D(:, :), wz, Ja(:)
    real(8), intent(in) :: grad_xi_2(:), grad_xi_dot_zeta(:), grad_zeta_2(:)
    integer :: nx, nz, n
    real(8) :: g, a, upwind_flag, gamma

    real(8) :: Gp, Gm, Tp, Tm, Fxp, Fxm, Fzp, Fzm, norm_grad_contra
    integer :: i, j, k, idx, stride, ip, im, ib

    idx = 0
    stride = nz * n * n
    do i=1,nx
        call solve_column(&
            u, w, h, s, q, T, mu, p, ie, &
            dudt, dwdt, dhdt, dsdt, dqdt, &
            D, wz, Ja, &
            grad_xi_2, grad_xi_dot_zeta, grad_zeta_2, &
            nz, n, idx, &
            a, upwind_flag, gamma &
        )

        idx = idx + stride
    end do

    do i=1,nx-1
    do j=1,nz
    do k=1,n
        im = (i - 1) * stride + (j - 1) * n * n + (n - 1) * n + k
        ip = i * stride + (j - 1) * n * n + k
        ib = ip

        call get_fluxes(&
            u(ip), w(ip), h(ip), s(ip), q(ip), T(ip), mu(ip), p(ip), ie(ip), &
            grad_xi_2(ib), grad_xi_dot_zeta(ib), grad_zeta_2(ib), &
            gamma, Gp, Fxp, Fzp &
        )

        call get_fluxes(&
            u(im), w(im), h(im), s(im), q(im), T(im), mu(im), p(im), ie(im), &
            grad_xi_2(ib), grad_xi_dot_zeta(ib), grad_zeta_2(ib), &
            gamma, Gm, Fxm, Fzm &
        )

        norm_grad_contra = sqrt(grad_xi_2(ib))

        call boundary_fluxes(&
            dudt(ip), dwdt(ip), &
            dhdt(ip), dsdt(ip), dqdt(ip), &
            u(ip), w(ip), h(ip), s(ip), q(ip), T(ip), mu(ip), p(ip), ie(ip), &
            Gp, Fxp, Fzp, &
            dudt(im), dwdt(im), &
            dhdt(im), dsdt(im), dqdt(im), &
            u(im), w(im), h(im), s(im), q(im), T(im), mu(im), p(im), ie(im), &
            Gm, Fxm, Fzm, &
            norm_grad_contra, wz, a, upwind_flag, gamma  &
        )

    end do
    end do
    end do

end subroutine


subroutine solve_column(&
        u, w, h, s, q, T, mu, p, ie, &
        dudt, dwdt, dhdt, dsdt, dqdt, &
        D, wz, Ja, &
        grad_xi_2, grad_xi_dot_zeta, grad_zeta_2, &
        nz, n, idx_start, &
        a, upwind_flag, gamma &
    )
    real(8), intent(in) :: u(:), w(:), h(:), s(:), q(:), T(:), mu(:), p(:), ie(:)
    real(8), intent(inout) :: dudt(:), dwdt(:), dhdt(:), dsdt(:), dqdt(:)
    real(8), intent(in) :: D(:, :), wz, Ja(:)
    real(8), intent(in) :: grad_xi_2(:), grad_xi_dot_zeta(:), grad_zeta_2(:)
    integer :: nz, n, idx_start
    real(8) :: a, upwind_flag, gamma

    integer :: il, j, k, l, m, idx, ip, im, ib, imx, imz
    real(8) :: Fz(n, n), Fx(n, n), GG(n, n)
    real(8) :: Gp, Gm, Tp, Tm, Fxp, Fxm, Fzp, Fzm
    real(8) :: enthalpy, norm_grad_contra, normal_vel_p, normal_vel_m, c_snd
    real(8) :: dsdx, dsdz, dqdx, dqdz, vort, Jinv, divF, divsF, divqF

    idx = idx_start
    do j=1, nz
        do k=1,n
            idx = idx_start + (j - 1) * n * n + (k - 1) * n
            do l=1,n
                il = idx + l
                ! calculate fluxes
                Fz(l, k) = h(il) * (grad_xi_dot_zeta(il) * u(il) + grad_zeta_2(il) * w(il))
                Fx(l, k) = h(il) * (grad_xi_2(il) * u(il) + grad_xi_dot_zeta(il) * w(il))

                GG(l, k) = grad_xi_2(il) * u(il) ** 2 + 2 * grad_xi_dot_zeta(il) * u(il) * w(il)
                GG(l, k) = GG(l, k) + grad_zeta_2(il) * w(il) ** 2

                enthalpy = (ie(il) + p(il)) / h(il)
                GG(l, k) = 0.5 * GG(l, k) + enthalpy - T(il) * s(il) - mu(il) * q(il)
            end do
        end do
        do k=1,n
            idx = idx_start + (j - 1) * n * n + (k - 1) * n
            ! derivatives
            do l=1,n
                il = idx + l
                vort = 0.0
                Jinv = 1.0 / Ja(il)
                dsdz = 0.0
                dsdx = 0.0
                dqdz = 0.0
                dqdx = 0.0
                divF = 0.0
                divsF = 0.0
                divqF = 0.0
                do m=1, n
                    imz = idx + m
                    imx = idx - (k - 1) * n + (m - 1) * n + l

                    dwdt(il) = dwdt(il) - D(m, l) * (GG(m, k) + 0.5 * s(imz) * T(imz) + 0.5 * q(imz) * mu(imz))
                    dwdt(il) = dwdt(il) - 0.5 * s(il) * D(m, l) * T(imz) - 0.5 * q(il) * D(m, l) * mu(imz)

                    dudt(il) = dudt(il) - D(m, k) * (GG(l, m) + 0.5 * s(imx) * T(imx) + 0.5 * q(imx) * mu(imx))
                    dudt(il) = dudt(il) - 0.5 * s(il) * D(m, k) * T(imx) - 0.5 * q(il) * D(m, k) * mu(imx)

                    dsdz = dsdz + D(m, l) * s(imz)
                    dsdx = dsdx + D(m, k) * s(imx)
                    dqdz = dqdz + D(m, l) * q(imz)
                    dqdx = dqdx + D(m, k) * q(imx)

                    vort = vort + D(m, l) * u(imz) - D(m, k) * w(imx)

                    divF = divF + D(m, l) * Fz(m, k) * Ja(imz) + D(m, k) * Fx(l, m) * Ja(imx)
                    divsF = divsF + D(m, l) * s(imz) * Fz(m, k) * Ja(imz) + D(m, k) * s(imx) * Fx(l, m) * Ja(imx)
                    divqF = divqF + D(m, l) * q(imz) * Fz(m, k) * Ja(imz) + D(m, k) * q(imx) * Fx(l, m) * Ja(imx)

                end do

                divF = divF * Jinv
                divsF = divsF * Jinv
                divqF = divqF * Jinv

                dsdt(il) = dsdt(il) - 0.5 * (divsF + Fz(l, k) * dsdz + Fx(l, k) * dsdx - s(il) * divF) / h(il)
                dqdt(il) = dqdt(il) - 0.5 * (divqF + Fz(l, k) * dqdz + Fx(l, k) * dqdx - q(il) * divF) / h(il)
                dhdt(il) = dhdt(il) - divF

                dudt(il) = dudt(il) - Fz(l, k) * vort / h(il) + 0.5 * T(il) * dsdx + 0.5 * mu(il) * dqdx
                dwdt(il) = dwdt(il) + Fx(l, k) * vort / h(il) + 0.5 * T(il) * dsdz + 0.5 * mu(il) * dqdz
            end do
        end do
    end do

!!    ! interior boundaries
    do j=1,nz-1
    do k=1,n
        idx = idx_start + (j - 1) * n * n + (k-1) * n
        im = idx + n
        ip = idx + n * n + 1
        ib = ip

        call get_fluxes(&
            u(ip), w(ip), h(ip), s(ip), q(ip), T(ip), mu(ip), p(ip), ie(ip), &
            grad_xi_2(ib), grad_xi_dot_zeta(ib), grad_zeta_2(ib), &
            gamma, Gp, Fxp, Fzp &
        )

        call get_fluxes(&
            u(im), w(im), h(im), s(im), q(im), T(im), mu(im), p(im), ie(im), &
            grad_xi_2(ib), grad_xi_dot_zeta(ib), grad_zeta_2(ib), &
            gamma, Gm, Fxm, Fzm &
        )

        norm_grad_contra = sqrt(grad_zeta_2(ib))

        call boundary_fluxes(&
            dwdt(ip), dudt(ip), &
            dhdt(ip), dsdt(ip), dqdt(ip), &
            w(ip), u(ip), h(ip), s(ip), q(ip), T(ip), mu(ip), p(ip), ie(ip), &
            Gp, Fzp, Fxp, &
            dwdt(im), dudt(im), &
            dhdt(im), dsdt(im), dqdt(im), &
            w(im), u(im), h(im), s(im), q(im), T(im), mu(im), p(im), ie(im), &
            Gm, Fzm, Fxm, &
            norm_grad_contra, wz, a, upwind_flag, gamma  &
        )

    end do
    end do
!!
!    ! exterior boundaries
    do k=1,n
        ip = idx_start + (k-1) * n + 1
        Fzp = h(ip) * (grad_xi_dot_zeta(ip) * u(ip) + grad_zeta_2(ip) * w(ip))
        dhdt(ip) = dhdt(ip) - Fzp / wz
        norm_grad_contra = sqrt(grad_zeta_2(ip))

        c_snd = sqrt(gamma * p(ip) / h(ip))
        normal_vel_p = Fzp / (norm_grad_contra * h(ip))
        dwdt(ip) = dwdt(ip) - 2 * a * (c_snd + abs(normal_vel_p)) * normal_vel_p / wz
    end do

    do k=1,n
        im = idx_start + (nz - 1) * n * n + (k-1) * n + n
        Fzm = h(im) * (grad_xi_dot_zeta(im) * u(im) + grad_zeta_2(im) * w(im))
        dhdt(im) = dhdt(im) + Fzm / wz
        norm_grad_contra = sqrt(grad_zeta_2(im))

        c_snd = sqrt(gamma * p(im) / h(im))
        normal_vel_m = Fzm / (norm_grad_contra * h(im))
        dwdt(im) = dwdt(im) - 2 * a * (c_snd + abs(normal_vel_m)) * normal_vel_m / wz
    end do

end subroutine


subroutine solve_horz_boundaries(&
    u, w, h, s, q, T, mu, p, ie, &
    um, wm, hm, sm, qm, Tm, mum, pm, iem, &
    up, wp, hp, sp, qp, Tp, mup, pp, iep, &
    dudt, dwdt, dhdt, dsdt, dqdt, &
    D, wz, Ja, &
    grad_xi_2, grad_xi_dot_zeta, grad_zeta_2, &
    nx, nz, n, &
    a, upwind_flag, gamma &
)
    real(8), intent(in) :: u(:), w(:), h(:), s(:), q(:), T(:), mu(:), p(:), ie(:)
    real(8), intent(in) :: um(:), wm(:), hm(:), sm(:), qm(:), Tm(:), mum(:), pm(:), iem(:)
    real(8), intent(in) :: up(:), wp(:), hp(:), sp(:), qp(:), Tp(:), mup(:), pp(:), iep(:)
    real(8), intent(inout) :: dudt(:), dwdt(:), dhdt(:), dsdt(:), dqdt(:)
    real(8), intent(in) :: D(:, :), wz, Ja(:)
    real(8), intent(in) :: grad_xi_2(:), grad_xi_dot_zeta(:), grad_zeta_2(:)
    integer :: nx, nz, n
    real(8) :: a, upwind_flag, gamma

    real(8) :: Gp, Gm, Fxp, Fxm, Fzp, Fzm, norm_grad_contra, dummy
    integer :: i, j, k, idx, stride, ip, im, ib

    stride = nz * n * n

    i = 1
    do j=1,nz
    do k=1,n
        im = (j - 1) * n + k
        ip = (j - 1) * n * n + k
        ib = ip

        call get_fluxes(&
            u(ip), w(ip), h(ip), s(ip), q(ip), T(ip), mu(ip), p(ip), ie(ip), &
            grad_xi_2(ib), grad_xi_dot_zeta(ib), grad_zeta_2(ib), &
            gamma, Gp, Fxp, Fzp &
        )

        call get_fluxes(&
            um(im), wm(im), hm(im), sm(im), qm(im), Tm(im), mum(im), pm(im), iem(im), &
            grad_xi_2(ib), grad_xi_dot_zeta(ib), grad_zeta_2(ib), &
            gamma, Gm, Fxm, Fzm &
        )

        norm_grad_contra = sqrt(grad_xi_2(ib))

        call boundary_fluxes(&
            dudt(ip), dwdt(ip), &
            dhdt(ip), dsdt(ip), dqdt(ip), &
            u(ip), w(ip), h(ip), s(ip), q(ip), T(ip), mu(ip), p(ip), ie(ip), &
            Gp, Fxp, Fzp, &
            dummy, dummy, &
            dummy, dummy, dummy, &
            um(im), wm(im), hm(im), sm(im), qm(im), Tm(im), mum(im), pm(im), iem(im), &
            Gm, Fxm, Fzm, &
            norm_grad_contra, wz, a, upwind_flag, gamma  &
        )

    end do
    end do

    i = nx
    do j=1,nz
    do k=1,n
        im = (i - 1) * stride + (j - 1) * n * n + (n - 1) * n + k
        ip = (j - 1) * n + k
        ib = im

        call get_fluxes(&
            up(ip), wp(ip), hp(ip), sp(ip), qp(ip), Tp(ip), mup(ip), pp(ip), iep(ip), &
            grad_xi_2(ib), grad_xi_dot_zeta(ib), grad_zeta_2(ib), &
            gamma, Gp, Fxp, Fzp &
        )

        call get_fluxes(&
            u(im), w(im), h(im), s(im), q(im), T(im), mu(im), p(im), ie(im), &
            grad_xi_2(ib), grad_xi_dot_zeta(ib), grad_zeta_2(ib), &
            gamma, Gm, Fxm, Fzm &
        )

        norm_grad_contra = sqrt(grad_xi_2(ib))

        call boundary_fluxes(&
            dummy, dummy, &
            dummy, dummy, dummy, &
            up(ip), wp(ip), hp(ip), sp(ip), qp(ip), Tp(ip), mup(ip), pp(ip), iep(ip), &
            Gp, Fxp, Fzp, &
            dudt(im), dwdt(im), &
            dhdt(im), dsdt(im), dqdt(im), &
            u(im), w(im), h(im), s(im), q(im), T(im), mu(im), p(im), ie(im), &
            Gm, Fxm, Fzm, &
            norm_grad_contra, wz, a, upwind_flag, gamma  &
        )

    end do
    end do

end subroutine


subroutine get_fluxes(&
    u, w, h, s, q, T, mu, p, ie, &
    grad_xi_2, grad_xi_dot_zeta, grad_zeta_2, &
    gamma, G, Fx, Fz &
)

    real(8), intent(in) :: u, w, h, s, q, T, mu, p, ie
    real(8), intent(in) :: grad_xi_2, grad_xi_dot_zeta, grad_zeta_2, gamma
    real(8), intent(inout) :: G, Fx, Fz

    real(8) :: enthalpy

    Fx = h * (grad_xi_2 * u + grad_xi_dot_zeta * w)
    Fz = h * (grad_xi_dot_zeta * u + grad_zeta_2 * w)

    G = (Fx * u + Fz * w) / h

    enthalpy = (ie + p) / h
    G = 0.5 * G + enthalpy - T * s - mu * q

end subroutine


subroutine boundary_fluxes(&
    ddt_u1p, ddt_u2p, &
    ddt_hp, ddt_sp, ddt_qp, &
    u1p, u2p, hp, sp, qp, Tp, mup, pp, iep, &
    Gp, F1p, F2p, &
    ddt_u1m, ddt_u2m, &
    ddt_hm, ddt_sm, ddt_qm, &
    u1m, u2m, hm, sm, qm, Tm, mum, pm, iem, &
    Gm, F1m, F2m, &
    norm_contra, wz, a, upwind_flag, gamma &
)

    real(8), intent(inout) :: ddt_u1p, ddt_u2p, ddt_hp, ddt_sp, ddt_qp
    real(8), intent(in) :: u1p, u2p, hp, sp, qp, Tp, mup, pp, iep, F1p, F2p, Gp
    real(8), intent(inout) :: ddt_u1m, ddt_u2m, ddt_hm, ddt_sm, ddt_qm
    real(8), intent(in) :: u1m, u2m, hm, sm, qm, Tm, mum, pm, iem, F1m, F2m, Gm
    real(8), intent(in) :: norm_contra, wz, gamma, a, upwind_flag

    real(8) :: fluxp, fluxm, num_flux, shat, qhat, F_avg
    real(8) :: normal_vel_p, normal_vel_m, cp, cm, c_snd, c_adv

    normal_vel_p = F1p / (hp * norm_contra)
    normal_vel_m = F1m / (hm * norm_contra)
    cp = sqrt(gamma * pp / hp)
    cm = sqrt(gamma * pm / hm)

    c_adv = abs(0.5 * (normal_vel_p + normal_vel_m))
    c_snd = 0.5 * (cp + cm)

    F_avg = 0.5 * (F1p + F1m) - a * (hp - hm) * (c_snd + c_adv) * norm_contra
    shat = 0.5 * (sp + sm) - upwind_flag * 0.5 * (sp - sm) * sign(real(1, 8), F_avg)
    qhat = 0.5 * (qp + qm) - upwind_flag * 0.5 * (qp - qm) * sign(real(1, 8), F_avg)

    fluxp = Gp
    fluxm = Gm
    num_flux = 0.5 * (fluxp + fluxm) - a * (normal_vel_p - normal_vel_m) * (c_snd + c_adv)
    ddt_u1p = ddt_u1p + (num_flux - fluxp) / wz
    ddt_u1m = ddt_u1m - (num_flux - fluxm) / wz

    fluxp = Tp
    fluxm = Tm
    num_flux = 0.5 * (fluxp + fluxm)
    ddt_u1p = ddt_u1p + shat * (num_flux - fluxp) / wz
    ddt_u1m = ddt_u1m - shat * (num_flux - fluxm) / wz

    fluxp = mup
    fluxm = mum
    num_flux = 0.5 * (fluxp + fluxm)
    ddt_u1p = ddt_u1p + qhat * (num_flux - fluxp) / wz
    ddt_u1m = ddt_u1m - qhat * (num_flux - fluxm) / wz

    fluxp = F1p
    fluxm = F1m
    num_flux = F_avg ! 0.5 * (F1p + F1m) - a * (hp - hm) * (c_snd + c_adv) * norm_contra
    ddt_hp = ddt_hp + (num_flux - fluxp) / wz
    ddt_hm = ddt_hm - (num_flux - fluxm) / wz

    fluxp = sp
    fluxm = sm
    num_flux = shat
    ddt_sp = ddt_sp + (F_avg / hp) * (num_flux - fluxp) / wz
    ddt_sm = ddt_sm - (F_avg / hm) * (num_flux - fluxm) / wz

    fluxp = qp
    fluxm = qm
    num_flux = qhat
    ddt_qp = ddt_qp + (F_avg / hp) * (num_flux - fluxp) / wz
    ddt_qm = ddt_qm - (F_avg / hm) * (num_flux - fluxm) / wz

    fluxp = u2p
    fluxm = u2m
    num_flux = 0.5 * (fluxm + fluxp)
    ddt_u2p = ddt_u2p + (F1p / hp) * (num_flux - fluxp) / wz
    ddt_u2m = ddt_u2m - (F1m / hm) * (num_flux - fluxm) / wz
    ddt_u1p = ddt_u1p - (F2p / hp) * (num_flux - fluxp) / wz
    ddt_u1m = ddt_u1m + (F2m / hm) * (num_flux - fluxm) / wz

    ! self.a
    num_flux = -a * c_adv * ((u1p - u1m) - (normal_vel_p - normal_vel_m) / norm_contra)
    ddt_u1p = ddt_u1p + norm_contra * num_flux / wz
    ddt_u1m = ddt_u1m - norm_contra * num_flux / wz

    num_flux = -a * c_adv * (u2p - u2m)
    ddt_u2p = ddt_u2p + norm_contra * num_flux / wz
    ddt_u2m = ddt_u2m - norm_contra * num_flux / wz

end subroutine

end module fmoist_euler_2D_dynamics