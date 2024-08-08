module three_phase_thermo

implicit none

contains

subroutine solve_fractions_from_entropy(&
    qv, ql, qi, T, mu, ind, density, s, qw, n, &
    Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, ci, &
    T0, logT0, p0, logp0, Lf0, Ls0, c0, c1, c2 &
    )

    ! arguments
    real(8), intent(inout) :: qv(:), ql(:), qi(:), T(:), mu(:), ind(:)
    real(8), intent(in) :: density(:), s(:), qw(:)
    integer, intent(in) :: n
    real(8), intent(in) :: Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, ci
    real(8), intent(in) :: T0, logT0, p0, logp0, Lf0, Ls0, c0, c1, c2

    ! local variables
    integer :: i
    do i = 1, n
        call solve_fractions_from_entropy_point(&
            qv(i), ql(i), qi(i), T(i), mu(i), ind(i), density(i), s(i), qw(i), &
            Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, ci, &
            T0, logT0, p0, logp0, Lf0, Ls0, c0, c1, c2 &
        )
    end do

end subroutine solve_fractions_from_entropy


subroutine solve_fractions_from_entropy_point(&
    qv_out, ql_out, qi_out, T, mu, ind, density, s, qw, &
    Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, ci, &
    T0, logT0, p0, logp0, Lf0, Ls0, c0, c1, c2 &
    )

    ! arguments
    real(8), intent(inout) :: qv_out, ql_out, qi_out, T, mu, ind
    real(8), intent(in) :: density, s, qw
    real(8), intent(in) :: Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, ci
    real(8), intent(in) :: T0, logT0, p0, logp0, Lf0, Ls0, c0, c1, c2

    ! local variables
    real(8) :: qv, ql, qi, logqv
    real(8) :: qd, logqd, logdensity, cv, R
    real(8) :: logT, pv, logpv, dlogTdqv, dlogTdql
    real(8) :: dlogTdqi, dTdqv, dTdql, dTdqi
    real(8) :: sa, sv, sc, sl, si, cvlogT
    real(8) :: gibbs_v, gibbs_l, gibbs_i, gibbs_d

    integer :: i

    ind = 0.0

    logdensity = log(density)
    qd = 1 - qw
    logqd = log(qd)

    ! check triple point
    qv = p0 / (T0 * Rv * density)

    sa = cvd * logT0 - Rd * (logqd + logdensity + logRd)
    sv = cvv * logT0 - Rv * log(qv * density) + c0
    sc = s - qd * sa - qv * sv

    sl = cl * logT0 + c1
    si = ci * logT0 + c2

    ql = (sc - si * (qw - qv)) / (sl - si)
    qi = (qw - qv) - ql

    if ((ql >= 0.0) .and. (qi >= 0)) then
        qv_out = qv
        ql_out = ql
        qi_out = qi
        T = T0
        gibbs_v = 0.0
        gibbs_d = cpd * T - T * cvd * logT0 + Rd * T * (logqd + logdensity + logRd)
        mu = gibbs_v - gibbs_d
        ind = 1.0
        return
    end if

    ! check vapour only
    qv = qw
    ql = 0.0
    qi = 0.0
    logqv = log(qv)

    R = qv * Rv + qd * Rd
    cv = qd * cvd + qv * cvv + ql * cl + qi * ci

    cvlogT = s + R * logdensity + qd * Rd * (logqd + logRd) + qv * Rv * logqv
    cvlogT = cvlogT - qv * c0 - ql * c1 - qi * c2
    logT = (1 / cv) * cvlogT
    T = exp(logT)

    pv = qv * Rv * density * T
    logpv = logqv + logRv + logdensity + logT

    gibbs_v = -cpv * T * (logT - logT0) + Rv * T * (logpv - logp0) + Ls0 * (1 - T / T0)
    gibbs_l = -cl * T * (logT - logT0) + Lf0 * (1 - T / T0)
    gibbs_i = -ci * T * (logT - logT0)

    if ((gibbs_v <= gibbs_l) .and. (gibbs_v <= gibbs_i)) then
        qv_out = qv
        ql_out = ql
        qi_out = qi
        gibbs_d = cpd * T - T * cvd * logT + Rd * T * (logqd + logdensity + logRd)
        mu = gibbs_v - gibbs_d
        ind = 2.0
        return
    end if

    if (qi_out > 0.0) then

        qv = qw * qv_out / (qv_out + qi_out + ql_out)
        ql = 0.0
        qi = qw - qv

        call solve_vapour_ice_fractions(qv, qi, T, mu, ind, density, s, qw, logdensity, logqd, &
            Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, ci, &
            T0, logT0, p0, logp0, Lf0, Ls0, c0, c1, c2)

        if (ind > 0) then
            qv_out = qv
            ql_out = ql
            qi_out = qi
            return
        else
            qv = qw
            ql = 0.0
            qi = 0.0
            call solve_vapour_liquid_fractions(qv, ql, T, mu, ind, density, s, qw, logdensity, logqd, &
                Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, ci, &
                T0, logT0, p0, logp0, Lf0, Ls0, c0, c1, c2)
            if (ind > 0) then
                qv_out = qv
                ql_out = 0.0
                qi_out = qi
            end if
        end if
    else
!        qv = qw * qv_out / (qv_out + ql_out)
!        ql = qw - qv
        qv = qw * qv_out / (qv_out + qi_out + ql_out)
        ql = qw - qv
        qi = 0.0

        call solve_vapour_liquid_fractions(qv, ql, T, mu, ind, density, s, qw, logdensity, logqd, &
            Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, ci, &
            T0, logT0, p0, logp0, Lf0, Ls0, c0, c1, c2)

        if (ind > 0) then
            qv_out = qv
            ql_out = ql
            qi_out = 0.0
            return
        else
            qv = qw
            qi = 0.0
            call solve_vapour_ice_fractions(qv, qi, T, mu, ind, density, s, qw, logdensity, logqd, &
                Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, ci, &
                T0, logT0, p0, logp0, Lf0, Ls0, c0, c1, c2)
            if (ind > 0) then
                qv_out = qv
                ql_out = 0.0
                qi_out = qi
            end if
        end if
    end if

end subroutine solve_fractions_from_entropy_point


subroutine solve_vapour_liquid_fractions(&
    qv_out, ql_out, T, mu, ind, density, s, qw, logdensity, logqd, &
    Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, ci, &
    T0, logT0, p0, logp0, Lf0, Ls0, c0, c1, c2 &
    )

    ! arguments
    real(8), intent(inout) :: qv_out, ql_out, T, mu, ind
    real(8), intent(in) :: density, s, qw, logdensity, logqd
    real(8), intent(in) :: Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, ci
    real(8), intent(in) :: T0, logT0, p0, logp0, Lf0, Ls0, c0, c1, c2

    ! local variables
    real(8) :: qv, ql, qi, logqv
    real(8) :: qd, cv, R
    real(8) :: logT, pv, logpv, dlogTdqv, dlogTdql
    real(8) :: dlogTdqi, dTdqv, dTdql, dTdqi
    real(8) :: cvlogT
    real(8) :: gibbs_v, gibbs_l, gibbs_i, gibbs_d
    real(8) :: dgibbs_vdqv, dgibbs_vdql, dgibbs_vdqi
    real(8) :: dgibbs_ldqv, dgibbs_ldql, dgibbs_ldqi
    real(8) :: dgibbs_idqv, dgibbs_idql, dgibbs_idqi
    real(8) :: dgibbs_vdT, dgibbs_vdpv
    real(8) :: dgibbs_ldT, dgibbs_ldpv
    real(8) :: dgibbs_idT, dgibbs_idpv
    real(8) :: dpvdqv, dpvdql, dpvdqi
    real(8) :: dvaldqv, val, update

    integer :: i
    logical :: is_solved

    is_solved = .false.

    qd = 1 - qw
    qv = qv_out
    ql = qw - qv
    qi = 0.0

    do i = 1, 10

        logqv = log(qv)

        R = qv * Rv + qd * Rd
        cv = qd * cvd + qv * cvv + ql * cl + qi * ci

        ! calculate temperature
        cvlogT = s + R * logdensity + qd * Rd * (logqd + logRd) + qv * Rv * logqv
        cvlogT = cvlogT - qv * c0 - ql * c1 - qi * c2
        logT = (1 / cv) * cvlogT

        T = exp(logT)

        pv = qv * Rv * density * T
        logpv = logqv + logRv + logdensity + logT

        ! gradients of T and pv w.r.t. fractions
        dlogTdqv = (1 / cv) * (Rv * logdensity + Rv * logqv + Rv - c0)
        dlogTdqv = dlogTdqv - (1 / cv) * logT * (cvv)

        dlogTdql = (1 / cv) * (-c1)
        dlogTdql = dlogTdql - (1 / cv) * logT * (cl)

        dlogTdqi = (1 / cv) * (-c2)
        dlogTdqi = dlogTdqi - (1 / cv) * logT * (ci)

        dTdqv = dlogTdqv * T
        dTdql = dlogTdql * T
        dTdqi = dlogTdqi * T

        dpvdqv = Rv * density * T + qv * Rv * density * dTdqv
        dpvdql = qv * Rv * density * dTdql
        dpvdqi = qv * Rv * density * dTdqi

        ! gibbs potentials
        gibbs_v = -cpv * T * (logT - logT0) + Rv * T * (logpv - logp0) + Ls0 * (1 - T / T0)
        gibbs_l = -cl * T * (logT - logT0) + Lf0 * (1 - T / T0)
        gibbs_i = -ci * T * (logT - logT0)

        ! gradient of gibbs w.r.t. T and pv
        dgibbs_vdT = -cpv * (logT - logT0) - cpv + Rv * (logpv - logp0) - Ls0 / T0
        dgibbs_ldT = -cl * (logT - logT0) - cl - Lf0 / T0
        dgibbs_idT = -ci * (logT - logT0) - ci

        dgibbs_vdpv = Rv * T / pv

        ! gradient of gibbs w.r.t. moisture fracs
        dgibbs_vdqv = dgibbs_vdT * dTdqv + dgibbs_vdpv * dpvdqv
        dgibbs_ldqv = dgibbs_ldT * dTdqv
        dgibbs_idqv = dgibbs_idT * dTdqv

        dgibbs_vdql = dgibbs_vdT * dTdql + dgibbs_vdpv * dpvdql
        dgibbs_ldql = dgibbs_ldT * dTdql
        dgibbs_idql = dgibbs_idT * dTdql

        dgibbs_vdqi = dgibbs_vdT * dTdqi + dgibbs_vdpv * dpvdqi
        dgibbs_ldqi = dgibbs_ldT * dTdqi
        dgibbs_idqi = dgibbs_idT * dTdqi

        ! newton step
        val = (gibbs_v - gibbs_l)
        dvaldqv = (dgibbs_vdqv - dgibbs_ldqv) - (dgibbs_vdql - dgibbs_ldql)
        update = -val / dvaldqv

        if ((T > T0) .and. (abs(update) < 1e-10)) then
            qv_out = qv
            ql_out = ql
            gibbs_d = cpd * T - T * cvd * logT + Rd * T * (logqd + logdensity + logRd)
            mu = gibbs_v - gibbs_d
            ind = 3.0
            return
        end if

        qv = qv + update
        qv = max(1e-15, qv)
        ql = qw - qv

    end do

!    if ((T >= T0)) then
!        qv_out = qv
!        ql_out = ql
!        ind = 3.0
!        return
!    end if

end subroutine solve_vapour_liquid_fractions


subroutine solve_vapour_ice_fractions(&
    qv_out, qi_out, T, mu, ind, density, s, qw, logdensity, logqd, &
    Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, ci, &
    T0, logT0, p0, logp0, Lf0, Ls0, c0, c1, c2 &
    )

    ! arguments
    real(8), intent(inout) :: qv_out, qi_out, T, mu, ind
    real(8), intent(in) :: density, s, qw, logdensity, logqd
    real(8), intent(in) :: Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, ci
    real(8), intent(in) :: T0, logT0, p0, logp0, Lf0, Ls0, c0, c1, c2

    ! local variables
    real(8) :: qv, ql, qi, logqv
    real(8) :: qd, cv, R
    real(8) :: logT, pv, logpv, dlogTdqv, dlogTdql
    real(8) :: dlogTdqi, dTdqv, dTdql, dTdqi
    real(8) :: cvlogT
    real(8) :: gibbs_v, gibbs_l, gibbs_i, gibbs_d
    real(8) :: dgibbs_vdqv, dgibbs_vdql, dgibbs_vdqi
    real(8) :: dgibbs_ldqv, dgibbs_ldql, dgibbs_ldqi
    real(8) :: dgibbs_idqv, dgibbs_idql, dgibbs_idqi
    real(8) :: dgibbs_vdT, dgibbs_vdpv
    real(8) :: dgibbs_ldT, dgibbs_ldpv
    real(8) :: dgibbs_idT, dgibbs_idpv
    real(8) :: dpvdqv, dpvdql, dpvdqi
    real(8) :: dvaldqv, val, update

    integer :: i
    logical :: is_solved

    is_solved = .false.

    qd = 1 - qw
    qv = qv_out
    ql = 0.0
    qi = qw - qv

    do i = 1, 10

        logqv = log(qv)

        R = qv * Rv + qd * Rd
        cv = qd * cvd + qv * cvv + ql * cl + qi * ci

        ! calculate temperature
        cvlogT = s + R * logdensity + qd * Rd * (logqd + logRd) + qv * Rv * logqv
        cvlogT = cvlogT - qv * c0 - ql * c1 - qi * c2
        logT = (1 / cv) * cvlogT

        T = exp(logT)

        pv = qv * Rv * density * T
        logpv = logqv + logRv + logdensity + logT

        ! gradients of T and pv w.r.t. fractions
        dlogTdqv = (1 / cv) * (Rv * logdensity + Rv * logqv + Rv - c0)
        dlogTdqv = dlogTdqv - (1 / cv) * logT * (cvv)

        dlogTdql = (1 / cv) * (-c1)
        dlogTdql = dlogTdql - (1 / cv) * logT * (cl)

        dlogTdqi = (1 / cv) * (-c2)
        dlogTdqi = dlogTdqi - (1 / cv) * logT * (ci)

        dTdqv = dlogTdqv * T
        dTdql = dlogTdql * T
        dTdqi = dlogTdqi * T

        dpvdqv = Rv * density * T + qv * Rv * density * dTdqv
        dpvdql = qv * Rv * density * dTdql
        dpvdqi = qv * Rv * density * dTdqi

        ! gibbs potentials
        gibbs_v = -cpv * T * (logT - logT0) + Rv * T * (logpv - logp0) + Ls0 * (1 - T / T0)
        gibbs_l = -cl * T * (logT - logT0) + Lf0 * (1 - T / T0)
        gibbs_i = -ci * T * (logT - logT0)

        ! gradient of gibbs w.r.t. T and pv
        dgibbs_vdT = -cpv * (logT - logT0) - cpv + Rv * (logpv - logp0) - Ls0 / T0
        dgibbs_ldT = -cl * (logT - logT0) - cl - Lf0 / T0
        dgibbs_idT = -ci * (logT - logT0) - ci

        dgibbs_vdpv = Rv * T / pv

        ! gradient of gibbs w.r.t. moisture fracs
        dgibbs_vdqv = dgibbs_vdT * dTdqv + dgibbs_vdpv * dpvdqv
        dgibbs_ldqv = dgibbs_ldT * dTdqv
        dgibbs_idqv = dgibbs_idT * dTdqv

        dgibbs_vdql = dgibbs_vdT * dTdql + dgibbs_vdpv * dpvdql
        dgibbs_ldql = dgibbs_ldT * dTdql
        dgibbs_idql = dgibbs_idT * dTdql

        dgibbs_vdqi = dgibbs_vdT * dTdqi + dgibbs_vdpv * dpvdqi
        dgibbs_ldqi = dgibbs_ldT * dTdqi
        dgibbs_idqi = dgibbs_idT * dTdqi

        ! newton step
        val = (gibbs_v - gibbs_i)
        dvaldqv = (dgibbs_vdqv - dgibbs_idqv) - (dgibbs_vdqi - dgibbs_idqi)
        update = -val / dvaldqv

        if ((T <= T0) .and. (abs(update) < 1e-10)) then
            qv_out = qv
            qi_out = qi
            gibbs_d = cpd * T - T * cvd * logT + Rd * T * (logqd + logdensity + logRd)
            mu = gibbs_v - gibbs_d
            ind = 4.0
            return
        end if

        qv = qv + update
        qv = max(1e-15, qv)
        qi = qw - qv

    end do

end subroutine solve_vapour_ice_fractions

end module three_phase_thermo