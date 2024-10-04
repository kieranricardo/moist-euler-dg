module two_phase_thermo

implicit none

contains

subroutine solve_fractions_from_entropy(&
    qv, ql, T, mu, ind, density, s, qw, n, &
    Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, &
    T0, logT0, p0, logp0, Lv0, c0, c1 &
    )

    ! arguments
    real(8), intent(inout) :: qv(:), ql(:), T(:), mu(:), ind(:)
    real(8), intent(in) :: density(:), s(:), qw(:)
    integer, intent(in) :: n
    real(8), intent(in) :: Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl
    real(8), intent(in) :: T0, logT0, p0, logp0, Lv0, c0, c1

    ! local variables
    integer :: i
    do i = 1, n
        call solve_fractions_from_entropy_point(&
            qv(i), ql(i), T(i), mu(i), ind(i), density(i), s(i), qw(i), &
            Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, &
            T0, logT0, p0, logp0, Lv0, c0, c1 &
        )
    end do

end subroutine solve_fractions_from_entropy


subroutine solve_fractions_from_entropy_point(&
    qv_out, ql_out, T, mu, ind, density, s, qw, &
    Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, &
    T0, logT0, p0, logp0, Lv0, c0, c1 &
    )

    ! arguments
    real(8), intent(inout) :: qv_out, ql_out, T, mu, ind
    real(8), intent(in) :: density, s, qw
    real(8), intent(in) :: Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl
    real(8), intent(in) :: T0, logT0, p0, logp0, Lv0, c0, c1

    ! local variables
    real(8) :: qv, ql, logqv
    real(8) :: qd, logqd, logdensity, cv, R
    real(8) :: logT, pv, logpv, dlogTdqv, dlogTdql
    real(8) :: dTdqv, dTdql
    real(8) :: sa, sv, sc, sl, si, cvlogT
    real(8) :: gibbs_v, gibbs_l, gibbs_d

    integer :: i

    ! convergence indicator
    ind = 0.0

    logdensity = log(density)
    qd = 1 - qw
    logqd = log(qd)

    ! check vapour only
    qv = qw
    ql = 0.0
    logqv = log(qv)

    R = qv * Rv + qd * Rd
    cv = qd * cvd + qv * cvv + ql * cl

    cvlogT = s + R * logdensity + qd * Rd * (logqd + logRd) + qv * Rv * (logqv + logRv)
    cvlogT = cvlogT - qv * c0 - ql * c1
    logT = (1 / cv) * cvlogT
    T = exp(logT)

    pv = qv * Rv * density * T
    logpv = logqv + logRv + logdensity + logT

    gibbs_v = -cpv * T * (logT - logT0) + Rv * T * (logpv - logp0) + Lv0 * (1 - T / T0)
    gibbs_l = -cl * T * (logT - logT0)

    if (gibbs_v <= gibbs_l)  then
        qv_out = qv
        ql_out = ql
        gibbs_d = cpd * T - T * cvd * logT + Rd * T * (logqd + logdensity + logRd)
        mu = gibbs_v - gibbs_d
        ind = 2.0
        return
    end if

    qv = qw * qv_out / (qv_out + ql_out)
    ql = qw - qv

    call solve_vapour_liquid_fractions(qv, ql, T, mu, ind, density, s, qw, logdensity, logqd, &
                Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, &
                T0, logT0, p0, logp0, Lv0, c0, c1)
    if (ind > 0) then
        qv_out = qv
        ql_out = ql
        return
    end if

end subroutine solve_fractions_from_entropy_point


subroutine solve_vapour_liquid_fractions(&
    qv_out, ql_out, T, mu, ind, density, s, qw, logdensity, logqd, &
    Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl, &
    T0, logT0, p0, logp0, Lv0, c0, c1 &
    )

    ! arguments
    real(8), intent(inout) :: qv_out, ql_out, T, mu, ind
    real(8), intent(in) :: density, s, qw, logdensity, logqd
    real(8), intent(in) :: Rd, logRd, Rv, logRv, cvd, cvv, cpv, cpd, cl
    real(8), intent(in) :: T0, logT0, p0, logp0, Lv0, c0, c1

    ! local variables
    real(8) :: qv, ql, logqv
    real(8) :: qd, cv, R
    real(8) :: logT, pv, logpv, dlogTdqv, dlogTdql
    real(8) :: dTdqv, dTdql
    real(8) :: cvlogT
    real(8) :: gibbs_v, gibbs_l, gibbs_d
    real(8) :: dgibbs_vdqv, dgibbs_vdql
    real(8) :: dgibbs_ldqv, dgibbs_ldql
    real(8) :: dgibbs_vdT, dgibbs_vdpv
    real(8) :: dgibbs_ldT, dgibbs_ldpv
    real(8) :: dpvdqv, dpvdql
    real(8) :: dvaldqv, val, update

    integer :: i
    logical :: is_solved

    is_solved = .false.

    qd = 1 - qw
    qv = qv_out
    ql = qw - qv

    do i = 1, 100

        logqv = log(qv)

        R = qv * Rv + qd * Rd
        cv = qd * cvd + qv * cvv + ql * cl

        ! calculate temperature
        cvlogT = s + R * logdensity + qd * Rd * (logqd + logRd) + qv * Rv * (logqv + logRv)
        cvlogT = cvlogT - qv * c0 - ql * c1
        logT = (1 / cv) * cvlogT

        T = exp(logT)

        pv = qv * Rv * density * T
        logpv = logqv + logRv + logdensity + logT

        ! gradients of T and pv w.r.t. fractions
        dlogTdqv = (1 / cv) * (Rv * logdensity + Rv * (logqv + logRv) + Rv - c0)
        dlogTdqv = dlogTdqv - (1 / cv) * logT * (cvv)

        dlogTdql = (1 / cv) * (-c1)
        dlogTdql = dlogTdql - (1 / cv) * logT * (cl)

        dTdqv = dlogTdqv * T
        dTdql = dlogTdql * T

        dpvdqv = Rv * density * T + qv * Rv * density * dTdqv
        dpvdql = qv * Rv * density * dTdql

        ! gibbs potentials
        gibbs_v = -cpv * T * (logT - logT0) + Rv * T * (logpv - logp0) + Lv0 * (1 - T / T0)
        gibbs_l = -cl * T * (logT - logT0)

        ! gradient of gibbs w.r.t. T and pv
        dgibbs_vdT = -cpv * (logT - logT0) - cpv + Rv * (logpv - logp0) - Lv0 / T0
        dgibbs_ldT = -cl * (logT - logT0) - cl

        dgibbs_vdpv = Rv * T / pv

        ! gradient of gibbs w.r.t. moisture fracs
        dgibbs_vdqv = dgibbs_vdT * dTdqv + dgibbs_vdpv * dpvdqv
        dgibbs_ldqv = dgibbs_ldT * dTdqv

        dgibbs_vdql = dgibbs_vdT * dTdql + dgibbs_vdpv * dpvdql
        dgibbs_ldql = dgibbs_ldT * dTdql

        ! newton step
        val = (gibbs_v - gibbs_l)
        dvaldqv = (dgibbs_vdqv - dgibbs_ldqv) - (dgibbs_vdql - dgibbs_ldql)
        update = -val / dvaldqv

        qv = qv + update
        qv = max(1e-15, qv)
        ql = qw - qv

        if ((abs(val) < 1e-7)) then

            qv_out = qv
            ql_out = ql
            gibbs_d = cpd * T - T * cvd * logT + Rd * T * (logqd + logdensity + logRd)
            mu = gibbs_v - gibbs_d
            ind = 3.0

            return
        end if

    end do

end subroutine solve_vapour_liquid_fractions


end module two_phase_thermo