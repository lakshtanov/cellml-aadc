#!/usr/bin/env python3
"""
3-Compartment cardiovascular model — backend-agnostic.

This is the SAME model math previously hard-wired to AADC in cvs3_aadc_python.py,
but every conditional / transcendental / constant now goes through a backend
object `B` (see backends.py). The exact same source therefore runs under AADC
(operator-overloading tape) or CasADI (`ca.if_else` symbolic graph) with no
changes — which is the whole point of the fair comparison.

Integrator: semi-implicit Euler with diagonal damping
    st[i] += dt * rates[i] / (1 + dt * lam[i])
followed by a [0, 1] clamp on the valve-state (zeta) variables.
"""

# ========== Model Parameters ==========
class Params:
    R_pvn = 1333000.0
    C_pvn = 0.0000000060015
    I_pvn = 0.000001
    q_C_init_pvn = 0.0001
    q_us_0_pvn = 0.0
    delta_q_us_pvn = 0.0
    u_ext_pvn = 0.0
    delta_C_pvn = 0.0
    R_par = 10664000.0
    C_par = 3.09077e-10
    I_par = 0.000001
    u_0_par = 1463.0
    u_ext_par = 0.0
    R_aortic = 1000000.0
    C_aortic = 0.000000012028  # calibration param
    I_aortic = 10000.0
    u_0_aortic = 13300.0
    u_ext_aortic = 0.0
    R_t_sys = 110000000.0
    C_t_sys = 0.0000001
    q_us_sys = 0.00245
    q_init_sys = 0.00245
    u_ext_sys = 0.0
    R_venous = 1114600.0
    C_venous = 0.000001
    I_venous = 0.01
    q_C_init_venous = 0.0013
    q_us_0_venous = 0.0
    delta_q_us_venous = 0.0
    u_ext_venous = 0.0
    delta_C_venous = 0.0
    rho = 1050.0
    T_period = 1.0
    q_ra_us = 0.000004
    q_rv_us = 0.00001
    q_la_us = 0.000004
    q_lv_us = 0.000005
    q_ra_init = 0.000004
    q_rv_init = 0.00001
    q_la_init = 0.000004
    q_lv_init = 0.002  # calibration param
    t_ac = 0.17; t_ar = 0.17; t_astart = 0.8
    t_vc = 0.30; t_vr = 0.15; t_vstart = 0.0
    E_ra_A = 7998000.0; E_ra_B = 9331000.0
    E_rv_A = 73315000.0; E_rv_B = 6665000.0
    E_la_A = 9331000.0; E_la_B = 11997000.0
    E_lv_A = 366575000.0  # calibration param
    E_lv_B = 10664000.0   # calibration param
    k_vo_trv = 0.3; k_vo_puv = 0.2; k_vo_miv = 0.3; k_vo_aov = 0.04
    k_vc_trv = 0.4; k_vc_puv = 0.2; k_vc_miv = 0.4; k_vc_aov = 0.04
    m_rg_trv = 0.0; m_rg_puv = 0.0; m_rg_miv = 0.0; m_rg_aov = 0.0
    m_st_trv = 1.0; m_st_puv = 1.0; m_st_miv = 1.0; m_st_aov = 1.0
    l_eff = 0.01
    a_nn_trv = 0.0009; a_nn_puv = 0.0004; a_nn_miv = 0.0006; a_nn_aov = 0.000314
    eps_1 = 0.07; eps_2 = 0.02; eps_m4 = 1e-14; eps_m2 = 1e-14
    I_t_sys = 1e-6

# State indices
V_PVN, V_PAR, QC_PVN, QC_PAR, V_PUV = 0, 1, 2, 3, 4
CHI_A, CHI_V, S_HEART, ZETA_TRV, ZETA_PUV = 5, 6, 7, 8, 9
ZETA_MIV, ZETA_AOV, V_TRV, V_MIV, V_AOV = 10, 11, 12, 13, 14
Q_RA, Q_RV, Q_LA, Q_LV, V_VENOUS = 15, 16, 17, 18, 19
QCD_AORTIC, QC_AORTIC, V_AORTIC, V_SYS, VT_SYS = 20, 21, 22, 23, 24
Q_SYS, QC_VENOUS = 25, 26
N_STATES = 27

ZETA_INDICES = (ZETA_TRV, ZETA_PUV, ZETA_MIV, ZETA_AOV)

PI = 3.14159265358979


def compute_rates_and_damping(B, st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B):
    """Compute rates and diagonal damping coefficients (backend-agnostic)."""
    rates = [None] * N_STATES
    lam = [0.0] * N_STATES

    r_v_pvn = 0.01 / p.C_pvn
    r_v_par = 0.01 / p.C_par
    r_v_aortic = 0.01 / C_aortic
    r_v_sys = 0.01 / p.C_t_sys
    r_v_venous = 0.01 / p.C_venous

    q_us_wcont_pvn = p.q_us_0_pvn * (1.0 - p.delta_q_us_pvn)
    c_wcont_pvn = p.C_pvn * (1.0 - p.delta_C_pvn)
    t_astart_norm = p.t_astart / p.T_period
    t_vstart_norm = p.t_vstart / p.T_period
    q_us_wcont_venous = p.q_us_0_venous * (1.0 - p.delta_q_us_venous)
    c_wcont_venous = p.C_venous * (1.0 - p.delta_C_venous)

    # Pulmonary venous
    q_c_pvn = st[QC_PVN] + p.q_us_0_pvn - q_us_wcont_pvn
    u_c_pvn = q_c_pvn / c_wcont_pvn
    u_pvn = u_c_pvn + p.u_ext_pvn + r_v_pvn * (st[V_PAR] - st[V_PVN])

    # Atrial activation
    chi_afloor = st[CHI_A] - B.floor(st[CHI_A])
    chi_afloor_final = B.iif(chi_afloor <= 0.5, chi_afloor * 2.0, 0.0)
    e_a = 0.5 * (1.0 - B.cos(2.0 * PI * chi_afloor_final))

    u_la = (e_a * p.E_la_A + p.E_la_B) * (st[Q_LA] - p.q_la_us)
    rates[V_PVN] = (u_pvn - u_la - p.R_pvn * st[V_PVN]) / p.I_pvn
    lam[V_PVN] = (p.R_pvn + r_v_pvn) / p.I_pvn
    rates[QC_PVN] = st[V_PAR] - st[V_PVN]

    # Pulmonary arterial
    u_c_par = st[QC_PAR] / p.C_par
    u_par = p.u_0_par + u_c_par + p.u_ext_par + r_v_par * (st[V_PUV] - st[V_PAR])
    rates[V_PAR] = (u_par - u_pvn - p.R_par * st[V_PAR]) / p.I_par
    lam[V_PAR] = (p.R_par + r_v_par + r_v_pvn) / p.I_par
    rates[QC_PAR] = st[V_PUV] - st[V_PAR]

    # Heart timing
    mt = st[S_HEART] - B.floor(st[S_HEART])
    chi_af = chi_afloor

    # Atrial activation rate (simplified)
    r1a = 0.25 / p.t_ac
    r2a = 0.25 / p.t_ar
    r3a = 0.5 / (p.T_period - p.t_ac - p.t_ar)
    trig_a = B.iif(mt >= t_astart_norm,
                   B.iif(mt <= t_astart_norm + p.eps_1,
                         B.iif(chi_af <= 0.25, r1a, 0.0), 0.0), 0.0)
    ongoing_a = B.iif(chi_af > p.eps_2, B.iif(chi_af <= 0.25, r1a, 0.0), 0.0)
    relax_a = B.iif(chi_af >= 0.25, B.iif(chi_af < 0.5, r2a, 0.0), 0.0)
    diast_a = B.iif(chi_af >= 0.5, r3a, 0.0)
    rates[CHI_A] = trig_a + ongoing_a + relax_a + diast_a

    # Ventricular activation
    chi_vfloor = st[CHI_V] - B.floor(st[CHI_V])
    chi_vfloor_final = B.iif(chi_vfloor <= 0.5, chi_vfloor * 2.0, 0.0)
    e_v = 0.5 * (1.0 - B.cos(2.0 * PI * chi_vfloor_final))

    r1v = 0.25 / p.t_vc
    r2v = 0.25 / p.t_vr
    r3v = 0.5 / (p.T_period - p.t_vc - p.t_vr)
    trig_v = B.iif(mt >= t_vstart_norm,
                   B.iif(mt <= t_vstart_norm + p.eps_1,
                         B.iif(chi_vfloor <= 0.25, r1v, 0.0), 0.0), 0.0)
    ongoing_v = B.iif(chi_vfloor > p.eps_2, B.iif(chi_vfloor <= 0.25, r1v, 0.0), 0.0)
    relax_v = B.iif(chi_vfloor >= 0.25, B.iif(chi_vfloor < 0.5, r2v, 0.0), 0.0)
    diast_v = B.iif(chi_vfloor >= 0.5, r3v, 0.0)
    rates[CHI_V] = trig_v + ongoing_v + relax_v + diast_v
    rates[S_HEART] = 1.0 / p.T_period

    # Chamber pressures
    u_rv = (e_v * p.E_rv_A + p.E_rv_B) * (st[Q_RV] - p.q_rv_us)
    u_ra = (e_a * p.E_ra_A + p.E_ra_B) * (st[Q_RA] - p.q_ra_us)
    u_lv = (e_v * E_lv_A + E_lv_B) * (st[Q_LV] - p.q_lv_us)

    # Aortic
    u_c_aortic = st[QC_AORTIC] / (C_aortic / 2.0)
    u_aortic = p.u_0_aortic + u_c_aortic + p.u_ext_aortic + \
               2.0 * r_v_aortic * (st[V_AOV] - st[V_AORTIC])

    # Valve zeta dynamics with damping
    def do_zeta(idx, u_up, u_down, k_vo, k_vc):
        du = u_up - u_down
        du_abs = B.iif(du >= 0.0, du, -du)
        rates[idx] = B.iif(u_up >= u_down,
                           (1.0 - st[idx]) * k_vo * du,
                           st[idx] * k_vc * du)
        lam[idx] = B.iif(u_up >= u_down, k_vo * du_abs, k_vc * du_abs)

    do_zeta(ZETA_TRV, u_ra, u_rv, p.k_vo_trv, p.k_vc_trv)
    do_zeta(ZETA_PUV, u_rv, u_par, p.k_vo_puv, p.k_vc_puv)
    do_zeta(ZETA_MIV, u_la, u_lv, p.k_vo_miv, p.k_vc_miv)
    do_zeta(ZETA_AOV, u_lv, u_aortic, p.k_vo_aov, p.k_vc_aov)

    # Valve flows
    def do_valve_flow(v_idx, zeta_idx, m_st, a_nn, m_rg, u_up, u_down):
        zeta = B.iif(st[zeta_idx] >= 0.0, st[zeta_idx], 0.0)
        a_eff = (m_st * a_nn - m_rg * a_nn) * zeta + m_rg * a_nn
        l = p.rho * p.l_eff / (a_eff + p.eps_m2)
        b = p.rho / (2.0 * a_eff * a_eff + p.eps_m4)
        v = st[v_idx]
        v_fabs = B.iif(v >= 0.0, v, -v)
        rates[v_idx] = (-b * v * v_fabs + u_up - u_down) / l
        lam[v_idx] = 2.0 * b * v_fabs / l

    do_valve_flow(V_TRV, ZETA_TRV, p.m_st_trv, p.a_nn_trv, p.m_rg_trv, u_ra, u_rv)
    do_valve_flow(V_PUV, ZETA_PUV, p.m_st_puv, p.a_nn_puv, p.m_rg_puv, u_rv, u_par)
    do_valve_flow(V_MIV, ZETA_MIV, p.m_st_miv, p.a_nn_miv, p.m_rg_miv, u_la, u_lv)
    do_valve_flow(V_AOV, ZETA_AOV, p.m_st_aov, p.a_nn_aov, p.m_rg_aov, u_lv, u_aortic)

    # Chamber volumes
    rates[Q_RA] = st[V_VENOUS] - st[V_TRV]
    rates[Q_RV] = st[V_TRV] - st[V_PUV]
    rates[Q_LA] = st[V_PVN] - st[V_MIV]
    rates[Q_LV] = st[V_MIV] - st[V_AOV]

    # Aortic root
    u_c_d_aortic = st[QCD_AORTIC] / (C_aortic / 2.0)
    u_d_aortic = p.u_0_aortic + u_c_d_aortic + p.u_ext_aortic + \
                 2.0 * r_v_aortic * (st[V_AORTIC] - st[V_SYS])
    rates[V_AORTIC] = (u_aortic - u_d_aortic - p.R_aortic * st[V_AORTIC]) / p.I_aortic
    lam[V_AORTIC] = (p.R_aortic + 4.0 * r_v_aortic) / p.I_aortic
    rates[QC_AORTIC] = st[V_AOV] - st[V_AORTIC]
    rates[QCD_AORTIC] = st[V_AORTIC] - st[V_SYS]

    # Systemic
    rates[Q_SYS] = st[V_SYS] - st[VT_SYS]
    u_c_sys = (st[Q_SYS] - p.q_us_sys) / p.C_t_sys
    u_sys = u_c_sys + p.u_ext_sys + r_v_sys * (st[V_SYS] - st[VT_SYS])
    rates[V_SYS] = (u_d_aortic - u_sys - st[V_SYS] * p.R_t_sys / 2.0) / p.I_t_sys
    lam[V_SYS] = (p.R_t_sys / 2.0 + 2.0 * r_v_aortic + r_v_sys) / p.I_t_sys

    # Venous
    q_c_venous = st[QC_VENOUS] + p.q_us_0_venous - q_us_wcont_venous
    u_c_venous = q_c_venous / c_wcont_venous
    v_venous_in = st[VT_SYS]
    u_venous = u_c_venous + p.u_ext_venous + r_v_venous * (v_venous_in - st[V_VENOUS])
    rates[VT_SYS] = (u_sys - u_venous - st[VT_SYS] * p.R_t_sys / 2.0) / p.I_t_sys
    lam[VT_SYS] = (p.R_t_sys / 2.0 + r_v_sys + r_v_venous) / p.I_t_sys
    rates[V_VENOUS] = (u_venous - u_ra - p.R_venous * st[V_VENOUS]) / p.I_venous
    lam[V_VENOUS] = (p.R_venous + r_v_venous) / p.I_venous
    rates[QC_VENOUS] = v_venous_in - st[V_VENOUS]

    return rates, lam


def initial_state(B, p, q_lv_init):
    """Initial state vector (length N_STATES). q_lv_init may be a backend input."""
    st = [B.const(0.0) for _ in range(N_STATES)]
    st[QC_PVN] = B.const(p.q_C_init_pvn)
    st[Q_RA] = B.const(p.q_ra_init)
    st[Q_RV] = B.const(p.q_rv_init)
    st[Q_LA] = B.const(p.q_la_init)
    st[Q_LV] = q_lv_init
    st[QC_VENOUS] = B.const(p.q_C_init_venous)
    st[Q_SYS] = B.const(p.q_init_sys)
    return st


def euler_step(B, st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B, dt):
    """One semi-implicit Euler step with diagonal damping + zeta clamp."""
    rates, lam = compute_rates_and_damping(B, st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B)
    new = [st[i] + dt * rates[i] / (1.0 + dt * lam[i]) for i in range(N_STATES)]
    for z in ZETA_INDICES:
        new[z] = B.iif(new[z] >= 0.0, new[z], B.const(0.0))
        new[z] = B.iif(new[z] <= 1.0, new[z], B.const(1.0))
    return new


def simulate(B, q_lv_init, C_aortic, E_lv_A, E_lv_B,
             pre_steps=2000, sim_steps=200, dt=0.01):
    """Run the model by Python-looping euler_step; return cost = Q_LV_final**2.

    Used by the AADC driver (recording). The CasADI driver builds the identical
    step as a ca.Function and unrolls it with mapaccum instead (see cvs3_casadi.py)
    — same integrator, just not re-traced every step.
    """
    p = Params()
    st = initial_state(B, p, q_lv_init)
    for _ in range(pre_steps + sim_steps):
        st = euler_step(B, st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B, dt)
    return st[Q_LV] * st[Q_LV]
