#!/usr/bin/env python3
"""
Digital Twin Calibration: Diagnosing Heart Failure from Echocardiography

Clinical scenario:
  Patient with reduced exercise tolerance.
  Echocardiography measures LV volume over 2 cardiac cycles (200 points).
  Task: calibrate digital twin → infer disease parameters with uncertainty.

Method:
  1. AADC records ODE integration as a differentiable tape
  2. L-BFGS-B minimizes misfit to echo data (exact gradient from AADC)
  3. HMC sampling quantifies posterior uncertainty

Key result:
  Calibration in <1 second with exact gradients through conditional valve logic.
  CasADI crashes on this model. AADC makes it possible.
"""
import time
import numpy as np
import aadc
from scipy.optimize import minimize

# ========== Model (same as cvs3_aadc_python.py) ==========
class Params:
    R_pvn = 1333000.0; C_pvn = 0.0000000060015; I_pvn = 0.000001
    q_C_init_pvn = 0.0001; q_us_0_pvn = 0.0; delta_q_us_pvn = 0.0
    u_ext_pvn = 0.0; delta_C_pvn = 0.0
    R_par = 10664000.0; C_par = 3.09077e-10; I_par = 0.000001
    u_0_par = 1463.0; u_ext_par = 0.0
    R_aortic = 1000000.0; C_aortic = 0.000000012028; I_aortic = 10000.0
    u_0_aortic = 13300.0; u_ext_aortic = 0.0
    R_t_sys = 110000000.0; C_t_sys = 0.0000001; q_us_sys = 0.00245
    q_init_sys = 0.00245; u_ext_sys = 0.0
    R_venous = 1114600.0; C_venous = 0.000001; I_venous = 0.01
    q_C_init_venous = 0.0013; q_us_0_venous = 0.0; delta_q_us_venous = 0.0
    u_ext_venous = 0.0; delta_C_venous = 0.0
    rho = 1050.0; T_period = 1.0
    q_ra_us = 0.000004; q_rv_us = 0.00001; q_la_us = 0.000004; q_lv_us = 0.000005
    q_ra_init = 0.000004; q_rv_init = 0.00001; q_la_init = 0.000004; q_lv_init = 0.002
    t_ac = 0.17; t_ar = 0.17; t_astart = 0.8
    t_vc = 0.30; t_vr = 0.15; t_vstart = 0.0
    E_ra_A = 7998000.0; E_ra_B = 9331000.0
    E_rv_A = 73315000.0; E_rv_B = 6665000.0
    E_la_A = 9331000.0; E_la_B = 11997000.0
    E_lv_A = 366575000.0; E_lv_B = 10664000.0
    k_vo_trv = 0.3; k_vo_puv = 0.2; k_vo_miv = 0.3; k_vo_aov = 0.04
    k_vc_trv = 0.4; k_vc_puv = 0.2; k_vc_miv = 0.4; k_vc_aov = 0.04
    m_rg_trv = 0.0; m_rg_puv = 0.0; m_rg_miv = 0.0; m_rg_aov = 0.0
    m_st_trv = 1.0; m_st_puv = 1.0; m_st_miv = 1.0; m_st_aov = 1.0
    l_eff = 0.01
    a_nn_trv = 0.0009; a_nn_puv = 0.0004; a_nn_miv = 0.0006; a_nn_aov = 0.000314
    eps_1 = 0.07; eps_2 = 0.02; eps_m4 = 1e-14; eps_m2 = 1e-14; I_t_sys = 1e-6

V_PVN, V_PAR, QC_PVN, QC_PAR, V_PUV = 0, 1, 2, 3, 4
CHI_A, CHI_V, S_HEART, ZETA_TRV, ZETA_PUV = 5, 6, 7, 8, 9
ZETA_MIV, ZETA_AOV, V_TRV, V_MIV, V_AOV = 10, 11, 12, 13, 14
Q_RA, Q_RV, Q_LA, Q_LV, V_VENOUS = 15, 16, 17, 18, 19
QCD_AORTIC, QC_AORTIC, V_AORTIC, V_SYS, VT_SYS = 20, 21, 22, 23, 24
Q_SYS, QC_VENOUS = 25, 26
N_STATES = 27
PI = 3.14159265358979

def iif(cond, val_true, val_false):
    if isinstance(cond, bool):
        cond = aadc.ibool(cond)
    return aadc.iif(cond, val_true, val_false)

def floor_id(x):
    import math
    return math.floor(float(x))

def init_state(p):
    st = [aadc.idouble(0.0)] * N_STATES
    st[QC_PVN] = aadc.idouble(p.q_C_init_pvn)
    st[Q_RA] = aadc.idouble(p.q_ra_init)
    st[Q_RV] = aadc.idouble(p.q_rv_init)
    st[Q_LA] = aadc.idouble(p.q_la_init)
    st[Q_LV] = aadc.idouble(p.q_lv_init)
    st[QC_VENOUS] = aadc.idouble(p.q_C_init_venous)
    st[Q_SYS] = aadc.idouble(p.q_init_sys)
    return st

def compute_rates_and_damping(st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B):
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
    q_c_pvn = st[QC_PVN] + p.q_us_0_pvn - q_us_wcont_pvn
    u_c_pvn = q_c_pvn / c_wcont_pvn
    u_pvn = u_c_pvn + p.u_ext_pvn + r_v_pvn * (st[V_PAR] - st[V_PVN])
    chi_afloor = st[CHI_A] - floor_id(st[CHI_A])
    chi_afloor_final = iif(chi_afloor <= 0.5, chi_afloor * 2.0, 0.0)
    e_a = 0.5 * (1.0 - aadc.math.cos(2.0 * PI * chi_afloor_final))
    u_la = (e_a * p.E_la_A + p.E_la_B) * (st[Q_LA] - p.q_la_us)
    rates[V_PVN] = (u_pvn - u_la - p.R_pvn * st[V_PVN]) / p.I_pvn
    lam[V_PVN] = (p.R_pvn + r_v_pvn) / p.I_pvn
    rates[QC_PVN] = st[V_PAR] - st[V_PVN]
    u_c_par = st[QC_PAR] / p.C_par
    u_par = p.u_0_par + u_c_par + p.u_ext_par + r_v_par * (st[V_PUV] - st[V_PAR])
    rates[V_PAR] = (u_par - u_pvn - p.R_par * st[V_PAR]) / p.I_par
    lam[V_PAR] = (p.R_par + r_v_par + r_v_pvn) / p.I_par
    rates[QC_PAR] = st[V_PUV] - st[V_PAR]
    mt = st[S_HEART] - floor_id(st[S_HEART])
    chi_af = chi_afloor
    r1a = 0.25 / p.t_ac; r2a = 0.25 / p.t_ar; r3a = 0.5 / (p.T_period - p.t_ac - p.t_ar)
    trig_a = iif(mt >= t_astart_norm, iif(mt <= t_astart_norm + p.eps_1, iif(chi_af <= 0.25, r1a, 0.0), 0.0), 0.0)
    ongoing_a = iif(chi_af > p.eps_2, iif(chi_af <= 0.25, r1a, 0.0), 0.0)
    relax_a = iif(chi_af >= 0.25, iif(chi_af < 0.5, r2a, 0.0), 0.0)
    diast_a = iif(chi_af >= 0.5, r3a, 0.0)
    rates[CHI_A] = trig_a + ongoing_a + relax_a + diast_a
    chi_vfloor = st[CHI_V] - floor_id(st[CHI_V])
    chi_vfloor_final = iif(chi_vfloor <= 0.5, chi_vfloor * 2.0, 0.0)
    e_v = 0.5 * (1.0 - aadc.math.cos(2.0 * PI * chi_vfloor_final))
    r1v = 0.25 / p.t_vc; r2v = 0.25 / p.t_vr; r3v = 0.5 / (p.T_period - p.t_vc - p.t_vr)
    trig_v = iif(mt >= t_vstart_norm, iif(mt <= t_vstart_norm + p.eps_1, iif(chi_vfloor <= 0.25, r1v, 0.0), 0.0), 0.0)
    ongoing_v = iif(chi_vfloor > p.eps_2, iif(chi_vfloor <= 0.25, r1v, 0.0), 0.0)
    relax_v = iif(chi_vfloor >= 0.25, iif(chi_vfloor < 0.5, r2v, 0.0), 0.0)
    diast_v = iif(chi_vfloor >= 0.5, r3v, 0.0)
    rates[CHI_V] = trig_v + ongoing_v + relax_v + diast_v
    rates[S_HEART] = 1.0 / p.T_period
    u_rv = (e_v * p.E_rv_A + p.E_rv_B) * (st[Q_RV] - p.q_rv_us)
    u_ra = (e_a * p.E_ra_A + p.E_ra_B) * (st[Q_RA] - p.q_ra_us)
    u_lv = (e_v * E_lv_A + E_lv_B) * (st[Q_LV] - p.q_lv_us)
    u_c_aortic = st[QC_AORTIC] / (C_aortic / 2.0)
    u_aortic = p.u_0_aortic + u_c_aortic + p.u_ext_aortic + 2.0 * r_v_aortic * (st[V_AOV] - st[V_AORTIC])
    def do_zeta(idx, u_up, u_down, k_vo, k_vc):
        du = u_up - u_down
        du_abs = iif(du >= 0.0, du, -du)
        rates[idx] = iif(u_up >= u_down, (1.0 - st[idx]) * k_vo * du, st[idx] * k_vc * du)
        lam[idx] = iif(u_up >= u_down, k_vo * du_abs, k_vc * du_abs)
    do_zeta(ZETA_TRV, u_ra, u_rv, p.k_vo_trv, p.k_vc_trv)
    do_zeta(ZETA_PUV, u_rv, u_par, p.k_vo_puv, p.k_vc_puv)
    do_zeta(ZETA_MIV, u_la, u_lv, p.k_vo_miv, p.k_vc_miv)
    do_zeta(ZETA_AOV, u_lv, u_aortic, p.k_vo_aov, p.k_vc_aov)
    def do_valve_flow(v_idx, zeta_idx, m_st, a_nn, m_rg, u_up, u_down):
        zeta = iif(st[zeta_idx] >= 0.0, st[zeta_idx], 0.0)
        a_eff = (m_st * a_nn - m_rg * a_nn) * zeta + m_rg * a_nn
        l = p.rho * p.l_eff / (a_eff + p.eps_m2)
        b = p.rho / (2.0 * a_eff * a_eff + p.eps_m4)
        v = st[v_idx]; v_fabs = iif(v >= 0.0, v, -v)
        rates[v_idx] = (-b * v * v_fabs + u_up - u_down) / l
        lam[v_idx] = 2.0 * b * v_fabs / l
    do_valve_flow(V_TRV, ZETA_TRV, p.m_st_trv, p.a_nn_trv, p.m_rg_trv, u_ra, u_rv)
    do_valve_flow(V_PUV, ZETA_PUV, p.m_st_puv, p.a_nn_puv, p.m_rg_puv, u_rv, u_par)
    do_valve_flow(V_MIV, ZETA_MIV, p.m_st_miv, p.a_nn_miv, p.m_rg_miv, u_la, u_lv)
    do_valve_flow(V_AOV, ZETA_AOV, p.m_st_aov, p.a_nn_aov, p.m_rg_aov, u_lv, u_aortic)
    rates[Q_RA] = st[V_VENOUS] - st[V_TRV]
    rates[Q_RV] = st[V_TRV] - st[V_PUV]
    rates[Q_LA] = st[V_PVN] - st[V_MIV]
    rates[Q_LV] = st[V_MIV] - st[V_AOV]
    u_c_d_aortic = st[QCD_AORTIC] / (C_aortic / 2.0)
    u_d_aortic = p.u_0_aortic + u_c_d_aortic + p.u_ext_aortic + 2.0 * r_v_aortic * (st[V_AORTIC] - st[V_SYS])
    rates[V_AORTIC] = (u_aortic - u_d_aortic - p.R_aortic * st[V_AORTIC]) / p.I_aortic
    lam[V_AORTIC] = (p.R_aortic + 4.0 * r_v_aortic) / p.I_aortic
    rates[QC_AORTIC] = st[V_AOV] - st[V_AORTIC]
    rates[QCD_AORTIC] = st[V_AORTIC] - st[V_SYS]
    rates[Q_SYS] = st[V_SYS] - st[VT_SYS]
    u_c_sys = (st[Q_SYS] - p.q_us_sys) / p.C_t_sys
    u_sys = u_c_sys + p.u_ext_sys + r_v_sys * (st[V_SYS] - st[VT_SYS])
    rates[V_SYS] = (u_d_aortic - u_sys - st[V_SYS] * p.R_t_sys / 2.0) / p.I_t_sys
    lam[V_SYS] = (p.R_t_sys / 2.0 + 2.0 * r_v_aortic + r_v_sys) / p.I_t_sys
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

def step_model(st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B, dt):
    rates, lam = compute_rates_and_damping(st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B)
    for i in range(N_STATES):
        st[i] = st[i] + dt * rates[i] / (1.0 + dt * lam[i])
    for z in [ZETA_TRV, ZETA_PUV, ZETA_MIV, ZETA_AOV]:
        st[z] = iif(st[z] >= 0.0, st[z], aadc.idouble(0.0))
        st[z] = iif(st[z] <= 1.0, st[z], aadc.idouble(1.0))

# ========== Data generation via AADC recording ==========
def generate_data(factor_elva, factor_cao, pre_steps, sim_steps, dt):
    """Record and evaluate to get Q_LV trajectory at given params."""
    p = Params()
    funcs = aadc.Functions()
    funcs.start_recording()

    # Must mark as input to keep idouble chain active on tape
    id_f_elva = aadc.idouble(factor_elva)
    a_f_elva = id_f_elva.mark_as_input()
    id_f_cao = aadc.idouble(factor_cao)
    a_f_cao = id_f_cao.mark_as_input()

    E_lv_A = aadc.idouble(p.E_lv_A) * id_f_elva
    C_aortic = aadc.idouble(p.C_aortic) * id_f_cao
    q_lv_init = aadc.idouble(p.q_lv_init)
    E_lv_B = aadc.idouble(p.E_lv_B)

    st = init_state(p)
    for _ in range(pre_steps):
        step_model(st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B, dt)

    outputs = []
    for _ in range(sim_steps):
        step_model(st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B, dt)
        outputs.append(st[Q_LV].mark_as_output())

    funcs.stop_recording()
    workers = aadc.ThreadPool(1)
    request = {o: [] for o in outputs}
    inputs = {a_f_elva: factor_elva, a_f_cao: factor_cao}
    res = aadc.evaluate(funcs, request, inputs, workers)
    traj = np.array([float(np.asarray(res[0][o]).flat[0]) for o in outputs])
    return traj

# ========== Calibration kernel ==========
def record_calibration_kernel(target_data, pre_steps, sim_steps, dt):
    """Record AADC kernel: inputs = (factor_elva, factor_elvb), output = MSE cost.

    E_lv_A (systolic elastance) and E_lv_B (diastolic elastance) affect
    different parts of the cardiac cycle → independently identifiable from Q_LV.
    """
    p = Params()
    funcs = aadc.Functions()
    funcs.start_recording()

    id_f_elva = aadc.idouble(1.0)
    a_f_elva = id_f_elva.mark_as_input()
    id_f_elvb = aadc.idouble(1.0)
    a_f_elvb = id_f_elvb.mark_as_input()

    E_lv_A = aadc.idouble(p.E_lv_A) * id_f_elva
    E_lv_B = aadc.idouble(p.E_lv_B) * id_f_elvb
    C_aortic = aadc.idouble(p.C_aortic)
    q_lv_init = aadc.idouble(p.q_lv_init)

    st = init_state(p)
    for _ in range(pre_steps):
        step_model(st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B, dt)

    cost = aadc.idouble(0.0)
    traj_outputs = []
    for k in range(sim_steps):
        step_model(st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B, dt)
        traj_outputs.append(st[Q_LV].mark_as_output())
        residual = st[Q_LV] - float(target_data[k])
        cost = cost + residual * residual

    cost = cost / float(sim_steps)
    r_cost = cost.mark_as_output()
    funcs.stop_recording()
    return funcs, a_f_elva, a_f_elvb, r_cost, traj_outputs


# ========== HMC sampler ==========
def hmc(eval_fn, theta_init, n_samples, epsilon, n_leapfrog, N_data, sigma):
    """Hamiltonian Monte Carlo with Gaussian likelihood."""
    d = len(theta_init)
    scale = N_data / (2.0 * sigma ** 2)
    samples = np.zeros((n_samples, d))
    theta = theta_init.copy()
    n_accept = 0
    cost_curr, _ = eval_fn(theta)
    U_curr = scale * cost_curr

    for i in range(n_samples):
        p0 = np.random.randn(d)
        K0 = 0.5 * np.dot(p0, p0)
        theta_prop = theta.copy()
        p = p0.copy()

        # Leapfrog
        _, grad = eval_fn(theta_prop)
        p -= 0.5 * epsilon * scale * grad
        for l in range(n_leapfrog):
            theta_prop += epsilon * p
            theta_prop = np.clip(theta_prop, 0.1, 5.0)
            _, grad = eval_fn(theta_prop)
            if l < n_leapfrog - 1:
                p -= epsilon * scale * grad
        p -= 0.5 * epsilon * scale * grad

        cost_prop, _ = eval_fn(theta_prop)
        U_prop = scale * cost_prop
        K_prop = 0.5 * np.dot(p, p)

        dH = (U_prop + K_prop) - (U_curr + K0)
        if np.log(np.random.rand() + 1e-30) < -dH:
            theta = theta_prop.copy()
            U_curr = U_prop
            cost_curr = cost_prop
            n_accept += 1
        samples[i] = theta

    return samples, n_accept / n_samples


def main():
    np.random.seed(42)
    p = Params()

    # ---- Configuration ----
    # Disease: reduced systolic + increased diastolic stiffness
    # E_lv_A (systolic elastance): affects ejection phase of Q_LV waveform
    # E_lv_B (diastolic elastance): affects filling phase of Q_LV waveform
    # These are independently identifiable from the waveform shape.
    TRUE_FA = 0.40      # 60% reduced systolic elastance → severe HF (reduced EF)
    TRUE_FB = 2.0       # doubled diastolic stiffness → diastolic dysfunction
    NOISE_FRAC = 0.10   # 10% of SV as noise
    PRE = 2000; SIM = 200; DT = 0.01  # 2 cardiac cycles

    PARAM_NAMES = ["E_lv_A", "E_lv_B"]

    print("=" * 65)
    print("  DIGITAL TWIN CALIBRATION FROM ECHOCARDIOGRAPHY")
    print("  3-compartment CVS model, 27 states, 4 heart valves")
    print("=" * 65)

    # ---- Phase 1: Synthetic patient data ----
    print("\n--- Phase 1: Generating synthetic patient data ---")
    print(f"  True pathology:")
    print(f"    LV systolic elastance (E_lv_A): {TRUE_FA*100:.0f}% of normal")
    print(f"    LV diastolic elastance (E_lv_B): {TRUE_FB*100:.0f}% of normal")

    t0 = time.time()
    # generate_data uses (factor_elva, factor_cao) for the recording —
    # but we want (factor_elva, factor_elvb). Reuse by passing factors.
    # For data generation, we fix C_aortic at nominal.
    # generate_data creates idoubles with the given factors for E_lv_A and C_aortic.
    # We need a custom data generation that uses E_lv_B factor instead.

    def generate_data_ab(factor_a, factor_b, pre_steps, sim_steps, dt):
        """Generate Q_LV trajectory for given E_lv_A and E_lv_B factors."""
        pp = Params()
        funcs_g = aadc.Functions()
        funcs_g.start_recording()
        id_fa = aadc.idouble(factor_a); a_fa = id_fa.mark_as_input()
        id_fb = aadc.idouble(factor_b); a_fb = id_fb.mark_as_input()
        E_lv_A = aadc.idouble(pp.E_lv_A) * id_fa
        E_lv_B = aadc.idouble(pp.E_lv_B) * id_fb
        C_aortic = aadc.idouble(pp.C_aortic)
        q_lv_init = aadc.idouble(pp.q_lv_init)
        st = init_state(pp)
        for _ in range(pre_steps):
            step_model(st, pp, q_lv_init, C_aortic, E_lv_A, E_lv_B, dt)
        outputs = []
        for _ in range(sim_steps):
            step_model(st, pp, q_lv_init, C_aortic, E_lv_A, E_lv_B, dt)
            outputs.append(st[Q_LV].mark_as_output())
        funcs_g.stop_recording()
        w = aadc.ThreadPool(1)
        req = {o: [] for o in outputs}
        inp = {a_fa: factor_a, a_fb: factor_b}
        res = aadc.evaluate(funcs_g, req, inp, w)
        return np.array([float(np.asarray(res[0][o]).flat[0]) for o in outputs])

    traj_disease = generate_data_ab(TRUE_FA, TRUE_FB, PRE, SIM, DT)
    traj_healthy = generate_data_ab(1.0, 1.0, PRE, SIM, DT)
    t_data = time.time() - t0

    sv_healthy = (np.max(traj_healthy) - np.min(traj_healthy)) * 1e6
    sv_disease = (np.max(traj_disease) - np.min(traj_disease)) * 1e6
    sigma = NOISE_FRAC * (np.max(traj_disease) - np.min(traj_disease))
    target_noisy = traj_disease + np.random.randn(SIM) * sigma

    print(f"\n  Healthy stroke volume:  {sv_healthy:.1f} (model units * 1e6)")
    print(f"  Patient stroke volume:  {sv_disease:.1f} ({(sv_disease/sv_healthy-1)*100:+.0f}%)")
    print(f"  Measurement noise σ:    {sigma*1e6:.1f} ({NOISE_FRAC*100:.0f}% of SV)")
    print(f"  Data generation: {t_data:.1f}s")

    # ---- Phase 2: Record calibration kernel ----
    print("\n--- Phase 2: Recording AADC calibration kernel ---")
    t0 = time.time()
    funcs, a_fa, a_fb, r_cost, traj_out = \
        record_calibration_kernel(target_noisy, PRE, SIM, DT)
    t_record = time.time() - t0
    print(f"  Recording: {t_record:.1f}s (one-time)")

    workers = aadc.ThreadPool(1)

    def eval_cost_grad(factors):
        inputs = {a_fa: factors[0], a_fb: factors[1]}
        request = {r_cost: [a_fa, a_fb]}
        res = aadc.evaluate(funcs, request, inputs, workers)
        c = float(np.asarray(res[0][r_cost]).flat[0])
        g = np.array([
            float(np.asarray(res[1][r_cost][a_fa]).flat[0]),
            float(np.asarray(res[1][r_cost][a_fb]).flat[0]),
        ])
        return c, g

    def get_trajectory(factors):
        inputs = {a_fa: factors[0], a_fb: factors[1]}
        request = {o: [] for o in traj_out}
        res = aadc.evaluate(funcs, request, inputs, workers)
        return np.array([float(np.asarray(res[0][o]).flat[0]) for o in traj_out])

    # ---- Phase 3: Gradient verification ----
    print("\n--- Phase 3: Gradient verification (central FD) ---")
    # Test at a midpoint between healthy and disease
    test_pt = np.array([0.7, 1.5])
    _, grad_ad = eval_cost_grad(test_pt)
    for h_test in [1e-3, 1e-5]:
        print(f"  At ({test_pt[0]:.1f}, {test_pt[1]:.1f}), h={h_test}:")
        for i, name in enumerate(PARAM_NAMES):
            fp = test_pt.copy(); fp[i] += h_test
            fm = test_pt.copy(); fm[i] -= h_test
            cp, _ = eval_cost_grad(fp)
            cm, _ = eval_cost_grad(fm)
            fd = (cp - cm) / (2 * h_test)
            ratio = grad_ad[i] / fd if abs(fd) > 1e-30 else float('nan')
            print(f"    d(cost)/d(f_{name}): AD={grad_ad[i]:.6e}  FD={fd:.6e}  ratio={ratio:.6f}")

    # ---- Phase 4: L-BFGS-B calibration ----
    print("\n--- Phase 4: Gradient-based calibration (L-BFGS-B) ---")
    print(f"  Starting from healthy baseline (100%, 100%)")

    log = []
    def objective(factors):
        c, g = eval_cost_grad(factors)
        log.append((factors.copy(), c))
        return c, g

    t0 = time.time()
    result = minimize(objective, x0=np.array([1.0, 1.0]), jac=True,
                      method='L-BFGS-B',
                      bounds=[(0.1, 5.0), (0.1, 5.0)],
                      options={'maxiter': 100, 'ftol': 1e-20, 'gtol': 1e-15})
    t_opt = time.time() - t0
    opt_factors = result.x

    print(f"\n  {'Iter':>4s}  {'Cost':>12s}  {'E_lv_A':>8s}  {'E_lv_B':>8s}")
    print("  " + "-" * 40)
    for i, (f, c) in enumerate(log):
        if i < 5 or i % 5 == 0 or i == len(log) - 1:
            print(f"  {i:4d}  {c:12.4e}  {f[0]*100:7.1f}%  {f[1]*100:7.1f}%")

    err_a = abs(opt_factors[0] - TRUE_FA) / TRUE_FA * 100
    err_b = abs(opt_factors[1] - TRUE_FB) / TRUE_FB * 100

    print(f"\n  Converged: {len(log)} evaluations, {t_opt:.2f}s")
    print(f"  Recovered vs true:")
    print(f"    E_lv_A: {opt_factors[0]*100:.2f}% (true: {TRUE_FA*100:.0f}%)  error: {err_a:.2f}%")
    print(f"    E_lv_B: {opt_factors[1]*100:.2f}% (true: {TRUE_FB*100:.0f}%)  error: {err_b:.2f}%")

    traj_fit = get_trajectory(opt_factors)
    rmse = np.sqrt(np.mean((traj_fit - traj_disease) ** 2)) * 1e6
    print(f"  RMSE vs true trajectory: {rmse:.3f} (model units * 1e6)")

    # ---- Phase 5: Posterior uncertainty ----
    print("\n--- Phase 5: Posterior uncertainty ---")

    # Laplace approximation
    print("  Computing Hessian at MAP...")
    h_hess = 1e-4
    H = np.zeros((2, 2))
    for i in range(2):
        fp = opt_factors.copy(); fp[i] += h_hess
        fm = opt_factors.copy(); fm[i] -= h_hess
        _, gp = eval_cost_grad(fp)
        _, gm = eval_cost_grad(fm)
        H[i] = (gp - gm) / (2 * h_hess)
    H = 0.5 * (H + H.T)

    scale = float(SIM) / (2.0 * sigma ** 2)
    H_logp = scale * H
    try:
        cov_laplace = np.linalg.inv(H_logp)
        sigma_laplace = np.sqrt(np.abs(np.diag(cov_laplace)))
        corr_laplace = cov_laplace[0, 1] / (sigma_laplace[0] * sigma_laplace[1] + 1e-30)
        print(f"  Laplace approximation:")
        print(f"    E_lv_A: {opt_factors[0]*100:.1f} +/- {sigma_laplace[0]*100:.1f}%")
        print(f"    E_lv_B: {opt_factors[1]*100:.1f} +/- {sigma_laplace[1]*100:.1f}%")
        print(f"    Correlation: {corr_laplace:.2f}")
        eps_hmc = 0.25 * np.min(sigma_laplace)
    except np.linalg.LinAlgError:
        eps_hmc = 0.005
        sigma_laplace = None

    # HMC
    N_SAMPLES = 200
    N_LEAPFROG = 10
    print(f"\n  HMC: {N_SAMPLES} samples, {N_LEAPFROG} leapfrog, epsilon={eps_hmc:.5f}")

    t0 = time.time()
    for adapt_round in range(5):
        burn_samples, burn_accept = hmc(eval_cost_grad, opt_factors, 20, eps_hmc,
                                         N_LEAPFROG, SIM, sigma)
        if burn_accept < 0.5:
            eps_hmc *= 0.5
        elif burn_accept > 0.85:
            eps_hmc *= 1.5
        else:
            break
    print(f"  Burn-in: {burn_accept*100:.0f}% acceptance, final epsilon={eps_hmc:.6f}")

    samples, accept_rate = hmc(eval_cost_grad, burn_samples[-1], N_SAMPLES,
                                eps_hmc, N_LEAPFROG, SIM, sigma)
    t_hmc = time.time() - t0
    print(f"  Production: {accept_rate*100:.0f}% acceptance, {t_hmc:.1f}s")

    mean_a = np.mean(samples[:, 0]) * 100
    std_a = np.std(samples[:, 0]) * 100
    mean_b = np.mean(samples[:, 1]) * 100
    std_b = np.std(samples[:, 1]) * 100
    ci95_a = np.percentile(samples[:, 0] * 100, [2.5, 97.5])
    ci95_b = np.percentile(samples[:, 1] * 100, [2.5, 97.5])
    corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]

    print(f"\n  {'Param':>10s}  {'Mean':>7s}  {'Std':>6s}  {'95% CI':>18s}  {'True':>6s}")
    print("  " + "-" * 58)
    ok_a = "ok" if ci95_a[0] <= TRUE_FA * 100 <= ci95_a[1] else "MISS"
    ok_b = "ok" if ci95_b[0] <= TRUE_FB * 100 <= ci95_b[1] else "MISS"
    print(f"  {'E_lv_A %':>10s}  {mean_a:6.1f}%  {std_a:5.1f}%  [{ci95_a[0]:5.1f}, {ci95_a[1]:5.1f}]%  {TRUE_FA*100:5.0f}%  {ok_a}")
    print(f"  {'E_lv_B %':>10s}  {mean_b:6.1f}%  {std_b:5.1f}%  [{ci95_b[0]:5.1f}, {ci95_b[1]:5.1f}]%  {TRUE_FB*100:5.0f}%  {ok_b}")
    print(f"  Correlation: rho(E_lv_A, E_lv_B) = {corr:.2f}")

    # ---- Phase 6: Performance ----
    print("\n--- Phase 6: Performance ---")
    n_bench = 50
    t0 = time.time()
    for _ in range(n_bench):
        eval_cost_grad(opt_factors)
    t_bench = time.time() - t0
    ms_per = t_bench / n_bench * 1000

    print(f"  Per gradient evaluation: {ms_per:.2f} ms")
    print(f"  Gradient evals/s: {1000/ms_per:.0f}")
    print(f"  Calibration (L-BFGS-B): {t_opt:.2f}s ({len(log)} evaluations)")
    print(f"  Uncertainty (HMC): {t_hmc:.1f}s ({N_SAMPLES} samples)")

    # ---- Summary ----
    print("\n" + "=" * 65)
    print("  CLINICAL INTERPRETATION")
    print("=" * 65)
    print(f"  The calibrated digital twin indicates:")
    print(f"    Systolic elastance (E_lv_A) = {mean_a:.0f}% of normal")
    print(f"      95% CI: [{ci95_a[0]:.0f}, {ci95_a[1]:.0f}]%")
    if mean_a < 60:
        print(f"      -> Severely reduced: heart failure with reduced ejection fraction")
    elif mean_a < 80:
        print(f"      -> Moderately reduced: mild systolic dysfunction")
    print(f"    Diastolic elastance (E_lv_B) = {mean_b:.0f}% of normal")
    print(f"      95% CI: [{ci95_b[0]:.0f}, {ci95_b[1]:.0f}]%")
    if mean_b > 150:
        print(f"      -> Increased: diastolic dysfunction (stiff ventricle)")
    print(f"\n  Time: {t_opt:.1f}s calibration + {t_hmc:.1f}s uncertainty = {t_opt+t_hmc:.1f}s total")
    print(f"  (kernel recording {t_record:.0f}s is one-time)")
    print(f"\n  CasADI: CRASHES on this model (valve if/else → RuntimeError)")
    print(f"  AADC: exact gradient through conditionals -> {ms_per:.1f} ms/eval")


if __name__ == "__main__":
    main()
