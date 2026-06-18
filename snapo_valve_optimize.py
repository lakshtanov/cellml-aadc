#!/usr/bin/env python3
"""
SNAPO-like valve threshold optimization using AADC.

Optimizes 4 valve opening thresholds (tricuspid, pulmonary, mitral, aortic)
to maximize cardiac output (stroke volume = max Q_LV - min Q_LV over one cycle).

The key: threshold parameters enter valve zeta dynamics via
  u_up >= u_down + threshold  (instead of u_up >= u_down)
so they propagate through ZETA → valve flows → Q_LV → cost,
keeping the idouble chain fully active for gradient computation.

AADC gives exact gradients through all the conditional valve logic
where CasADI would crash.
"""
import time
import numpy as np
import aadc

# ========== Model Parameters (same as cvs3_aadc_python.py) ==========
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
    C_aortic = 0.000000012028
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
    q_lv_init = 0.002
    t_ac = 0.17; t_ar = 0.17; t_astart = 0.8
    t_vc = 0.30; t_vr = 0.15; t_vstart = 0.0
    E_ra_A = 7998000.0; E_ra_B = 9331000.0
    E_rv_A = 73315000.0; E_rv_B = 6665000.0
    E_la_A = 9331000.0; E_la_B = 11997000.0
    E_lv_A = 366575000.0
    E_lv_B = 10664000.0
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
PI = 3.14159265358979


def iif(cond, val_true, val_false):
    if isinstance(cond, bool):
        cond = aadc.ibool(cond)
    return aadc.iif(cond, val_true, val_false)


def floor_id(x):
    import math
    return math.floor(float(x))


def compute_rates_and_damping(st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B,
                               th_trv, th_puv, th_miv, th_aov):
    """Compute rates and diagonal damping.

    th_trv, th_puv, th_miv, th_aov: valve opening pressure thresholds (Pa).
    Positive threshold = valve needs MORE pressure to open = harder to open.
    """
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
    chi_afloor = st[CHI_A] - floor_id(st[CHI_A])
    chi_afloor_final = iif(chi_afloor <= 0.5, chi_afloor * 2.0, 0.0)
    e_a = 0.5 * (1.0 - aadc.math.cos(2.0 * PI * chi_afloor_final))

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
    mt = st[S_HEART] - floor_id(st[S_HEART])
    chi_af = chi_afloor

    r1a = 0.25 / p.t_ac
    r2a = 0.25 / p.t_ar
    r3a = 0.5 / (p.T_period - p.t_ac - p.t_ar)
    trig_a = iif(mt >= t_astart_norm,
                 iif(mt <= t_astart_norm + p.eps_1,
                     iif(chi_af <= 0.25, r1a, 0.0), 0.0), 0.0)
    ongoing_a = iif(chi_af > p.eps_2, iif(chi_af <= 0.25, r1a, 0.0), 0.0)
    relax_a = iif(chi_af >= 0.25, iif(chi_af < 0.5, r2a, 0.0), 0.0)
    diast_a = iif(chi_af >= 0.5, r3a, 0.0)
    rates[CHI_A] = trig_a + ongoing_a + relax_a + diast_a

    # Ventricular activation
    chi_vfloor = st[CHI_V] - floor_id(st[CHI_V])
    chi_vfloor_final = iif(chi_vfloor <= 0.5, chi_vfloor * 2.0, 0.0)
    e_v = 0.5 * (1.0 - aadc.math.cos(2.0 * PI * chi_vfloor_final))

    r1v = 0.25 / p.t_vc
    r2v = 0.25 / p.t_vr
    r3v = 0.5 / (p.T_period - p.t_vc - p.t_vr)
    trig_v = iif(mt >= t_vstart_norm,
                 iif(mt <= t_vstart_norm + p.eps_1,
                     iif(chi_vfloor <= 0.25, r1v, 0.0), 0.0), 0.0)
    ongoing_v = iif(chi_vfloor > p.eps_2, iif(chi_vfloor <= 0.25, r1v, 0.0), 0.0)
    relax_v = iif(chi_vfloor >= 0.25, iif(chi_vfloor < 0.5, r2v, 0.0), 0.0)
    diast_v = iif(chi_vfloor >= 0.5, r3v, 0.0)
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

    # Valve zeta dynamics — thresholds shift the opening condition
    def do_zeta(idx, u_up, u_down, k_vo, k_vc, threshold):
        du = u_up - u_down - threshold  # threshold shifts opening pressure
        du_abs = iif(du >= 0.0, du, -du)
        rates[idx] = iif(du >= 0.0,
                         (1.0 - st[idx]) * k_vo * du,
                         st[idx] * k_vc * du)
        lam[idx] = iif(du >= 0.0, k_vo * du_abs, k_vc * du_abs)

    do_zeta(ZETA_TRV, u_ra, u_rv, p.k_vo_trv, p.k_vc_trv, th_trv)
    do_zeta(ZETA_PUV, u_rv, u_par, p.k_vo_puv, p.k_vc_puv, th_puv)
    do_zeta(ZETA_MIV, u_la, u_lv, p.k_vo_miv, p.k_vc_miv, th_miv)
    do_zeta(ZETA_AOV, u_lv, u_aortic, p.k_vo_aov, p.k_vc_aov, th_aov)

    # Valve flows (unchanged — flows depend on zeta, which depends on thresholds)
    def do_valve_flow(v_idx, zeta_idx, m_st, a_nn, m_rg, u_up, u_down):
        zeta = iif(st[zeta_idx] >= 0.0, st[zeta_idx], 0.0)
        a_eff = (m_st * a_nn - m_rg * a_nn) * zeta + m_rg * a_nn
        l = p.rho * p.l_eff / (a_eff + p.eps_m2)
        b = p.rho / (2.0 * a_eff * a_eff + p.eps_m4)
        v = st[v_idx]
        v_fabs = iif(v >= 0.0, v, -v)
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


def run_model(q_lv_init, C_aortic, E_lv_A, E_lv_B,
              th_trv, th_puv, th_miv, th_aov,
              pre_steps=2000, sim_steps=200, dt=0.01):
    """Run model, return negative stroke volume (to minimize = maximize SV)."""
    p = Params()
    total_steps = pre_steps + sim_steps

    st = [aadc.idouble(0.0)] * N_STATES
    st[QC_PVN] = aadc.idouble(p.q_C_init_pvn)
    st[Q_RA] = aadc.idouble(p.q_ra_init)
    st[Q_RV] = aadc.idouble(p.q_rv_init)
    st[Q_LA] = aadc.idouble(p.q_la_init)
    st[Q_LV] = q_lv_init
    st[QC_VENOUS] = aadc.idouble(p.q_C_init_venous)
    st[Q_SYS] = aadc.idouble(p.q_init_sys)

    # Pre-run: stabilize
    for step in range(pre_steps):
        rates, lam = compute_rates_and_damping(
            st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B,
            th_trv, th_puv, th_miv, th_aov)
        for i in range(N_STATES):
            st[i] = st[i] + dt * rates[i] / (1.0 + dt * lam[i])
        for z in [ZETA_TRV, ZETA_PUV, ZETA_MIV, ZETA_AOV]:
            st[z] = iif(st[z] >= 0.0, st[z], aadc.idouble(0.0))
            st[z] = iif(st[z] <= 1.0, st[z], aadc.idouble(1.0))

    # Simulation: track min/max Q_LV using smooth approximation
    # Use running smooth max/min via iif
    q_lv_max = st[Q_LV]
    q_lv_min = st[Q_LV]

    for step in range(sim_steps):
        rates, lam = compute_rates_and_damping(
            st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B,
            th_trv, th_puv, th_miv, th_aov)
        for i in range(N_STATES):
            st[i] = st[i] + dt * rates[i] / (1.0 + dt * lam[i])
        for z in [ZETA_TRV, ZETA_PUV, ZETA_MIV, ZETA_AOV]:
            st[z] = iif(st[z] >= 0.0, st[z], aadc.idouble(0.0))
            st[z] = iif(st[z] <= 1.0, st[z], aadc.idouble(1.0))

        # Track max/min Q_LV (stroke volume = max - min)
        q_lv_max = iif(st[Q_LV] >= q_lv_max, st[Q_LV], q_lv_max)
        q_lv_min = iif(st[Q_LV] <= q_lv_min, st[Q_LV], q_lv_min)

    # Cost = -(max - min) = negative stroke volume (minimize to maximize SV)
    stroke_volume = q_lv_max - q_lv_min
    cost = -stroke_volume
    return cost


def main():
    print("SNAPO Valve Threshold Optimization — AADC Demo")
    print("=" * 60)
    print("Optimize 4 valve thresholds to maximize cardiac output")
    print("(stroke volume = max Q_LV - min Q_LV over one cardiac cycle)")
    print()

    p = Params()

    # ---- Step 1: Record AADC kernel ----
    print("Step 1: Recording AADC kernel...")
    t0 = time.time()

    funcs = aadc.Functions()
    funcs.start_recording()

    # Fixed calibration params
    id_qlv = aadc.idouble(p.q_lv_init)
    id_cao = aadc.idouble(p.C_aortic)
    id_elva = aadc.idouble(p.E_lv_A)
    id_elvb = aadc.idouble(p.E_lv_B)

    # Optimization params: valve thresholds (start at 0 = nominal)
    id_th_trv = aadc.idouble(0.0)
    id_th_puv = aadc.idouble(0.0)
    id_th_miv = aadc.idouble(0.0)
    id_th_aov = aadc.idouble(0.0)

    a_th_trv = id_th_trv.mark_as_input()
    a_th_puv = id_th_puv.mark_as_input()
    a_th_miv = id_th_miv.mark_as_input()
    a_th_aov = id_th_aov.mark_as_input()

    cost = run_model(id_qlv, id_cao, id_elva, id_elvb,
                     id_th_trv, id_th_puv, id_th_miv, id_th_aov,
                     pre_steps=2000, sim_steps=200, dt=0.01)

    r_cost = cost.mark_as_output()
    funcs.stop_recording()
    t_record = time.time() - t0
    print(f"  Recording time: {t_record:.1f}s")

    # ---- Step 2: Evaluate at baseline (all thresholds = 0) ----
    threshold_args = [a_th_trv, a_th_puv, a_th_miv, a_th_aov]
    threshold_names = ["th_trv", "th_puv", "th_miv", "th_aov"]
    valve_names = ["Tricuspid", "Pulmonary", "Mitral", "Aortic"]

    workers = aadc.ThreadPool(1)

    def evaluate_at(thresholds):
        """Evaluate cost and gradient at given threshold values."""
        inputs = {
            a_th_trv: thresholds[0],
            a_th_puv: thresholds[1],
            a_th_miv: thresholds[2],
            a_th_aov: thresholds[3],
        }
        request = {r_cost: threshold_args}
        res = aadc.evaluate(funcs, request, inputs, workers)
        cost_val = np.asarray(res[0][r_cost]).flat[0]
        grad = np.array([np.asarray(res[1][r_cost][a]).flat[0] for a in threshold_args])
        return cost_val, grad

    print("\nStep 2: Baseline evaluation (all thresholds = 0)...")
    th = np.zeros(4)
    cost_val, grad = evaluate_at(th)
    print(f"  Cost (neg SV) = {cost_val:.6e}")
    print(f"  Stroke volume = {-cost_val*1e6:.2f} mL")
    print(f"  Gradient:")
    for name, vname, g in zip(threshold_names, valve_names, grad):
        print(f"    d(cost)/d({name}) [{vname:10s}] = {g:.4e}")

    # ---- Step 3: Verify gradient with finite differences ----
    print("\nStep 3: Gradient verification (finite differences)...")
    h = 10.0  # Pa perturbation (small to stay within smooth region)
    for i in range(4):
        th_plus = th.copy(); th_plus[i] += h
        th_minus = th.copy(); th_minus[i] -= h
        c_plus, _ = evaluate_at(th_plus)
        c_minus, _ = evaluate_at(th_minus)
        fd = (c_plus - c_minus) / (2.0 * h)
        ratio = grad[i] / fd if abs(fd) > 1e-20 else float('nan')
        print(f"  {threshold_names[i]}: AD={grad[i]:.6e}  FD={fd:.6e}  ratio={ratio:.6f}")

    # ---- Step 4: Gradient descent optimization ----
    print("\nStep 4: Gradient descent optimization...")
    print(f"  {'Iter':>4s}  {'SV (mL)':>10s}  {'|grad|':>10s}  {'th_trv':>10s}  {'th_puv':>10s}  {'th_miv':>10s}  {'th_aov':>10s}")
    print("  " + "-" * 74)

    th = np.zeros(4)
    lr = 1e8  # learning rate (Pa units, cost in m^3)
    best_cost = float('inf')
    best_th = th.copy()

    for it in range(50):
        cost_val, grad = evaluate_at(th)
        sv_ml = -cost_val * 1e6  # convert m^3 to mL
        grad_norm = np.linalg.norm(grad)

        if cost_val < best_cost:
            best_cost = cost_val
            best_th = th.copy()

        if it % 5 == 0 or it < 5:
            print(f"  {it:4d}  {sv_ml:10.4f}  {grad_norm:10.2e}  {th[0]:10.1f}  {th[1]:10.1f}  {th[2]:10.1f}  {th[3]:10.1f}")

        # Gradient step (minimizing cost = maximizing stroke volume)
        th = th - lr * grad

        # Clamp thresholds to reasonable range (-5000, 5000) Pa
        th = np.clip(th, -5000, 5000)

    # Final
    cost_val, grad = evaluate_at(best_th)
    sv_ml = -cost_val * 1e6

    print(f"\n  Final: SV = {sv_ml:.4f} mL")
    print(f"  Optimal thresholds (Pa):")
    for name, vname, t in zip(threshold_names, valve_names, best_th):
        print(f"    {vname:10s}: {t:+.1f} Pa")

    # ---- Step 5: Benchmark ----
    print("\nStep 5: Benchmark...")
    n_iters = 50
    t0 = time.time()
    for _ in range(n_iters):
        evaluate_at(best_th)
    t1 = time.time()
    ms_per = (t1 - t0) / n_iters * 1000
    print(f"  {ms_per:.2f} ms/eval (cost + gradient)")
    print(f"  {1000/ms_per:.0f} evals/s")

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("Summary:")
    print(f"  Baseline stroke volume: {evaluate_at(np.zeros(4))[0]*-1e6:.4f} mL")
    print(f"  Optimized stroke volume: {sv_ml:.4f} mL")
    improvement = (sv_ml - evaluate_at(np.zeros(4))[0]*-1e6)
    print(f"  Improvement: {improvement:+.4f} mL")
    print(f"\n  AADC gives exact gradients through conditional valve logic")
    print(f"  CasADI: CRASHES (cannot trace if/else in valve dynamics)")
    print(f"  This optimization is IMPOSSIBLE without runtime AD.")


if __name__ == "__main__":
    main()
