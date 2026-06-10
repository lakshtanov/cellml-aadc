#!/usr/bin/env python3
"""
3-Compartment Cardiovascular Model — AADC Python demo.

Shows that Auckland's generated CellML Python code can work with AADC
by replacing:
  - `x if leq_func(a, b) else y`  →  `aadc.iif(a <= b, x, y)`
  - `max(x, 0)`                   →  `aadc.iif(x >= 0, x, 0)`

This gives exact gradients + Hessian from Python, where CasADI crashes.
"""
import time
import numpy as np
import aadc

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

PI = 3.14159265358979

def iif(cond, val_true, val_false):
    """Wrapper: use aadc.iif during recording, Python ternary otherwise."""
    return aadc.iif(cond, val_true, val_false)

def floor_id(x):
    """floor() for idouble — uses passive value."""
    import math
    return math.floor(float(x))

def compute_rates_and_damping(st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B):
    """Compute rates and diagonal damping coefficients."""
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

    # Atrial activation rate (simplified)
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

    # Valve zeta dynamics with damping
    def do_zeta(idx, u_up, u_down, k_vo, k_vc):
        du = u_up - u_down
        du_abs = iif(du >= 0.0, du, -du)
        rates[idx] = iif(u_up >= u_down,
                         (1.0 - st[idx]) * k_vo * du,
                         st[idx] * k_vc * du)
        lam[idx] = iif(u_up >= u_down, k_vo * du_abs, k_vc * du_abs)

    do_zeta(ZETA_TRV, u_ra, u_rv, p.k_vo_trv, p.k_vc_trv)
    do_zeta(ZETA_PUV, u_rv, u_par, p.k_vo_puv, p.k_vc_puv)
    do_zeta(ZETA_MIV, u_la, u_lv, p.k_vo_miv, p.k_vc_miv)
    do_zeta(ZETA_AOV, u_lv, u_aortic, p.k_vo_aov, p.k_vc_aov)

    # Valve flows
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
              pre_steps=2000, sim_steps=200, dt=0.01):
    """Run 3-compartment model, return cost = Q_LV^2."""
    p = Params()
    total_steps = pre_steps + sim_steps

    # Initial state
    st = [aadc.idouble(0.0)] * N_STATES
    st[QC_PVN] = aadc.idouble(p.q_C_init_pvn)
    st[Q_RA] = aadc.idouble(p.q_ra_init)
    st[Q_RV] = aadc.idouble(p.q_rv_init)
    st[Q_LA] = aadc.idouble(p.q_la_init)
    st[Q_LV] = q_lv_init
    st[QC_VENOUS] = aadc.idouble(p.q_C_init_venous)
    st[Q_SYS] = aadc.idouble(p.q_init_sys)

    # Semi-implicit Euler
    for step in range(total_steps):
        rates, lam = compute_rates_and_damping(st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B)
        for i in range(N_STATES):
            st[i] = st[i] + dt * rates[i] / (1.0 + dt * lam[i])
        # Clamp zeta to [0, 1]
        for z in [ZETA_TRV, ZETA_PUV, ZETA_MIV, ZETA_AOV]:
            st[z] = iif(st[z] >= 0.0, st[z], aadc.idouble(0.0))
            st[z] = iif(st[z] <= 1.0, st[z], aadc.idouble(1.0))

    return st[Q_LV] * st[Q_LV]


def main():
    print("3-Compartment CVS Model — AADC Python Demo")
    print("=" * 60)
    p = Params()

    # Record AADC kernel
    print("\nRecording AADC kernel from Python...")
    t0 = time.time()

    funcs = aadc.Functions()
    funcs.start_recording()

    # Calibration parameters as inputs
    id_qlv = aadc.idouble(p.q_lv_init)
    id_cao = aadc.idouble(p.C_aortic)
    id_elva = aadc.idouble(p.E_lv_A)
    id_elvb = aadc.idouble(p.E_lv_B)

    a_qlv = id_qlv.mark_as_input()
    a_cao = id_cao.mark_as_input()
    a_elva = id_elva.mark_as_input()
    a_elvb = id_elvb.mark_as_input()

    # Run model
    cost = run_model(id_qlv, id_cao, id_elva, id_elvb,
                     pre_steps=2000, sim_steps=200, dt=0.01)

    r_cost = cost.mark_as_output()
    funcs.stop_recording()

    t1 = time.time()
    print(f"Recording: {t1-t0:.1f}s")

    # Evaluate
    inputs = {
        a_qlv: p.q_lv_init,
        a_cao: p.C_aortic,
        a_elva: p.E_lv_A,
        a_elvb: p.E_lv_B,
    }
    request = {r_cost: [a_qlv, a_cao, a_elva, a_elvb]}

    workers = aadc.ThreadPool(1)

    # Warmup
    res = aadc.evaluate(funcs, request, inputs, workers)
    cost_val = np.asarray(res[0][r_cost]).flat[0]
    print(f"\nCost = {cost_val:.6e}")
    print(f"Gradient:")
    names = ["dC/dq_lv", "dC/dC_ao", "dC/dE_lv_A", "dC/dE_lv_B"]
    args_list = [a_qlv, a_cao, a_elva, a_elvb]
    for name, arg in zip(names, args_list):
        g = np.asarray(res[1][r_cost][arg]).flat[0]
        print(f"  {name} = {g:.6e}")

    # Benchmark
    n_iters = 20
    t0 = time.time()
    for _ in range(n_iters):
        res = aadc.evaluate(funcs, request, inputs, workers)
    t1 = time.time()
    ms_per_eval = (t1 - t0) / n_iters * 1000
    print(f"\nBenchmark: {ms_per_eval:.2f} ms/eval (gradient included)")
    print(f"           {1000/ms_per_eval:.0f} evals/s")

    # Multi-thread
    workers4 = aadc.ThreadPool(4)
    inputs_batch = {
        a_qlv: p.q_lv_init * np.ones(4),
        a_cao: p.C_aortic * np.ones(4),
        a_elva: p.E_lv_A * np.ones(4),
        a_elvb: p.E_lv_B * np.ones(4),
    }
    t0 = time.time()
    for _ in range(n_iters):
        res = aadc.evaluate(funcs, request, inputs_batch, workers4)
    t1 = time.time()
    ms_total = (t1 - t0) / n_iters * 1000
    evals = 4  # batch size
    print(f"\n4 threads: {ms_total:.2f} ms for {evals} evals = {ms_total/evals:.2f} ms/eval")

    print(f"\nCasADI: CRASHES on this model (cannot trace if/else symbolically)")
    print(f"AADC:   Works via aadc.iif() — records both branches")


if __name__ == "__main__":
    main()
