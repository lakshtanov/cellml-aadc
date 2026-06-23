#!/usr/bin/env python3
"""
Gradient verification for aadc_ast_transform.py

Strategy: take the MANUAL port (cvs3_aadc_python.py) which has verified
gradient (AD/FD = 1.000000), convert it through aadc_ast_transform,
and verify that the converted code gives the SAME gradient.

Same parameters, same damping, same discretization → gradient must match.

This is the definitive test that the AST converter preserves AD correctness.
"""
import sys
import os
import math
import time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import aadc
from aadc_ast_transform import transform_to_aadc

# ============================================================
# Step 1: Build a "libcellml-style" version of the manual model
# that uses if/else, math.cos, math.floor, max, abs — the
# patterns that the converter needs to transform.
# ============================================================

ORIGINAL_CODE = '''
import math

PI = 3.14159265358979
N_STATES = 27

def leq_func(a, b):
    return a <= b

def geq_func(a, b):
    return a >= b

def floor_val(x):
    return math.floor(x)

def compute_rates_raw(st, p_R_pvn, p_C_pvn, p_I_pvn, p_q_C_init_pvn,
                      p_q_us_0_pvn, p_delta_q_us_pvn, p_u_ext_pvn, p_delta_C_pvn,
                      p_R_par, p_C_par, p_I_par, p_u_0_par, p_u_ext_par,
                      p_R_aortic, C_aortic, p_I_aortic, p_u_0_aortic, p_u_ext_aortic,
                      p_R_t_sys, p_C_t_sys, p_q_us_sys, p_u_ext_sys,
                      p_R_venous, p_C_venous, p_I_venous,
                      p_q_us_0_venous, p_delta_q_us_venous, p_u_ext_venous, p_delta_C_venous,
                      p_rho, p_T_period,
                      p_q_ra_us, p_q_rv_us, p_q_la_us, p_q_lv_us,
                      p_t_ac, p_t_ar, p_t_astart, p_t_vc, p_t_vr, p_t_vstart,
                      p_E_ra_A, p_E_ra_B, p_E_rv_A, p_E_rv_B,
                      p_E_la_A, p_E_la_B, E_lv_A, E_lv_B,
                      p_k_vo_trv, p_k_vc_trv, p_k_vo_puv, p_k_vc_puv,
                      p_k_vo_miv, p_k_vc_miv, p_k_vo_aov, p_k_vc_aov,
                      p_m_st_trv, p_m_st_puv, p_m_st_miv, p_m_st_aov,
                      p_m_rg_trv, p_m_rg_puv, p_m_rg_miv, p_m_rg_aov,
                      p_l_eff, p_a_nn_trv, p_a_nn_puv, p_a_nn_miv, p_a_nn_aov,
                      p_eps_1, p_eps_2, p_eps_m4, p_eps_m2, p_I_t_sys):
    """Same math as cvs3_aadc_python.py but using if/else, math.cos, max, abs."""
    rates = [0.0] * N_STATES
    lam = [0.0] * N_STATES

    r_v_pvn = 0.01 / p_C_pvn
    r_v_par = 0.01 / p_C_par
    r_v_aortic = 0.01 / C_aortic
    r_v_sys = 0.01 / p_C_t_sys
    r_v_venous = 0.01 / p_C_venous

    q_us_wcont_pvn = p_q_us_0_pvn * (1.0 - p_delta_q_us_pvn)
    c_wcont_pvn = p_C_pvn * (1.0 - p_delta_C_pvn)
    t_astart_norm = p_t_astart / p_T_period
    t_vstart_norm = p_t_vstart / p_T_period
    q_us_wcont_venous = p_q_us_0_venous * (1.0 - p_delta_q_us_venous)
    c_wcont_venous = p_C_venous * (1.0 - p_delta_C_venous)

    q_c_pvn = st[2] + p_q_us_0_pvn - q_us_wcont_pvn
    u_c_pvn = q_c_pvn / c_wcont_pvn
    u_pvn = u_c_pvn + p_u_ext_pvn + r_v_pvn * (st[1] - st[0])

    chi_afloor = st[5] - floor_val(st[5])
    chi_afloor_final = chi_afloor * 2.0 if leq_func(chi_afloor, 0.5) else 0.0
    e_a = 0.5 * (1.0 - math.cos(2.0 * PI * chi_afloor_final))

    u_la = (e_a * p_E_la_A + p_E_la_B) * (st[17] - p_q_la_us)
    rates[0] = (u_pvn - u_la - p_R_pvn * st[0]) / p_I_pvn
    lam[0] = (p_R_pvn + r_v_pvn) / p_I_pvn
    rates[2] = st[1] - st[0]

    u_c_par = st[3] / p_C_par
    u_par = p_u_0_par + u_c_par + p_u_ext_par + r_v_par * (st[4] - st[1])
    rates[1] = (u_par - u_pvn - p_R_par * st[1]) / p_I_par
    lam[1] = (p_R_par + r_v_par + r_v_pvn) / p_I_par
    rates[3] = st[4] - st[1]

    mt = st[7] - floor_val(st[7])

    r1a = 0.25 / p_t_ac; r2a = 0.25 / p_t_ar; r3a = 0.5 / (p_T_period - p_t_ac - p_t_ar)
    trig_a = r1a if geq_func(mt, t_astart_norm) and leq_func(mt, t_astart_norm + p_eps_1) and leq_func(chi_afloor, 0.25) else 0.0
    ongoing_a = r1a if chi_afloor > p_eps_2 and leq_func(chi_afloor, 0.25) else 0.0
    relax_a = r2a if geq_func(chi_afloor, 0.25) and chi_afloor < 0.5 else 0.0
    diast_a = r3a if geq_func(chi_afloor, 0.5) else 0.0
    rates[5] = trig_a + ongoing_a + relax_a + diast_a

    chi_vfloor = st[6] - floor_val(st[6])
    chi_vfloor_final = chi_vfloor * 2.0 if leq_func(chi_vfloor, 0.5) else 0.0
    e_v = 0.5 * (1.0 - math.cos(2.0 * PI * chi_vfloor_final))

    r1v = 0.25 / p_t_vc; r2v = 0.25 / p_t_vr; r3v = 0.5 / (p_T_period - p_t_vc - p_t_vr)
    trig_v = r1v if geq_func(mt, t_vstart_norm) and leq_func(mt, t_vstart_norm + p_eps_1) and leq_func(chi_vfloor, 0.25) else 0.0
    ongoing_v = r1v if chi_vfloor > p_eps_2 and leq_func(chi_vfloor, 0.25) else 0.0
    relax_v = r2v if geq_func(chi_vfloor, 0.25) and chi_vfloor < 0.5 else 0.0
    diast_v = r3v if chi_vfloor > 0.5 else 0.0
    rates[6] = trig_v + ongoing_v + relax_v + diast_v
    rates[7] = 1.0 / p_T_period

    u_rv = (e_v * p_E_rv_A + p_E_rv_B) * (st[16] - p_q_rv_us)
    u_ra = (e_a * p_E_ra_A + p_E_ra_B) * (st[15] - p_q_ra_us)
    u_lv = (e_v * E_lv_A + E_lv_B) * (st[18] - p_q_lv_us)

    u_c_aortic = st[21] / (C_aortic / 2.0)
    u_aortic = p_u_0_aortic + u_c_aortic + p_u_ext_aortic + 2.0 * r_v_aortic * (st[14] - st[22])

    # Valve zeta — using if/else (will be converted to iif)
    du_trv = u_ra - u_rv
    rates[8] = (1.0 - st[8]) * p_k_vo_trv * du_trv if geq_func(u_ra, u_rv) else st[8] * p_k_vc_trv * du_trv
    lam[8] = p_k_vo_trv * abs(du_trv) if geq_func(u_ra, u_rv) else p_k_vc_trv * abs(du_trv)

    du_puv = u_rv - u_par
    rates[9] = (1.0 - st[9]) * p_k_vo_puv * du_puv if geq_func(u_rv, u_par) else st[9] * p_k_vc_puv * du_puv
    lam[9] = p_k_vo_puv * abs(du_puv) if geq_func(u_rv, u_par) else p_k_vc_puv * abs(du_puv)

    du_miv = u_la - u_lv
    rates[10] = (1.0 - st[10]) * p_k_vo_miv * du_miv if geq_func(u_la, u_lv) else st[10] * p_k_vc_miv * du_miv
    lam[10] = p_k_vo_miv * abs(du_miv) if geq_func(u_la, u_lv) else p_k_vc_miv * abs(du_miv)

    du_aov = u_lv - u_aortic
    rates[11] = (1.0 - st[11]) * p_k_vo_aov * du_aov if geq_func(u_lv, u_aortic) else st[11] * p_k_vc_aov * du_aov
    lam[11] = p_k_vo_aov * abs(du_aov) if geq_func(u_lv, u_aortic) else p_k_vc_aov * abs(du_aov)

    # Valve flows — using max and abs
    def do_flow(v_idx, z_idx, m_st, a_nn, m_rg, u_up, u_down):
        zeta = max(st[z_idx], 0.0)
        a_eff = (m_st * a_nn - m_rg * a_nn) * zeta + m_rg * a_nn
        l = p_rho * p_l_eff / (a_eff + p_eps_m2)
        b = p_rho / (2.0 * a_eff * a_eff + p_eps_m4)
        v = st[v_idx]
        v_fabs = abs(v)
        rates[v_idx] = (-b * v * v_fabs + u_up - u_down) / l
        lam[v_idx] = 2.0 * b * v_fabs / l

    do_flow(12, 8, p_m_st_trv, p_a_nn_trv, p_m_rg_trv, u_ra, u_rv)
    do_flow(4, 9, p_m_st_puv, p_a_nn_puv, p_m_rg_puv, u_rv, u_par)
    do_flow(13, 10, p_m_st_miv, p_a_nn_miv, p_m_rg_miv, u_la, u_lv)
    do_flow(14, 11, p_m_st_aov, p_a_nn_aov, p_m_rg_aov, u_lv, u_aortic)

    rates[15] = st[19] - st[12]
    rates[16] = st[12] - st[4]
    rates[17] = st[0] - st[13]
    rates[18] = st[13] - st[14]

    u_c_d_aortic = st[20] / (C_aortic / 2.0)
    u_d_aortic = p_u_0_aortic + u_c_d_aortic + p_u_ext_aortic + 2.0 * r_v_aortic * (st[22] - st[23])
    rates[22] = (u_aortic - u_d_aortic - p_R_aortic * st[22]) / p_I_aortic
    lam[22] = (p_R_aortic + 4.0 * r_v_aortic) / p_I_aortic
    rates[21] = st[14] - st[22]
    rates[20] = st[22] - st[23]

    rates[25] = st[23] - st[24]
    u_c_sys = (st[25] - p_q_us_sys) / p_C_t_sys
    u_sys = u_c_sys + p_u_ext_sys + r_v_sys * (st[23] - st[24])
    rates[23] = (u_d_aortic - u_sys - st[23] * p_R_t_sys / 2.0) / p_I_t_sys
    lam[23] = (p_R_t_sys / 2.0 + 2.0 * r_v_aortic + r_v_sys) / p_I_t_sys

    q_c_venous = st[26] + p_q_us_0_venous - q_us_wcont_venous
    u_c_venous = q_c_venous / c_wcont_venous
    u_venous = u_c_venous + p_u_ext_venous + r_v_venous * (st[24] - st[19])
    rates[24] = (u_sys - u_venous - st[24] * p_R_t_sys / 2.0) / p_I_t_sys
    lam[24] = (p_R_t_sys / 2.0 + r_v_sys + r_v_venous) / p_I_t_sys
    rates[19] = (u_venous - u_ra - p_R_venous * st[19]) / p_I_venous
    lam[19] = (p_R_venous + r_v_venous) / p_I_venous
    rates[26] = st[24] - st[19]

    return rates, lam
'''

# ============================================================
# Step 2: Convert with aadc_ast_transform
# ============================================================

print("=" * 60)
print("Gradient verification: original vs AST-converted")
print("=" * 60)

converted = transform_to_aadc(ORIGINAL_CODE)
print(f"\n1. Converted {len(ORIGINAL_CODE)} → {len(converted)} bytes")

# Verify no unconverted patterns remain
import re
remaining = len(re.findall(r'(?<!_)leq_func\(|(?<!_)geq_func\(|(?<!\.)(?<!aadc\.)cos\(|(?<!\w)abs\((?!.*__)', converted))
print(f"   Unconverted calls: {remaining}")

# ============================================================
# Step 3: Load parameters (same as cvs3_aadc_python.py)
# ============================================================

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'QLtest', 'exp', 'CellML'))
try:
    from cvs3_aadc_python import Params, N_STATES
except ImportError:
    sys.path.insert(0, os.path.dirname(__file__))
    from cvs3_aadc_python import Params, N_STATES

p = Params()
N = N_STATES

# Build param tuple
params = (p.R_pvn, p.C_pvn, p.I_pvn, p.q_C_init_pvn,
          p.q_us_0_pvn, p.delta_q_us_pvn, p.u_ext_pvn, p.delta_C_pvn,
          p.R_par, p.C_par, p.I_par, p.u_0_par, p.u_ext_par,
          p.R_aortic, p.C_aortic, p.I_aortic, p.u_0_aortic, p.u_ext_aortic,
          p.R_t_sys, p.C_t_sys, p.q_us_sys, p.u_ext_sys,
          p.R_venous, p.C_venous, p.I_venous,
          p.q_us_0_venous, p.delta_q_us_venous, p.u_ext_venous, p.delta_C_venous,
          p.rho, p.T_period,
          p.q_ra_us, p.q_rv_us, p.q_la_us, p.q_lv_us,
          p.t_ac, p.t_ar, p.t_astart, p.t_vc, p.t_vr, p.t_vstart,
          p.E_ra_A, p.E_ra_B, p.E_rv_A, p.E_rv_B,
          p.E_la_A, p.E_la_B, p.E_lv_A, p.E_lv_B,
          p.k_vo_trv, p.k_vc_trv, p.k_vo_puv, p.k_vc_puv,
          p.k_vo_miv, p.k_vc_miv, p.k_vo_aov, p.k_vc_aov,
          p.m_st_trv, p.m_st_puv, p.m_st_miv, p.m_st_aov,
          p.m_rg_trv, p.m_rg_puv, p.m_rg_miv, p.m_rg_aov,
          p.l_eff, p.a_nn_trv, p.a_nn_puv, p.a_nn_miv, p.a_nn_aov,
          p.eps_1, p.eps_2, p.eps_m4, p.eps_m2, p.I_t_sys)

# Initial states
st0 = [0.0] * N
st0[2] = p.q_C_init_pvn  # QC_PVN
st0[15] = p.q_ra_init     # Q_RA
st0[16] = p.q_rv_init     # Q_RV
st0[17] = p.q_la_init     # Q_LA
st0[18] = p.q_lv_init     # Q_LV
st0[26] = p.q_C_init_venous  # QC_VENOUS
st0[25] = p.q_init_sys    # Q_SYS

# ============================================================
# Step 4: Run original code (plain floats) — get reference cost
# ============================================================

print("\n2. Running original (plain floats)...")

# Execute original code
ns_orig = {}
exec(ORIGINAL_CODE, ns_orig)
compute_rates_orig = ns_orig['compute_rates_raw']

st = list(st0)
dt = 0.01
for step in range(2200):
    rates, lam = compute_rates_orig(st, *params)
    for i in range(N):
        st[i] = st[i] + dt * rates[i] / (1.0 + dt * lam[i])
    for z in [8, 9, 10, 11]:
        st[z] = max(0.0, min(1.0, st[z]))

cost_orig = st[18] ** 2
print(f"   Cost = {cost_orig:.10e}")

# ============================================================
# Step 5: Run converted code (plain floats) — must match
# ============================================================

print("\n3. Running converted (plain floats)...")

def _aadc_passive(x):
    return x.val() if hasattr(x, 'val') else float(x)

ns_conv = {'math': math, 'aadc': aadc, '_aadc_passive': _aadc_passive}
exec(converted, ns_conv)
compute_rates_conv = ns_conv['compute_rates_raw']

st = list(st0)
for step in range(2200):
    rates, lam = compute_rates_conv(st, *params)
    for i in range(N):
        st[i] = st[i] + dt * rates[i] / (1.0 + dt * lam[i])
    for z in [8, 9, 10, 11]:
        st[z] = max(0.0, min(1.0, st[z]))

cost_conv = st[18] ** 2
print(f"   Cost = {cost_conv:.10e}")
print(f"   Match: {abs(cost_conv - cost_orig) < 1e-15}")

# ============================================================
# Step 6: AADC gradient — CONVERTED code
# ============================================================

print("\n4. AADC gradient — converted code...")

funcs = aadc.Functions()
funcs.start_recording()

id_st = [aadc.idouble(s) for s in st0]
id_elva = aadc.idouble(p.E_lv_A)
a_elva = id_elva.mark_as_input()

# ALL params as idouble (iif needs idouble on both sides)
id_params = [aadc.idouble(float(x)) for x in params]
id_params[46] = id_elva   # E_lv_A — the input we differentiate
id_params = tuple(id_params)

t0 = time.time()
for step in range(2200):
    rates, lam = compute_rates_conv(id_st, *id_params)
    for i in range(N):
        id_st[i] = id_st[i] + dt * rates[i] / (1.0 + dt * lam[i])
    for z in [8, 9, 10, 11]:
        id_st[z] = aadc.iif(id_st[z] >= 0.0, id_st[z], aadc.idouble(0.0))
        id_st[z] = aadc.iif(id_st[z] <= 1.0, id_st[z], aadc.idouble(1.0))

cost_id = id_st[18] * id_st[18]
r_cost = cost_id.mark_as_output()
funcs.stop_recording()
t_rec = time.time() - t0
print(f"   Recording: {t_rec:.1f}s")

w = aadc.ThreadPool(1)
res = aadc.evaluate(funcs, {r_cost: [a_elva]}, {a_elva: p.E_lv_A}, w)
cost_ad = float(np.asarray(res[0][r_cost]).flat[0])
grad_ad = float(np.asarray(res[1][r_cost][a_elva]).flat[0])
print(f"   Cost = {cost_ad:.10e}")
print(f"   Gradient = {grad_ad:.6e}")

# ============================================================
# Step 7: FD gradient — original code (plain floats)
# Same model, same damping, same discretization.
# ============================================================

print("\n5. FD gradient — original code (plain floats)...")

def run_float_cost(elva_val):
    st_f = list(st0)
    p_f = list(params)
    p_f[46] = elva_val  # E_lv_A
    p_f = tuple(p_f)
    for step in range(2200):
        rates_f, lam_f = compute_rates_orig(st_f, *p_f)
        for i in range(N):
            st_f[i] = st_f[i] + dt * rates_f[i] / (1.0 + dt * lam_f[i])
        for z in [8, 9, 10, 11]:
            st_f[z] = max(0.0, min(1.0, st_f[z]))
    return st_f[18] ** 2

h = p.E_lv_A * 1e-5
fd_plus = run_float_cost(p.E_lv_A + h)
fd_minus = run_float_cost(p.E_lv_A - h)
fd_grad = (fd_plus - fd_minus) / (2 * h)
print(f"   FD gradient = {fd_grad:.6e}")

# Also FD on tape (same discretization)
cp_tape = float(np.asarray(aadc.evaluate(funcs, {r_cost:[]}, {a_elva:p.E_lv_A+h}, w)[0][r_cost]).flat[0])
cm_tape = float(np.asarray(aadc.evaluate(funcs, {r_cost:[]}, {a_elva:p.E_lv_A-h}, w)[0][r_cost]).flat[0])
fd_tape = (cp_tape - cm_tape) / (2 * h)
print(f"   FD on tape = {fd_tape:.6e}")

# ============================================================
# Step 8: Results
# ============================================================

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

cost_match = abs(cost_ad - cost_orig) < 1e-15
ad_fd_float = grad_ad / fd_grad if abs(fd_grad) > 1e-30 else float('nan')
ad_fd_tape = grad_ad / fd_tape if abs(fd_tape) > 1e-30 else float('nan')

print(f"  Cost (float):     {cost_orig:.10e}")
print(f"  Cost (AADC):      {cost_ad:.10e}")
print(f"  Costs match:      {cost_match}")
print(f"")
print(f"  AD gradient:      {grad_ad:.6e}")
print(f"  FD (float, same damping): {fd_grad:.6e}")
print(f"  FD (tape replay): {fd_tape:.6e}")
print(f"  AD/FD (float):    {ad_fd_float:.6f}")
print(f"  AD/FD (tape):     {ad_fd_tape:.6f}")

# Benchmark
n = 20; t0 = time.time()
for _ in range(n):
    aadc.evaluate(funcs, {r_cost:[a_elva]}, {a_elva:p.E_lv_A}, w)
ms = (time.time()-t0)/n*1000
print(f"  Benchmark:        {ms:.1f} ms/eval")

all_pass = cost_match and abs(ad_fd_tape - 1.0) < 0.01
print(f"\n{'='*60}")
if all_pass:
    print("*** PASS: converter preserves exact gradient (AD/FD tape = 1.0) ***")
else:
    if not cost_match:
        print("*** FAIL: costs don't match ***")
    else:
        print(f"*** AD/FD (tape) = {ad_fd_tape:.6f} ***")
print(f"{'='*60}")

sys.exit(0 if all_pass else 1)
