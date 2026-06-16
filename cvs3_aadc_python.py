#!/usr/bin/env python3
"""
3-Compartment Cardiovascular Model — AADC driver.

The model math now lives in cvs3_model.py (backend-agnostic) so the SAME source
runs under AADC here and under CasADI in cvs3_casadi.py — see compare_fair.py for
the side-by-side gradient check. This module keeps its original public API
(run_model, Params, Q_LV, N_STATES, ...) so the examples/tests are unchanged.

AADC records the actual execution onto an idouble tape; conditionals are handled
by aadc.iif (wired in via backends.get_aadc_backend()). Gradients are exact.
"""
import time
import numpy as np
import aadc

from backends import get_aadc_backend
# Re-export model symbols so existing imports keep working.
from cvs3_model import (  # noqa: F401
    Params, N_STATES, PI,
    V_PVN, V_PAR, QC_PVN, QC_PAR, V_PUV,
    CHI_A, CHI_V, S_HEART, ZETA_TRV, ZETA_PUV,
    ZETA_MIV, ZETA_AOV, V_TRV, V_MIV, V_AOV,
    Q_RA, Q_RV, Q_LA, Q_LV, V_VENOUS,
    QCD_AORTIC, QC_AORTIC, V_AORTIC, V_SYS, VT_SYS,
    Q_SYS, QC_VENOUS,
    simulate,
)

_AADC = get_aadc_backend()


def run_model(q_lv_init, C_aortic, E_lv_A, E_lv_B,
              pre_steps=2000, sim_steps=200, dt=0.01):
    """Run 3-compartment model under AADC, return cost = Q_LV^2."""
    return simulate(_AADC, q_lv_init, C_aortic, E_lv_A, E_lv_B,
                    pre_steps=pre_steps, sim_steps=sim_steps, dt=dt)


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

    print(f"\nCasADI with raw Python if/else: crashes.")
    print(f"CasADI with ca.if_else (cvs3_casadi.py): works — see compare_fair.py")


if __name__ == "__main__":
    main()
