#!/usr/bin/env python3
"""
FAIR side-by-side comparison: AADC vs CasADI.

Unlike the original compare_casadi_vs_aadc.py, this differs ONLY in the AD
backend:
  * Same model source (cvs3_model.py / the Lotka model below).
  * Same integrator: semi-implicit Euler with diagonal damping (forward Euler,
    i.e. lam=0, for the non-stiff Lotka model).
  * CasADI uses ca.if_else for the valve conditionals (its analogue of aadc.iif),
    so it does NOT crash.
  * Single thread, no AVX batching, build/record time reported separately.

The script asserts that AADC and CasADI gradients agree with each other and with
finite differences. If they do, the original "CasADI crashes" headline is shown
to be an artifact of feeding CasADI raw Python if/else rather than ca.if_else.

Run:  venv/bin/python compare_fair.py
"""
import time
import numpy as np

from backends import get_aadc_backend, get_casadi_backend, get_numeric_backend

SEP = "=" * 72


# ============================================================
# Model 1: Lotka-Volterra (shared source, shared forward-Euler integrator)
# ============================================================
def lotka_cost(B, params, dt=0.01, N=500):
    a, b, d, g = params
    x = B.const(1.0)
    y = B.const(1.0)
    for _ in range(N):
        dx = a * x - b * x * y
        dy = d * x * y - g * y
        x = x + dt * dx
        y = y + dt * dy
    return x * x + y * y


LOTKA_PARAMS = [1.5, 1.0, 3.0, 1.0]


def lotka_aadc():
    import aadc
    B = get_aadc_backend()
    funcs = aadc.Functions()
    funcs.start_recording()
    ps = [aadc.idouble(v) for v in LOTKA_PARAMS]
    args = [pi.mark_as_input() for pi in ps]
    cost = lotka_cost(B, ps)
    r = cost.mark_as_output()
    funcs.stop_recording()

    inputs = {a: v for a, v in zip(args, LOTKA_PARAMS)}
    workers = aadc.ThreadPool(1)
    res = aadc.evaluate(funcs, {r: list(args)}, inputs, workers)
    cost_val = float(np.asarray(res[0][r]).flat[0])
    grad = [float(np.asarray(res[1][r][a]).flat[0]) for a in args]
    return cost_val, grad


def lotka_casadi():
    import casadi as ca
    B = get_casadi_backend()
    p = ca.SX.sym("p", 4)
    cost = lotka_cost(B, [p[0], p[1], p[2], p[3]])
    fn = ca.Function("f", [p], [cost, ca.gradient(cost, p)])
    cost_v, grad = fn(ca.DM(LOTKA_PARAMS))
    return float(cost_v), [float(grad[i]) for i in range(4)]


def lotka_fd(eps=1e-7):
    B = get_numeric_backend()
    grad = []
    for i in range(4):
        up = list(LOTKA_PARAMS); up[i] += eps
        dn = list(LOTKA_PARAMS); dn[i] -= eps
        grad.append((lotka_cost(B, up) - lotka_cost(B, dn)) / (2 * eps))
    return lotka_cost(B, LOTKA_PARAMS), grad


# ============================================================
# Model 2: 3-compartment CVS (shared source, semi-implicit Euler + damping)
# ============================================================
CVS_PARAM_NAMES = ["q_lv_init", "C_aortic", "E_lv_A", "E_lv_B"]


def cvs_param_vals():
    from cvs3_model import Params
    p = Params()
    return [p.q_lv_init, p.C_aortic, p.E_lv_A, p.E_lv_B]


def cvs_aadc(pre_steps=2000, sim_steps=200, dt=0.01):
    import aadc
    from cvs3_model import simulate
    B = get_aadc_backend()
    vals = cvs_param_vals()

    t0 = time.time()
    funcs = aadc.Functions()
    funcs.start_recording()
    ids = [aadc.idouble(v) for v in vals]
    args = [i.mark_as_input() for i in ids]
    cost = simulate(B, *ids, pre_steps=pre_steps, sim_steps=sim_steps, dt=dt)
    r = cost.mark_as_output()
    funcs.stop_recording()
    build = time.time() - t0

    inputs = {a: v for a, v in zip(args, vals)}
    workers = aadc.ThreadPool(1)
    res = aadc.evaluate(funcs, {r: list(args)}, inputs, workers)
    cost_val = float(np.asarray(res[0][r]).flat[0])
    grad = [float(np.asarray(res[1][r][a]).flat[0]) for a in args]

    n = 20
    t0 = time.time()
    for _ in range(n):
        aadc.evaluate(funcs, {r: list(args)}, inputs, workers)
    eval_ms = (time.time() - t0) / n * 1000
    return cost_val, grad, build, eval_ms


def cvs_casadi(pre_steps=2000, sim_steps=200, dt=0.01):
    from cvs3_casadi import build_casadi_kernel, evaluate
    fn, build = build_casadi_kernel(pre_steps, sim_steps, dt)
    vals = cvs_param_vals()
    cost_val, grad = evaluate(fn, vals)

    n = 20
    t0 = time.time()
    for _ in range(n):
        evaluate(fn, vals)
    eval_ms = (time.time() - t0) / n * 1000
    return cost_val, grad, build, eval_ms


def cvs_fd(pre_steps=2000, sim_steps=200, dt=0.01, eps=1e-6):
    """Central finite differences using the plain-float backend (no AD)."""
    from cvs3_model import simulate
    B = get_numeric_backend()
    vals = cvs_param_vals()
    grad = []
    for i in range(4):
        h = vals[i] * eps
        up = list(vals); up[i] += h
        dn = list(vals); dn[i] -= h
        c_up = simulate(B, *up, pre_steps=pre_steps, sim_steps=sim_steps, dt=dt)
        c_dn = simulate(B, *dn, pre_steps=pre_steps, sim_steps=sim_steps, dt=dt)
        grad.append((c_up - c_dn) / (2 * h))
    return grad


# ============================================================
# Reporting helpers
# ============================================================
def _rel(a, b):
    """Max relative difference between two gradient vectors (scale-aware)."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    denom = np.maximum(np.abs(a), np.abs(b))
    denom = np.where(denom == 0.0, 1.0, denom)
    return float(np.max(np.abs(a - b) / denom))


def _print_grads(names, **named_grads):
    head = "  {:<12}" + "".join(f"{n:>16}" for n in named_grads)
    print(head.format("param"))
    for i, nm in enumerate(names):
        row = "  {:<12}".format(nm) + "".join(f"{g[i]:>16.6e}" for g in named_grads.values())
        print(row)


def main():
    failures = []

    # ---- Lotka-Volterra ----
    print(SEP)
    print("MODEL 1: Lotka-Volterra (forward Euler, shared source)")
    print(SEP)
    c_aa, g_aa = lotka_aadc()
    c_ca, g_ca = lotka_casadi()
    c_fd, g_fd = lotka_fd()
    print(f"  cost  AADC={c_aa:.6e}  CasADI={c_ca:.6e}  (FD ref {c_fd:.6e})")
    _print_grads(["alpha", "beta", "delta", "gamma"], AADC=g_aa, CasADI=g_ca, FD=g_fd)
    r_adad = _rel(g_aa, g_ca)
    r_fd = max(_rel(g_aa, g_fd), _rel(g_ca, g_fd))
    print(f"  AADC vs CasADI: {r_adad:.2e}   |   AD vs FD: {r_fd:.2e}")
    if r_adad > 1e-9: failures.append(f"Lotka AADC-vs-CasADI {r_adad:.2e}")
    if r_fd > 1e-5: failures.append(f"Lotka AD-vs-FD {r_fd:.2e}")

    # ---- 3-compartment ----
    print("\n" + SEP)
    print("MODEL 2: 3-Compartment CVS (semi-implicit Euler + damping, conditionals)")
    print(SEP)
    print("  Building/recording both kernels...")
    c_aa, g_aa, b_aa, e_aa = cvs_aadc()
    c_ca, g_ca, b_ca, e_ca = cvs_casadi()
    g_fd = cvs_fd()
    print(f"  cost  AADC={c_aa:.6e}  CasADI={c_ca:.6e}")
    _print_grads(CVS_PARAM_NAMES, AADC=g_aa, CasADI=g_ca, FD=g_fd)
    r_adad = _rel(g_aa, g_ca)
    r_fd = max(_rel(g_aa, g_fd), _rel(g_ca, g_fd))
    print(f"  AADC vs CasADI: {r_adad:.2e}   |   AD vs FD: {r_fd:.2e}")
    print(f"  build/record:  AADC={b_aa:.1f}s   CasADI={b_ca:.1f}s")
    print(f"  eval (1 thr):  AADC={e_aa:.2f} ms   CasADI={e_ca:.2f} ms")
    if r_adad > 1e-6: failures.append(f"CVS AADC-vs-CasADI {r_adad:.2e}")
    if r_fd > 1e-3: failures.append(f"CVS AD-vs-FD {r_fd:.2e}")

    # ---- Verdict ----
    print("\n" + SEP)
    if failures:
        print("RESULT: MISMATCH — " + "; ".join(failures))
        return 1
    print("RESULT: CasADI (with ca.if_else) produces finite gradients that match")
    print("AADC and finite differences on BOTH models. The 'CasADI crashes' claim")
    print("was an artifact of raw Python if/else, not a real CasADI limitation.")
    print(SEP)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
