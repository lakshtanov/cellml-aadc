#!/usr/bin/env python3
"""
CasADI driver for the 3-compartment model — the FAIR counterpart to AADC.

This builds the EXACT same model (cvs3_model.py) and the EXACT same
semi-implicit-Euler-with-damping integrator, but through CasADI symbolics using
`ca.if_else` for the valve conditionals. The original repo never did this — it
fed CasADI raw Python `if/else`, which is what "crashed".

The one-step update is built as a ca.Function and unrolled with `mapaccum`, which
applies the identical step N times (same integrator, just not re-traced 2200x).
Gradients come from CasADI reverse-mode AD via `ca.gradient`.
"""
import time
import casadi as ca

from backends import get_casadi_backend
from cvs3_model import Params, N_STATES, Q_LV, initial_state, euler_step

PARAM_NAMES = ["q_lv_init", "C_aortic", "E_lv_A", "E_lv_B"]


def build_casadi_kernel(pre_steps=2000, sim_steps=200, dt=0.01):
    """Build a CasADI Function p4 -> (cost, grad). Returns (fn, build_seconds)."""
    B = get_casadi_backend()
    p = Params()
    total = pre_steps + sim_steps

    t0 = time.time()

    x = ca.SX.sym("x", N_STATES)
    p4 = ca.SX.sym("p", 4)
    q_lv_init, C_aortic, E_lv_A, E_lv_B = p4[0], p4[1], p4[2], p4[3]

    st = [x[i] for i in range(N_STATES)]
    new = euler_step(B, st, p, q_lv_init, C_aortic, E_lv_A, E_lv_B, dt)
    x_next = ca.vertcat(*new)
    step = ca.Function("step", [x, p4], [x_next])

    # Unroll the identical step `total` times. The state accumulates; params are
    # constant across steps, so broadcast them as `total` identical columns.
    acc = step.mapaccum("acc", total)
    x0 = ca.vertcat(*initial_state(B, p, q_lv_init))
    P = ca.repmat(p4, 1, total)
    Xtraj = acc(x0, P)

    q_lv_final = Xtraj[Q_LV, total - 1]
    cost = q_lv_final * q_lv_final
    grad = ca.gradient(cost, p4)

    fn = ca.Function("cost_grad", [p4], [cost, grad])
    build_seconds = time.time() - t0
    return fn, build_seconds


def evaluate(fn, param_vals):
    """Return (cost, grad-as-list) at the given 4 parameter values."""
    cost, grad = fn(ca.DM(param_vals))
    return float(cost), [float(grad[i]) for i in range(4)]


def main():
    p = Params()
    print("Building CasADI kernel (ca.if_else, semi-implicit Euler + damping)...")
    fn, build_s = build_casadi_kernel()
    print(f"  Build: {build_s:.1f}s")

    vals = [p.q_lv_init, p.C_aortic, p.E_lv_A, p.E_lv_B]
    cost, grad = evaluate(fn, vals)
    print(f"\nCost = {cost:.6e}")
    print("Gradient:")
    for name, g in zip(PARAM_NAMES, grad):
        print(f"  dC/d{name} = {g:.6e}")

    n = 20
    t0 = time.time()
    for _ in range(n):
        evaluate(fn, vals)
    ms = (time.time() - t0) / n * 1000
    print(f"\nBenchmark: {ms:.2f} ms/eval (cost+gradient, 1 thread)")
    print("WORKS — including all conditional valve logic, via ca.if_else.")


if __name__ == "__main__":
    main()
