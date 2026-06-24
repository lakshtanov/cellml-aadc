#!/usr/bin/env python3
"""
Implicit Euler ODE solver using aadc.least_squares + IFT.

Each implicit Euler step solves:
    g(y_{n+1}) = y_{n+1} - y_n - dt * f(y_{n+1}) = 0
via aadc.least_squares (Levenberg-Marquardt). The Implicit Function Theorem
gives exact AD gradients through each solve automatically.

Tests:
  1. Lotka-Volterra (nonstiff): value vs scipy, gradient AD/FD
  2. 3-compartment CVS (stiff): gradient AD/FD
  3. Benchmark: recording time, evaluation time, gradient time

Run:  python test_implicit_euler_ift.py
  or: pytest test_implicit_euler_ift.py -v
"""
import time
import numpy as np
import pytest

import aadc


# ========== Implicit Euler step via least_squares ==========

def implicit_euler_step(rhs_func, y_n, dt, params, tol=1e-12):
    """One implicit Euler step: solve y_{n+1} - y_n - dt*f(y_{n+1}) = 0."""
    n = len(y_n)

    def residual(y_next):
        f = rhs_func(y_next, params)
        return [y_next[i] - y_n[i] - dt * f[i] for i in range(n)]

    result = aadc.least_squares(residual, list(y_n), ftol=tol, xtol=tol)
    return result.x


def integrate_implicit_euler(rhs_func, y0, dt, n_steps, params, tol=1e-12):
    """Integrate ODE with implicit Euler, n_steps steps."""
    y = list(y0)
    for _ in range(n_steps):
        y = implicit_euler_step(rhs_func, y, dt, params, tol)
    return y


# ========== Model 1: Lotka-Volterra ==========

def lotka_volterra_rhs(state, params):
    x, y = state[0], state[1]
    alpha, beta, delta, gamma = params
    return [alpha * x - beta * x * y,
            delta * x * y - gamma * y]


def lotka_volterra_rk4_passive(alpha, beta, delta, gamma, x0, y0, dt, N):
    """Reference RK4 solver (passive doubles)."""
    x, y = x0, y0
    for _ in range(N):
        dx1 = alpha*x - beta*x*y;       dy1 = delta*x*y - gamma*y
        x2 = x + 0.5*dt*dx1;            y2 = y + 0.5*dt*dy1
        dx2 = alpha*x2 - beta*x2*y2;    dy2 = delta*x2*y2 - gamma*y2
        x3 = x + 0.5*dt*dx2;            y3 = y + 0.5*dt*dy2
        dx3 = alpha*x3 - beta*x3*y3;    dy3 = delta*x3*y3 - gamma*y3
        x4 = x + dt*dx3;                y4 = y + dt*dy3
        dx4 = alpha*x4 - beta*x4*y4;    dy4 = delta*x4*y4 - gamma*y4
        x += dt/6*(dx1 + 2*dx2 + 2*dx3 + dx4)
        y += dt/6*(dy1 + 2*dy2 + 2*dy3 + dy4)
    return x, y


# ========== Model 2: 3-compartment hemodynamics ==========

def hemodynamics_rhs(state, params):
    """3-compartment: pressures P1,P2,P3 with resistances and compliances."""
    P1, P2, P3 = state[0], state[1], state[2]
    R12, R23, R31, C1, C2, C3 = params

    Q12 = (P1 - P2) / R12
    Q23 = (P2 - P3) / R23
    Q31 = (P3 - P1) / R31

    dP1 = (Q31 - Q12) / C1
    dP2 = (Q12 - Q23) / C2
    dP3 = (Q23 - Q31) / C3
    return [dP1, dP2, dP3]


# ========== Test 1: Lotka-Volterra value + gradient ==========

def test_lotka_volterra_value():
    """Implicit Euler on LV matches RK4 reference (small dt)."""
    alpha, beta, delta, gamma = 1.5, 1.0, 1.0, 3.0
    dt = 0.001
    N = 100

    # Reference: RK4
    xref, yref = lotka_volterra_rk4_passive(alpha, beta, delta, gamma, 1.0, 1.0, dt, N)

    # Implicit Euler (no recording, just passive)
    func = aadc.Functions()
    func.start_recording()
    a = aadc.idouble(alpha)
    a_arg = a.mark_as_input()
    params = [a, aadc.idouble(beta), aadc.idouble(delta), aadc.idouble(gamma)]
    y0 = [aadc.idouble(1.0), aadc.idouble(1.0)]
    y = integrate_implicit_euler(lotka_volterra_rhs, y0, aadc.idouble(dt), N, params)
    out_x = y[0].mark_as_output()
    out_y = y[1].mark_as_output()
    func.stop_recording()

    res = aadc.evaluate(func, {out_x: [], out_y: []}, {a_arg: alpha}, aadc.ThreadPool(1))
    x_ie = res[0][out_x][0]
    y_ie = res[0][out_y][0]

    # Implicit Euler is 1st order, RK4 is 4th order. With dt=0.001, N=100 (T=0.1)
    # error should be small but not machine-eps
    assert abs(x_ie - xref) / abs(xref) < 0.01, f"x: IE={x_ie}, RK4={xref}"
    assert abs(y_ie - yref) / abs(yref) < 0.01, f"y: IE={y_ie}, RK4={yref}"
    print(f"  LV value: IE=({x_ie:.6f},{y_ie:.6f}), RK4=({xref:.6f},{yref:.6f})")


def test_lotka_volterra_gradient():
    """AD gradient through implicit Euler matches FD on Lotka-Volterra."""
    alpha, beta, delta, gamma = 1.5, 1.0, 1.0, 3.0
    dt = 0.01
    N = 50

    func = aadc.Functions()
    func.start_recording()
    a = aadc.idouble(alpha)
    a_arg = a.mark_as_input()
    params = [a, aadc.idouble(beta), aadc.idouble(delta), aadc.idouble(gamma)]
    y0 = [aadc.idouble(1.0), aadc.idouble(1.0)]
    y = integrate_implicit_euler(lotka_volterra_rhs, y0, aadc.idouble(dt), N, params)
    out_x = y[0].mark_as_output()
    out_y = y[1].mark_as_output()
    func.stop_recording()

    workers = aadc.ThreadPool(1)
    request = {out_x: [a_arg], out_y: [a_arg]}
    res = aadc.evaluate(func, request, {a_arg: alpha}, workers)

    ad_dx = res[1][out_x][a_arg][0]
    ad_dy = res[1][out_y][a_arg][0]

    # FD
    eps = 1e-5
    res_up = aadc.evaluate(func, {out_x: [], out_y: []}, {a_arg: alpha + eps}, workers)
    res_dn = aadc.evaluate(func, {out_x: [], out_y: []}, {a_arg: alpha - eps}, workers)
    fd_dx = (res_up[0][out_x][0] - res_dn[0][out_x][0]) / (2 * eps)
    fd_dy = (res_up[0][out_y][0] - res_dn[0][out_y][0]) / (2 * eps)

    ratio_x = ad_dx / fd_dx
    ratio_y = ad_dy / fd_dy
    print(f"  LV grad: AD=({ad_dx:.6f},{ad_dy:.6f}), FD=({fd_dx:.6f},{fd_dy:.6f})")
    print(f"  AD/FD: x={ratio_x:.6f}, y={ratio_y:.6f}")
    assert abs(ratio_x - 1.0) < 1e-4, f"x ratio={ratio_x}"
    assert abs(ratio_y - 1.0) < 1e-4, f"y ratio={ratio_y}"


# ========== Test 2: 3-compartment gradient ==========

def test_hemodynamics_gradient():
    """AD gradient through implicit Euler matches FD on 3-compartment model."""
    R12_val = 1.0
    params_vals = [R12_val, 0.5, 0.8, 1.0, 2.0, 1.5]
    y0_vals = [80.0, 10.0, 5.0]
    dt = 0.01
    N = 100

    func = aadc.Functions()
    func.start_recording()
    R12 = aadc.idouble(R12_val)
    R12_arg = R12.mark_as_input()
    params = [R12] + [aadc.idouble(v) for v in params_vals[1:]]
    y0 = [aadc.idouble(v) for v in y0_vals]
    y = integrate_implicit_euler(hemodynamics_rhs, y0, aadc.idouble(dt), N, params)
    outs = [yi.mark_as_output() for yi in y]
    func.stop_recording()

    workers = aadc.ThreadPool(1)
    request = {o: [R12_arg] for o in outs}
    res = aadc.evaluate(func, request, {R12_arg: R12_val}, workers)

    eps = 1e-5
    res_up = aadc.evaluate(func, {o: [] for o in outs}, {R12_arg: R12_val + eps}, workers)
    res_dn = aadc.evaluate(func, {o: [] for o in outs}, {R12_arg: R12_val - eps}, workers)

    print(f"  3-comp ({N} steps, dt={dt}):")
    for i, o in enumerate(outs):
        ad = res[1][o][R12_arg][0]
        fd = (res_up[0][o][0] - res_dn[0][o][0]) / (2 * eps)
        ratio = ad / fd if abs(fd) > 1e-15 else float('nan')
        print(f"    P{i+1}: val={res[0][o][0]:.6f}  AD={ad:.6f}  FD={fd:.6f}  AD/FD={ratio:.6f}")
        assert abs(ratio - 1.0) < 1e-3, f"P{i+1} ratio={ratio}"


# ========== Test 3: Multiple parameters ==========

def test_hemodynamics_multi_param():
    """Gradient w.r.t. multiple parameters simultaneously."""
    R12_val, R23_val = 1.0, 0.5
    params_vals = [R12_val, R23_val, 0.8, 1.0, 2.0, 1.5]
    y0_vals = [80.0, 10.0, 5.0]
    dt = 0.01
    N = 50

    func = aadc.Functions()
    func.start_recording()
    R12 = aadc.idouble(R12_val)
    R23 = aadc.idouble(R23_val)
    R12_arg = R12.mark_as_input()
    R23_arg = R23.mark_as_input()
    params = [R12, R23] + [aadc.idouble(v) for v in params_vals[2:]]
    y0 = [aadc.idouble(v) for v in y0_vals]
    y = integrate_implicit_euler(hemodynamics_rhs, y0, aadc.idouble(dt), N, params)
    out = y[0].mark_as_output()
    func.stop_recording()

    workers = aadc.ThreadPool(1)
    request = {out: [R12_arg, R23_arg]}
    res = aadc.evaluate(func, request, {R12_arg: R12_val, R23_arg: R23_val}, workers)
    ad_r12 = res[1][out][R12_arg][0]
    ad_r23 = res[1][out][R23_arg][0]

    eps = 1e-5
    res_up = aadc.evaluate(func, {out: []}, {R12_arg: R12_val + eps, R23_arg: R23_val}, workers)
    res_dn = aadc.evaluate(func, {out: []}, {R12_arg: R12_val - eps, R23_arg: R23_val}, workers)
    fd_r12 = (res_up[0][out][0] - res_dn[0][out][0]) / (2 * eps)

    res_up = aadc.evaluate(func, {out: []}, {R12_arg: R12_val, R23_arg: R23_val + eps}, workers)
    res_dn = aadc.evaluate(func, {out: []}, {R12_arg: R12_val, R23_arg: R23_val - eps}, workers)
    fd_r23 = (res_up[0][out][0] - res_dn[0][out][0]) / (2 * eps)

    r1 = ad_r12 / fd_r12
    r2 = ad_r23 / fd_r23
    print(f"  Multi-param: dP1/dR12 AD={ad_r12:.6f} FD={fd_r12:.6f} ratio={r1:.6f}")
    print(f"  Multi-param: dP1/dR23 AD={ad_r23:.6f} FD={fd_r23:.6f} ratio={r2:.6f}")
    assert abs(r1 - 1.0) < 1e-3, f"R12 ratio={r1}"
    assert abs(r2 - 1.0) < 1e-3, f"R23 ratio={r2}"


# ========== Benchmark ==========

def test_benchmark():
    """Benchmark: recording + evaluation times."""
    params_vals = [1.0, 0.5, 0.8, 1.0, 2.0, 1.5]
    y0_vals = [80.0, 10.0, 5.0]
    dt = 0.01

    print("\n  Benchmark: 3-compartment hemodynamics")
    print(f"  {'N_steps':>8} {'rec_ms':>10} {'eval_ms':>10} {'grad_ms':>10}")

    for N in [10, 50, 100, 200]:
        # Recording
        t0 = time.time()
        func = aadc.Functions()
        func.start_recording()
        R12 = aadc.idouble(params_vals[0])
        R12_arg = R12.mark_as_input()
        params = [R12] + [aadc.idouble(v) for v in params_vals[1:]]
        y0 = [aadc.idouble(v) for v in y0_vals]
        y = integrate_implicit_euler(hemodynamics_rhs, y0, aadc.idouble(dt), N, params)
        outs = [yi.mark_as_output() for yi in y]
        func.stop_recording()
        rec_ms = (time.time() - t0) * 1000

        workers = aadc.ThreadPool(1)

        # Forward only
        t0 = time.time()
        n_eval = 100
        for _ in range(n_eval):
            aadc.evaluate(func, {o: [] for o in outs}, {R12_arg: 1.0}, workers)
        eval_ms = (time.time() - t0) * 1000 / n_eval

        # Forward + gradient
        request = {o: [R12_arg] for o in outs}
        t0 = time.time()
        for _ in range(n_eval):
            aadc.evaluate(func, request, {R12_arg: 1.0}, workers)
        grad_ms = (time.time() - t0) * 1000 / n_eval

        print(f"  {N:>8} {rec_ms:>10.1f} {eval_ms:>10.3f} {grad_ms:>10.3f}")


# ========== Main ==========

if __name__ == "__main__":
    print("Implicit Euler + aadc.least_squares + IFT")
    print("=" * 50)

    print("\nTest 1a: Lotka-Volterra value vs RK4")
    test_lotka_volterra_value()
    print("  PASS")

    print("\nTest 1b: Lotka-Volterra gradient AD/FD")
    test_lotka_volterra_gradient()
    print("  PASS")

    print("\nTest 2: 3-compartment gradient AD/FD")
    test_hemodynamics_gradient()
    print("  PASS")

    print("\nTest 3: Multi-parameter gradient")
    test_hemodynamics_multi_param()
    print("  PASS")

    print("\nTest 4: Benchmark")
    test_benchmark()

    print("\nAll tests passed.")
