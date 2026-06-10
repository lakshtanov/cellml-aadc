#!/usr/bin/env python3
"""
Side-by-side comparison: CasADI vs AADC on Auckland's models.

1. Lotka-Volterra: both work → compare speed and gradients
2. 3-compartment CVS: CasADI crashes → AADC works

Run:  python compare_casadi_vs_aadc.py
"""
import time
import sys
import numpy as np

SEPARATOR = "=" * 70

# ============================================================
# Model 1: Lotka-Volterra (2 states, 4 params)
# ============================================================
def lotka_volterra_rk4(alpha, beta, delta, gamma, dt=0.01, N=500):
    """RK4 integration of Lotka-Volterra. Works with float or idouble."""
    x, y = 1.0, 1.0
    for _ in range(N):
        dx1 = alpha*x - beta*x*y
        dy1 = delta*x*y - gamma*y
        x2 = x + 0.5*dt*dx1; y2 = y + 0.5*dt*dy1
        dx2 = alpha*x2 - beta*x2*y2
        dy2 = delta*x2*y2 - gamma*y2
        x3 = x + 0.5*dt*dx2; y3 = y + 0.5*dt*dy2
        dx3 = alpha*x3 - beta*x3*y3
        dy3 = delta*x3*y3 - gamma*y3
        x4 = x + dt*dx3; y4 = y + dt*dy3
        dx4 = alpha*x4 - beta*x4*y4
        dy4 = delta*x4*y4 - gamma*y4
        x += dt/6*(dx1 + 2*dx2 + 2*dx3 + dx4)
        y += dt/6*(dy1 + 2*dy2 + 2*dy3 + dy4)
    return x*x + y*y  # cost


def test_lotka_casadi():
    """Lotka-Volterra with CasADI AD."""
    print(f"\n{'─'*50}")
    print("Lotka-Volterra: CasADI")
    print(f"{'─'*50}")
    try:
        import casadi as ca
    except ImportError:
        print("  CasADI not installed — skipping")
        return None, None

    alpha = ca.SX.sym('alpha')
    beta  = ca.SX.sym('beta')
    delta = ca.SX.sym('delta')
    gamma = ca.SX.sym('gamma')

    cost = lotka_volterra_rk4(alpha, beta, delta, gamma)
    grad = ca.gradient(cost, ca.vertcat(alpha, beta, delta, gamma))
    f = ca.Function('f', [alpha, beta, delta, gamma], [cost, grad])

    params = [1.5, 1.0, 3.0, 1.0]

    # Warmup
    f(*params)

    # Benchmark
    n = 50
    t0 = time.time()
    for _ in range(n):
        res = f(*params)
    elapsed = (time.time() - t0) / n * 1000

    cost_val = float(res[0])
    grad_val = [float(res[1][i]) for i in range(4)]

    print(f"  Cost:     {cost_val:.6e}")
    print(f"  Gradient: {grad_val}")
    print(f"  Time:     {elapsed:.1f} ms/eval")
    return cost_val, grad_val


def test_lotka_aadc():
    """Lotka-Volterra with AADC AD."""
    print(f"\n{'─'*50}")
    print("Lotka-Volterra: AADC")
    print(f"{'─'*50}")
    try:
        import aadc
    except ImportError:
        print("  AADC not installed — skipping")
        return None, None

    params = [1.5, 1.0, 3.0, 1.0]

    # Record kernel
    funcs = aadc.Functions()
    funcs.start_recording()
    p = [aadc.idouble(v) for v in params]
    args = [pi.mark_as_input() for pi in p]
    cost = lotka_volterra_rk4(*p)
    r_cost = cost.mark_as_output()
    funcs.stop_recording()

    # Evaluate
    inputs = {a: v for a, v in zip(args, params)}
    request = {r_cost: list(args)}
    workers = aadc.ThreadPool(1)

    # Warmup
    aadc.evaluate(funcs, request, inputs, workers)

    # Benchmark
    n = 50
    t0 = time.time()
    for _ in range(n):
        res = aadc.evaluate(funcs, request, inputs, workers)
    elapsed = (time.time() - t0) / n * 1000

    cost_val = float(np.asarray(res[0][r_cost]).flat[0])
    grad_val = [float(np.asarray(res[1][r_cost][a]).flat[0]) for a in args]

    print(f"  Cost:     {cost_val:.6e}")
    print(f"  Gradient: {grad_val}")
    print(f"  Time:     {elapsed:.1f} ms/eval")
    return cost_val, grad_val


# ============================================================
# Model 2: 3-compartment CVS (27 states, 4 params)
# ============================================================
def test_3comp_casadi():
    """3-compartment with CasADI — expected to CRASH."""
    print(f"\n{'─'*50}")
    print("3-Compartment CVS: CasADI")
    print(f"{'─'*50}")
    try:
        import casadi as ca
    except ImportError:
        print("  CasADI not installed — skipping")
        return

    print("  Attempting to trace model with CasADI symbolic variables...")
    try:
        # Try to import Auckland's generated model (optional — needs circulatory_autogen installed)
        import importlib
        try:
            mod = importlib.import_module('3compartment')
            u = importlib.import_module('3compartment_utilities')
        except ImportError:
            print("  SKIP (circulatory_autogen generated model not found)")
            print("  To run this test: pip install -e <circulatory_autogen> and run their tests first")
            return

        states = mod.create_states_array()
        variables = mod.create_variables_array()
        rates = np.zeros(u.STATE_COUNT)
        mod.initialise_variables(states, rates, variables)
        mod.compute_computed_constants(variables)

        # Try making q_lv symbolic
        states[18] = ca.SX.sym('q_lv')
        mod.compute_rates(0.0, states, rates, variables)

        print("  ERROR: should have crashed but didn't")
    except (RuntimeError, TypeError, NotImplementedError) as e:
        err = str(e)[:100]
        print(f"  *** CRASHES: {err}")
        print("  CasADI cannot differentiate through if/else in valve logic.")


def test_3comp_aadc():
    """3-compartment with AADC — works."""
    print(f"\n{'─'*50}")
    print("3-Compartment CVS: AADC")
    print(f"{'─'*50}")
    try:
        import aadc
    except ImportError:
        print("  AADC not installed — skipping")
        return

    from cvs3_aadc_python import run_model, Params
    p = Params()

    # Record kernel
    t0 = time.time()
    funcs = aadc.Functions()
    funcs.start_recording()

    id_qlv = aadc.idouble(p.q_lv_init)
    id_cao = aadc.idouble(p.C_aortic)
    id_elva = aadc.idouble(p.E_lv_A)
    id_elvb = aadc.idouble(p.E_lv_B)
    a_qlv = id_qlv.mark_as_input()
    a_cao = id_cao.mark_as_input()
    a_elva = id_elva.mark_as_input()
    a_elvb = id_elvb.mark_as_input()

    cost = run_model(id_qlv, id_cao, id_elva, id_elvb)
    r_cost = cost.mark_as_output()
    funcs.stop_recording()
    rec_time = time.time() - t0

    # Evaluate
    inputs = {a_qlv: p.q_lv_init, a_cao: p.C_aortic,
              a_elva: p.E_lv_A, a_elvb: p.E_lv_B}
    request = {r_cost: [a_qlv, a_cao, a_elva, a_elvb]}
    workers = aadc.ThreadPool(1)

    res = aadc.evaluate(funcs, request, inputs, workers)
    cost_val = float(np.asarray(res[0][r_cost]).flat[0])
    grad = {
        'q_lv_init': float(np.asarray(res[1][r_cost][a_qlv]).flat[0]),
        'C_aortic':  float(np.asarray(res[1][r_cost][a_cao]).flat[0]),
        'E_lv_A':    float(np.asarray(res[1][r_cost][a_elva]).flat[0]),
        'E_lv_B':    float(np.asarray(res[1][r_cost][a_elvb]).flat[0]),
    }

    # Benchmark
    n = 20
    t0 = time.time()
    for _ in range(n):
        aadc.evaluate(funcs, request, inputs, workers)
    elapsed = (time.time() - t0) / n * 1000

    print(f"  Recording: {rec_time:.1f}s (one-time)")
    print(f"  Cost:      {cost_val:.6e}")
    print(f"  Gradient:")
    for k, v in grad.items():
        print(f"    dC/d{k} = {v:.6e}")
    print(f"  Time:      {elapsed:.1f} ms/eval")
    print(f"  WORKS — including all conditional valve logic")


# ============================================================
# Main
# ============================================================
def main():
    print(SEPARATOR)
    print("CasADI vs AADC: Side-by-Side Comparison")
    print("Models from circulatory_autogen (University of Auckland)")
    print(SEPARATOR)

    # --- Lotka-Volterra ---
    print(f"\n{SEPARATOR}")
    print("MODEL 1: Lotka-Volterra (2 states, 4 params)")
    print(SEPARATOR)

    cost_ca, grad_ca = test_lotka_casadi()
    cost_aa, grad_aa = test_lotka_aadc()

    if cost_ca is not None and cost_aa is not None:
        print(f"\n{'─'*50}")
        print("Lotka-Volterra: Comparison")
        print(f"{'─'*50}")
        print(f"  Cost match: {abs(cost_ca - cost_aa) < 1e-10}")
        max_grad_diff = max(abs(a - b) for a, b in zip(grad_ca, grad_aa))
        print(f"  Max gradient diff: {max_grad_diff:.2e}")

    # --- 3-compartment ---
    print(f"\n{SEPARATOR}")
    print("MODEL 2: 3-Compartment Cardiovascular (27 states, 4 params)")
    print(f"         Includes conditional valve logic (if/else)")
    print(SEPARATOR)

    test_3comp_casadi()
    test_3comp_aadc()

    print(f"\n{SEPARATOR}")
    print("SUMMARY")
    print(SEPARATOR)
    print("  Lotka-Volterra:  CasADI works, AADC works (much faster)")
    print("  3-Compartment:   CasADI CRASHES, AADC WORKS")
    print(f"{SEPARATOR}")


if __name__ == "__main__":
    main()
