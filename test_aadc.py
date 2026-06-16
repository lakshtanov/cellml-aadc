#!/usr/bin/env python3
"""
Tests for AADC CellML backend.

Run: python test_aadc.py
  or: pytest test_aadc.py -v

Tests mirror the structure of circulatory_autogen's test_param_id.py:
  1. Lotka-Volterra: AADC gradient matches CasADI gradient
  2. Lotka-Volterra: AADC gradient matches finite differences
  3. 3-compartment: CasADI crashes, AADC works
  4. 3-compartment: AADC gradient matches finite differences
  5. 3-compartment: Hessian is symmetric
  6. 3-compartment: HMC produces reasonable posterior
"""
import sys
import time
import math
import numpy as np

try:
    import aadc
    HAS_AADC = True
except ImportError:
    HAS_AADC = False

try:
    import casadi as ca
    HAS_CASADI = True
except ImportError:
    HAS_CASADI = False

# ---- Lotka-Volterra model (shared) ----
def lotka_rk4(alpha, beta, delta, gamma, dt=0.01, N=500):
    x, y = 1.0, 1.0
    for _ in range(N):
        dx1 = alpha*x - beta*x*y
        dy1 = delta*x*y - gamma*y
        x2, y2 = x + 0.5*dt*dx1, y + 0.5*dt*dy1
        dx2 = alpha*x2 - beta*x2*y2
        dy2 = delta*x2*y2 - gamma*y2
        x3, y3 = x + 0.5*dt*dx2, y + 0.5*dt*dy2
        dx3 = alpha*x3 - beta*x3*y3
        dy3 = delta*x3*y3 - gamma*y3
        x4, y4 = x + dt*dx3, y + dt*dy3
        dx4 = alpha*x4 - beta*x4*y4
        dy4 = delta*x4*y4 - gamma*y4
        x += dt/6*(dx1 + 2*dx2 + 2*dx3 + dx4)
        y += dt/6*(dy1 + 2*dy2 + 2*dy3 + dy4)
    return x*x + y*y

PARAMS_LV = [1.5, 1.0, 3.0, 1.0]

# ---- helpers ----
def aadc_lotka_gradient(params):
    """Record and evaluate AADC gradient for Lotka-Volterra."""
    funcs = aadc.Functions()
    funcs.start_recording()
    p = [aadc.idouble(v) for v in params]
    args = [pi.mark_as_input() for pi in p]
    cost = lotka_rk4(*p)
    r_cost = cost.mark_as_output()
    funcs.stop_recording()

    inputs = {a: v for a, v in zip(args, params)}
    request = {r_cost: list(args)}
    workers = aadc.ThreadPool(1)
    res = aadc.evaluate(funcs, request, inputs, workers)

    cost_val = float(np.asarray(res[0][r_cost]).flat[0])
    grad = [float(np.asarray(res[1][r_cost][a]).flat[0]) for a in args]
    return cost_val, grad


def fd_lotka_gradient(params, eps=1e-7):
    """Finite difference gradient for Lotka-Volterra."""
    cost0 = lotka_rk4(*params)
    grad = []
    for i in range(len(params)):
        p_up = list(params)
        p_up[i] += eps
        p_dn = list(params)
        p_dn[i] -= eps
        grad.append((lotka_rk4(*p_up) - lotka_rk4(*p_dn)) / (2 * eps))
    return cost0, grad


# ---- Tests ----
class Results:
    passed = 0
    failed = 0
    skipped = 0

def run_test(name, func):
    try:
        func()
        Results.passed += 1
        print(f"  PASS  {name}")
    except AssertionError as e:
        Results.failed += 1
        print(f"  FAIL  {name}: {e}")
    except Exception as e:
        Results.failed += 1
        print(f"  ERROR {name}: {e}")


def test_lotka_aadc_vs_casadi():
    """Lotka-Volterra: AADC gradient matches CasADI."""
    if not HAS_AADC:
        Results.skipped += 1; print("  SKIP  (no aadc)"); return
    if not HAS_CASADI:
        Results.skipped += 1; print("  SKIP  (no casadi)"); return

    # CasADI
    alpha, beta, delta, gamma = [ca.SX.sym(n) for n in ['a','b','d','g']]
    cost_sx = lotka_rk4(alpha, beta, delta, gamma)
    grad_sx = ca.gradient(cost_sx, ca.vertcat(alpha, beta, delta, gamma))
    f = ca.Function('f', [alpha, beta, delta, gamma], [cost_sx, grad_sx])
    res_ca = f(*PARAMS_LV)
    cost_ca = float(res_ca[0])
    grad_ca = [float(res_ca[1][i]) for i in range(4)]

    # AADC
    cost_aa, grad_aa = aadc_lotka_gradient(PARAMS_LV)

    assert abs(cost_ca - cost_aa) < 1e-12, f"cost diff: {abs(cost_ca - cost_aa)}"
    for i in range(4):
        assert abs(grad_ca[i] - grad_aa[i]) < 1e-10, \
            f"grad[{i}] diff: {abs(grad_ca[i] - grad_aa[i])}"


def test_lotka_aadc_vs_fd():
    """Lotka-Volterra: AADC gradient matches finite differences."""
    if not HAS_AADC:
        Results.skipped += 1; print("  SKIP  (no aadc)"); return

    cost_aa, grad_aa = aadc_lotka_gradient(PARAMS_LV)
    cost_fd, grad_fd = fd_lotka_gradient(PARAMS_LV)

    assert abs(cost_aa - cost_fd) < 1e-12, f"cost diff: {abs(cost_aa - cost_fd)}"
    for i in range(4):
        ratio = grad_aa[i] / grad_fd[i] if grad_fd[i] != 0 else 1.0
        assert abs(ratio - 1.0) < 1e-6, f"grad[{i}] ratio: {ratio}"


def test_3comp_casadi_crashes():
    """3-compartment: CasADI crashes on conditional valve logic."""
    if not HAS_CASADI:
        Results.skipped += 1; print("  SKIP  (no casadi)"); return

    try:
        import importlib
        mod = importlib.import_module('3compartment')
        u = importlib.import_module('3compartment_utilities')
    except ImportError:
        Results.skipped += 1; print("  SKIP  (circulatory_autogen model not found)"); return
    try:

        states = mod.create_states_array()
        variables = mod.create_variables_array()
        rates = np.zeros(u.STATE_COUNT)
        mod.initialise_variables(states, rates, variables)
        mod.compute_computed_constants(variables)

        states[18] = ca.SX.sym('q_lv')
        mod.compute_rates(0.0, states, rates, variables)
        raise AssertionError("CasADI should have crashed but didn't")
    except (RuntimeError, TypeError):
        pass  # expected crash


def test_3comp_aadc_works():
    """3-compartment: AADC produces gradient (where CasADI crashes)."""
    if not HAS_AADC:
        Results.skipped += 1; print("  SKIP  (no aadc)"); return

    from cvs3_aadc_python import run_model, Params
    p = Params()

    funcs = aadc.Functions()
    funcs.start_recording()
    id_qlv = aadc.idouble(p.q_lv_init)
    id_cao = aadc.idouble(p.C_aortic)
    a_qlv = id_qlv.mark_as_input()
    a_cao = id_cao.mark_as_input()
    cost = run_model(id_qlv, id_cao, aadc.idouble(p.E_lv_A), aadc.idouble(p.E_lv_B))
    r_cost = cost.mark_as_output()
    funcs.stop_recording()

    inputs = {a_qlv: p.q_lv_init, a_cao: p.C_aortic}
    request = {r_cost: [a_qlv, a_cao]}
    workers = aadc.ThreadPool(1)
    res = aadc.evaluate(funcs, request, inputs, workers)

    cost_val = float(np.asarray(res[0][r_cost]).flat[0])
    g_qlv = float(np.asarray(res[1][r_cost][a_qlv]).flat[0])
    g_cao = float(np.asarray(res[1][r_cost][a_cao]).flat[0])

    assert math.isfinite(cost_val), f"cost is not finite: {cost_val}"
    assert math.isfinite(g_qlv), f"dC/dq_lv is not finite: {g_qlv}"
    assert math.isfinite(g_cao), f"dC/dC_ao is not finite: {g_cao}"
    assert cost_val > 0, f"cost should be positive: {cost_val}"


def test_3comp_aadc_gradient_vs_fd():
    """3-compartment: AADC gradient matches finite differences."""
    if not HAS_AADC:
        Results.skipped += 1; print("  SKIP  (no aadc)"); return

    from cvs3_aadc_python import run_model, Params
    p = Params()

    # Record kernel with 2 params
    funcs = aadc.Functions()
    funcs.start_recording()
    id_qlv = aadc.idouble(p.q_lv_init)
    id_cao = aadc.idouble(p.C_aortic)
    a_qlv = id_qlv.mark_as_input()
    a_cao = id_cao.mark_as_input()
    cost = run_model(id_qlv, id_cao, aadc.idouble(p.E_lv_A), aadc.idouble(p.E_lv_B))
    r_cost = cost.mark_as_output()
    funcs.stop_recording()

    inputs = {a_qlv: p.q_lv_init, a_cao: p.C_aortic}
    request = {r_cost: [a_qlv, a_cao]}
    workers = aadc.ThreadPool(1)
    res = aadc.evaluate(funcs, request, inputs, workers)

    ad_qlv = float(np.asarray(res[1][r_cost][a_qlv]).flat[0])
    ad_cao = float(np.asarray(res[1][r_cost][a_cao]).flat[0])

    # FD via AADC forward
    eps = 1e-6
    for param_arg, param_val, ad_val, name in [
        (a_qlv, p.q_lv_init, ad_qlv, "q_lv"),
        (a_cao, p.C_aortic, ad_cao, "C_ao"),
    ]:
        h = param_val * eps
        inp_up = dict(inputs); inp_up[param_arg] = param_val + h
        inp_dn = dict(inputs); inp_dn[param_arg] = param_val - h
        r_up = aadc.evaluate(funcs, {r_cost: []}, inp_up, workers)
        r_dn = aadc.evaluate(funcs, {r_cost: []}, inp_dn, workers)
        c_up = float(np.asarray(r_up[0][r_cost]).flat[0])
        c_dn = float(np.asarray(r_dn[0][r_cost]).flat[0])
        fd = (c_up - c_dn) / (2 * h)
        ratio = ad_val / fd if fd != 0 else 1.0
        assert abs(ratio - 1.0) < 1e-4, f"dC/d{name}: AD/FD ratio = {ratio}"


def test_3comp_hessian_symmetric():
    """3-compartment: Hessian is symmetric."""
    if not HAS_AADC:
        Results.skipped += 1; print("  SKIP  (no aadc)"); return

    from cvs3_aadc_python import run_model, Params
    p = Params()

    funcs = aadc.Functions()
    funcs.start_recording()
    id_qlv = aadc.idouble(p.q_lv_init)
    id_cao = aadc.idouble(p.C_aortic)
    a_qlv = id_qlv.mark_as_input()
    a_cao = id_cao.mark_as_input()
    cost = run_model(id_qlv, id_cao, aadc.idouble(p.E_lv_A), aadc.idouble(p.E_lv_B))
    r_cost = cost.mark_as_output()
    funcs.stop_recording()

    args = [a_qlv, a_cao]
    vals = [p.q_lv_init, p.C_aortic]
    workers = aadc.ThreadPool(1)
    eps = 1e-5

    hess = np.zeros((2, 2))
    for i in range(2):
        h = vals[i] * eps
        inp_up = {a: v for a, v in zip(args, vals)}; inp_up[args[i]] = vals[i] + h
        inp_dn = {a: v for a, v in zip(args, vals)}; inp_dn[args[i]] = vals[i] - h

        r_up = aadc.evaluate(funcs, {r_cost: args}, inp_up, workers)
        r_dn = aadc.evaluate(funcs, {r_cost: args}, inp_dn, workers)

        for j in range(2):
            g_up = float(np.asarray(r_up[1][r_cost][args[j]]).flat[0])
            g_dn = float(np.asarray(r_dn[1][r_cost][args[j]]).flat[0])
            hess[i, j] = (g_up - g_dn) / (2 * h)

    # Check symmetry
    asym = abs(hess[0, 1] - hess[1, 0])
    avg = 0.5 * (abs(hess[0, 1]) + abs(hess[1, 0]))
    rel_asym = asym / avg if avg > 0 else 0
    assert rel_asym < 0.01, f"Hessian asymmetry: {rel_asym:.4e}"


# ---- SimulationHelper / RK45 / discrete adjoint tests ----

def _make_lv_model():
    """Build a minimal in-memory CellML-style module for Lotka-Volterra."""
    import types
    mod = types.ModuleType("lv_model")
    mod.STATE_COUNT = 2
    mod.VARIABLE_INFO = [
        {"type": type("T", (), {"name": "CONSTANT"})()},  # alpha
        {"type": type("T", (), {"name": "CONSTANT"})()},  # beta
        {"type": type("T", (), {"name": "CONSTANT"})()},  # delta
        {"type": type("T", (), {"name": "CONSTANT"})()},  # gamma
    ]
    mod.STATE_INFO = [
        {"name": "x", "units": "dimensionless"},
        {"name": "y", "units": "dimensionless"},
    ]

    def create_states_array():
        return [1.0, 1.0]

    def initialise_variables(states, rates, variables):
        states[0] = 1.0
        states[1] = 1.0
        variables[0] = 1.5  # alpha
        variables[1] = 1.0  # beta
        variables[2] = 3.0  # delta
        variables[3] = 1.0  # gamma

    def compute_computed_constants(variables):
        pass

    def compute_rates(t, states, rates, variables):
        x, y = states[0], states[1]
        alpha, beta, delta, gamma = variables[0], variables[1], variables[2], variables[3]
        rates[0] = alpha * x - beta * x * y
        rates[1] = delta * x * y - gamma * y

    mod.create_states_array = create_states_array
    mod.initialise_variables = initialise_variables
    mod.compute_computed_constants = compute_computed_constants
    mod.compute_rates = compute_rates
    return mod


def _make_lv_helper():
    """Build a SimulationHelper wrapping the minimal LV model."""
    import tempfile, os, types
    from aadc_solver_helper import SimulationHelper

    # Write a minimal model file (SimulationHelper loads via file path)
    src = """\
import math
STATE_COUNT = 2
VARIABLE_INFO = [
    type('T', (), {'type': type('V', (), {'name': 'CONSTANT'})()})(),
    type('T', (), {'type': type('V', (), {'name': 'CONSTANT'})()})(),
    type('T', (), {'type': type('V', (), {'name': 'CONSTANT'})()})(),
    type('T', (), {'type': type('V', (), {'name': 'CONSTANT'})()})(),
]
STATE_INFO = [
    {'name': 'x', 'units': 'dimensionless'},
    {'name': 'y', 'units': 'dimensionless'},
]
def create_states_array(): return [1.0, 1.0]
def initialise_variables(s, r, v): s[0]=1.0; s[1]=1.0; v[0]=1.5; v[1]=1.0; v[2]=3.0; v[3]=1.0
def compute_computed_constants(v): pass
def compute_rates(t, s, r, v):
    x, y = s[0], s[1]
    r[0] = v[0]*x - v[1]*x*y
    r[1] = v[2]*x*y - v[3]*y
"""
    fd, path = tempfile.mkstemp(suffix=".py")
    os.write(fd, src.encode())
    os.close(fd)
    return path


def test_rk45_vs_scipy():
    """RK45 trajectory matches scipy solve_ivp(RK45) on Lotka-Volterra."""
    try:
        from scipy.integrate import solve_ivp
    except ImportError:
        Results.skipped += 1; print("  SKIP  (no scipy)"); return
    if not HAS_AADC:
        Results.skipped += 1; print("  SKIP  (no aadc)"); return

    import os
    from aadc_solver_helper import SimulationHelper

    path = _make_lv_helper()
    try:
        dt = 0.01
        sim_time = 5.0
        h = SimulationHelper(path, dt=dt, sim_time=sim_time, pre_time=0.0)
        h.run()
        aadc_x = h.state_traj[0, :]
        aadc_y = h.state_traj[1, :]

        def lv_rhs(t, s):
            return [1.5*s[0] - 1.0*s[0]*s[1], 3.0*s[0]*s[1] - 1.0*s[1]]

        sol = solve_ivp(lv_rhs, [0.0, sim_time], [1.0, 1.0], method='RK45',
                        t_eval=h.tSim, rtol=1e-8, atol=1e-10)

        diff_x = np.max(np.abs(aadc_x - sol.y[0]))
        diff_y = np.max(np.abs(aadc_y - sol.y[1]))
        assert diff_x < 1e-5, f"x trajectory max diff vs scipy: {diff_x:.2e}"
        assert diff_y < 1e-5, f"y trajectory max diff vs scipy: {diff_y:.2e}"
    finally:
        os.unlink(path)


def test_compute_gradient_vs_fd():
    """compute_gradient() matches central FD to ratio ~1 on Lotka-Volterra."""
    if not HAS_AADC:
        Results.skipped += 1; print("  SKIP  (no aadc)"); return

    import os
    from aadc_solver_helper import SimulationHelper

    path = _make_lv_helper()
    try:
        dt = 0.02
        sim_time = 2.0

        def run_cost(alpha_val, beta_val):
            h = SimulationHelper(path, dt=dt, sim_time=sim_time, pre_time=0.0)
            # manually override constants (var index 0 = alpha, 1 = beta)
            h.variables[0] = alpha_val
            h.variables[1] = beta_val
            h.run()
            x_T = h.state_traj[:, -1]
            return float(x_T[0]**2 + x_T[1]**2)

        alpha0, beta0 = 1.5, 1.0
        eps = 1e-5

        # FD gradients
        fd_alpha = (run_cost(alpha0 + eps, beta0) - run_cost(alpha0 - eps, beta0)) / (2 * eps)
        fd_beta = (run_cost(alpha0, beta0 + eps) - run_cost(alpha0, beta0 - eps)) / (2 * eps)

        # Discrete adjoint gradient
        # param names: SimulationHelper uses VariableNameResolver which is None
        # in standalone mode — skip this test if resolver unavailable
        h = SimulationHelper(path, dt=dt, sim_time=sim_time, pre_time=0.0)
        if h._resolver is None:
            # Can't resolve by name without circulatory_autogen — use index-based workaround
            # by directly calling the internal methods with known var indices
            Results.skipped += 1
            print("  SKIP  (no VariableNameResolver in standalone mode)")
            return

        h._ad_param_names = ["alpha", "beta"]
        h._ad_param_var_indices = [0, 1]
        h._do_ad = True
        variables_all = list(h._numeric_variables_all)
        for ci, vi in enumerate(h.constant_indices):
            variables_all[vi] = h.variables[ci]
        h._record_rhs_aad(variables_all)
        h.run()

        def cost_func(x_T):
            return float(x_T[0]**2 + x_T[1]**2)

        grad = h.compute_gradient(cost_func)
        ratio_alpha = grad[0] / fd_alpha if abs(fd_alpha) > 1e-12 else 1.0
        ratio_beta = grad[1] / fd_beta if abs(fd_beta) > 1e-12 else 1.0
        assert abs(ratio_alpha - 1.0) < 0.01, f"d/d_alpha ratio={ratio_alpha:.6f}"
        assert abs(ratio_beta - 1.0) < 0.01, f"d/d_beta ratio={ratio_beta:.6f}"
    finally:
        os.unlink(path)


def test_reset_and_gradient_correctness():
    """After reset_and_clear + re-run, compute_gradient still correct (catches Bug 2)."""
    if not HAS_AADC:
        Results.skipped += 1; print("  SKIP  (no aadc)"); return

    import os
    from aadc_solver_helper import SimulationHelper

    path = _make_lv_helper()
    try:
        dt = 0.02
        sim_time = 2.0
        eps = 1e-5

        def run_cost_fd(alpha_val):
            h2 = SimulationHelper(path, dt=dt, sim_time=sim_time, pre_time=0.0)
            h2.variables[0] = alpha_val
            h2.run()
            x_T = h2.state_traj[:, -1]
            return float(x_T[0]**2 + x_T[1]**2)

        fd_alpha = (run_cost_fd(1.5 + eps) - run_cost_fd(1.5 - eps)) / (2 * eps)

        h = SimulationHelper(path, dt=dt, sim_time=sim_time, pre_time=0.0)
        if h._resolver is None:
            Results.skipped += 1
            print("  SKIP  (no VariableNameResolver in standalone mode)")
            return

        # First run with adjoint
        h._ad_param_names = ["alpha", "beta"]
        h._ad_param_var_indices = [0, 1]
        h._do_ad = True
        variables_all = list(h._numeric_variables_all)
        for ci, vi in enumerate(h.constant_indices):
            variables_all[vi] = h.variables[ci]
        h._record_rhs_aad(variables_all)
        h.run()
        grad1 = h.compute_gradient(lambda x: float(x[0]**2 + x[1]**2))

        # Reset and redo — should NOT use stale _rk_data or _aad_funcs
        h.reset_and_clear()
        h._ad_param_names = ["alpha", "beta"]
        h._ad_param_var_indices = [0, 1]
        h._do_ad = True
        variables_all = list(h._numeric_variables_all)
        for ci, vi in enumerate(h.constant_indices):
            variables_all[vi] = h.variables[ci]
        h._record_rhs_aad(variables_all)
        h.run()
        grad2 = h.compute_gradient(lambda x: float(x[0]**2 + x[1]**2))

        # Both runs should match each other and FD
        assert abs(grad1[0] - grad2[0]) < 1e-10, f"grad[alpha] changed after reset: {grad1[0]} vs {grad2[0]}"
        ratio = grad2[0] / fd_alpha if abs(fd_alpha) > 1e-12 else 1.0
        assert abs(ratio - 1.0) < 0.01, f"post-reset d/d_alpha ratio={ratio:.6f}"
    finally:
        os.unlink(path)


def main():
    print("=" * 60)
    print("AADC CellML Backend Tests")
    print("=" * 60)

    tests = [
        ("Lotka-Volterra: AADC vs CasADI", test_lotka_aadc_vs_casadi),
        ("Lotka-Volterra: AADC vs FD", test_lotka_aadc_vs_fd),
        ("3-compartment: CasADI crashes", test_3comp_casadi_crashes),
        ("3-compartment: AADC works", test_3comp_aadc_works),
        ("3-compartment: AADC gradient vs FD", test_3comp_aadc_gradient_vs_fd),
        ("3-compartment: Hessian symmetric", test_3comp_hessian_symmetric),
        ("RK45 vs scipy solve_ivp", test_rk45_vs_scipy),
        ("compute_gradient vs FD", test_compute_gradient_vs_fd),
        ("reset_and_clear gradient correctness", test_reset_and_gradient_correctness),
    ]

    for name, func in tests:
        run_test(name, func)

    print(f"\n{'=' * 60}")
    print(f"Results: {Results.passed} passed, {Results.failed} failed, {Results.skipped} skipped")
    print(f"{'=' * 60}")
    return 1 if Results.failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
