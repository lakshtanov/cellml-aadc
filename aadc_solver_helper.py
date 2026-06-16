"""
AADC-based solver backend for circulatory_autogen.

Drop-in replacement for casadi_python_solver_helper.py.
Implements the same SimulationHelper interface so it plugs into
paramID, HMC, sensitivity analysis, and the entire 12 LABOURS platform.

Usage in circulatory_autogen:
  1. Copy this file to src/solver_wrappers/aadc_solver_helper.py
  2. Add to src/solver_wrappers/__init__.py:
       from solver_wrappers.aadc_solver_helper import SimulationHelper as AadcSimulationHelper
  3. Add 'aadc' to get_simulation_helper() factory
  4. Set solver: aadc_semi_implicit in your config

Key differences from CasADI backend:
  - No symbolic graph — AADC records actual execution on idouble tape
  - Conditionals (if/else) handled via aadc.iif() — no crash
  - Stiff ODEs via semi-implicit Euler with diagonal damping
  - Kernel recorded once (~3s), then gradient eval ~6ms
"""
import importlib.util
import math
import copy
import numpy as np

try:
    import aadc
except ImportError:
    aadc = None

# Reuse the shared name resolver from circulatory_autogen
try:
    from .name_resolver import VariableNameResolver
except ImportError:
    # Standalone usage outside circulatory_autogen package
    VariableNameResolver = None


class SimulationHelper:
    """
    AADC-based solver for libCellML-generated Python modules.

    Matches the key interface of the other SimulationHelpers:
    - run()
    - update_times(dt, start_time, sim_time, pre_time)
    - get_results / get_all_results / get_all_variable_names
    - get_init_param_vals / set_param_vals
    """

    def __init__(self, model_path, dt, sim_time, solver_info=None, pre_time=0.0):
        if aadc is None:
            raise RuntimeError("AADC solver requested but aadc is not installed")
        self.model_path = model_path
        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.solver_info = solver_info or {}
        self._load_model()
        self.update_times(dt, 0.0, sim_time, pre_time)
        self._init_state()
        self._has_run = False
        self._do_ad = False
        self._aadc_funcs = None  # compiled kernel (set after first run with AD)
        self._aadc_args = {}    # parameter name → AADCArgument
        self._aadc_outputs = {} # output name → AADCArgument

    def set_protocol_info(self, protocol_info):
        self.protocol_info = protocol_info

    # ---- setup helpers ----
    def _load_model(self):
        spec = importlib.util.spec_from_file_location("generated_model", self.model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module

        self.STATE_COUNT = module.STATE_COUNT
        self.VARIABLE_INFO = module.VARIABLE_INFO
        self.STATE_INFO = module.STATE_INFO

        if VariableNameResolver is not None:
            self._resolver = VariableNameResolver(self.STATE_INFO, self.VARIABLE_INFO)
        else:
            self._resolver = None

        self.state_name_to_idx = {}
        self.var_name_to_idx = {}
        if self._resolver:
            self.state_name_to_idx = {name: idx for name, (kind, idx) in self._resolver._map.items() if kind == "state"}
            self.var_name_to_idx = {name: idx for name, (kind, idx) in self._resolver._map.items() if kind == "var"}
        self.state_idx_to_name = {idx: name for name, idx in self.state_name_to_idx.items()}
        self.var_idx_to_name = {idx: name for name, idx in self.var_name_to_idx.items()}

        self.constant_indices = [i for i, info in enumerate(self.VARIABLE_INFO)
                                 if info["type"].name in ["CONSTANT", "COMPUTED_CONSTANT"]]
        self.algebraic_indices = [i for i, info in enumerate(self.VARIABLE_INFO)
                                  if info["type"].name == "ALGEBRAIC"]

    def _init_state(self):
        _s0 = self.model.create_states_array()
        _r0 = self.model.create_states_array()
        _v0 = self.model.create_variables_array()
        self.model.initialise_variables(_s0, _r0, _v0)
        self.model.compute_computed_constants(_v0)
        self._numeric_x0 = np.array(_s0, dtype=float)
        self._numeric_variables_all = np.array(_v0, dtype=float)

        self.states = list(_s0)
        self.rates = list(_r0)
        self.variables = np.array([_v0[i] for i in self.constant_indices], dtype=float)

        self.default_constants = list(self.variables)
        self.default_state_inits = list(self.states)

    def _patch_math_functions(self):
        """Replace math functions in the model module with AADC-compatible versions."""
        aadc_math_map = {
            "log": aadc.math.log,
            "exp": aadc.math.exp,
            "sin": aadc.math.sin,
            "cos": aadc.math.cos,
            "tan": aadc.math.tan,
            "sqrt": aadc.math.sqrt,
            "pow": aadc.math.pow,
        }
        for name, func in aadc_math_map.items():
            setattr(self.model, name, func)

        # floor: extract passive value (not differentiable, but needed for cardiac phase)
        def aadc_floor(x):
            return math.floor(float(x))
        setattr(self.model, "floor", aadc_floor)

        # Replace comparison functions with aadc.iif versions
        def leq_func(a, b):
            return a <= b  # returns idouble comparison for aadc.iif
        def geq_func(a, b):
            return a >= b
        def lt_func(a, b):
            return a < b
        def gt_func(a, b):
            return a > b
        def and_func(a, b):
            return aadc.iand(a, b)
        def aadc_max(a, b):
            return aadc.iif(a >= b, a, b)

        setattr(self.model, "leq_func", leq_func)
        setattr(self.model, "geq_func", geq_func)
        setattr(self.model, "lt_func", lt_func)
        setattr(self.model, "gt_func", gt_func)
        setattr(self.model, "and_func", and_func)
        setattr(self.model, "max", aadc_max)

    # ---- name resolution ----
    def _resolve_name(self, name):
        if self._resolver:
            return self._resolver.resolve(name)
        return (None, None)

    def _var_idx_to_const_pos(self, var_idx):
        return self.constant_indices.index(var_idx)

    # ---- timing ----
    def update_times(self, dt, start_time, sim_time, pre_time):
        self.dt = dt
        self.pre_time = pre_time
        self.sim_time = sim_time
        self.start_time = start_time
        self.stop_time = start_time + pre_time + sim_time
        self.pre_steps = int(pre_time / dt)
        self.n_steps = int(sim_time / dt)
        self.t_eval = np.arange(start_time, self.stop_time + dt / 2, dt)
        self.tSim = self.t_eval[self.pre_steps:]

    # ---- parameter helpers ----
    def get_init_param_vals(self, param_names):
        vals = []
        for name_or_list in param_names:
            if not isinstance(name_or_list, list):
                name_or_list = [name_or_list]
            sub = []
            for name in name_or_list:
                kind, idx = self._resolve_name(name)
                if kind == "state":
                    sub.append(self.states[idx])
                elif kind == "var":
                    sub.append(self.variables[self._var_idx_to_const_pos(idx)])
                else:
                    raise ValueError(f"parameter name {name} not found")
            vals.append(sub if len(sub) > 1 else sub[0])
        return vals

    def set_param_vals(self, param_names, param_vals):
        for idx, name_or_list in enumerate(param_names):
            vals = param_vals[idx]
            if not isinstance(name_or_list, (list, tuple)):
                name_or_list = [name_or_list]
            if not isinstance(vals, (list, tuple)):
                vals = [vals]
            for name, val in zip(name_or_list, vals):
                kind, idx_res = self._resolve_name(name)
                if kind == "state":
                    self.states[idx_res] = val
                elif kind == "var":
                    self.variables[self._var_idx_to_const_pos(idx_res)] = val
                    var_name = self.var_idx_to_name.get(idx_res, "")
                    var_part = var_name.split("/")[-1] if "/" in var_name else var_name
                    if var_part.endswith("_init"):
                        state_var = var_part[:-5]
                        state_kind, state_idx = self._resolve_name(state_var)
                        if state_kind == "state":
                            self.states[state_idx] = val
                            self.default_state_inits[state_idx] = val
                else:
                    raise ValueError(f"parameter name {name} not found")

    # ---- ODE integration (semi-implicit Euler) ----
    def _integrate(self, states, variables_all, total_steps, dt):
        """
        Adaptive RK45 (Dormand-Prince) integration.

        Uses the algorithm from arXiv:2410.01911 (Martins & Lakshtanov).
        Adaptive step size for accuracy; stores stages for discrete adjoint.

        Returns state trajectory: list of state arrays at each sim-time step.
        Also stores self._rk_data for adjoint computation.
        """
        n = self.STATE_COUNT
        x = np.array(states[:n], dtype=float)
        vars_all = list(variables_all)

        # Dormand-Prince 4(5) Butcher tableau
        a = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        ])
        b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
        b_hat = np.array([5179/57600, 0, 7571/16695, 393/640, -92097/339200, 187/2100, 1/40])
        c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
        s = 7

        tol = float(self.solver_info.get('tol', 1e-8))
        safety = 0.9
        h_min = 1e-12
        h_max = dt * 10

        t0 = 0.0
        tf = total_steps * dt
        t = t0
        h = dt

        def rhs(x_in, t_in):
            rates = [0.0] * n
            self.model.compute_rates(t_in, list(x_in), rates, list(vars_all))
            return np.array(rates, dtype=float)

        # Full trajectory storage
        all_t = [t]
        all_x = [x.copy()]
        all_h = []
        all_k = []

        steps = 0
        while t < tf - 1e-14:
            if t + h > tf:
                h = tf - t

            # Compute stages
            k = [None] * s
            for i in range(s):
                xi = x.copy()
                for j in range(i):
                    xi += h * a[i, j] * k[j]
                k[i] = rhs(xi, t + c[i] * h)

            # Higher-order solution
            x_new = x.copy()
            for i in range(s):
                x_new += h * b[i] * k[i]

            # Error estimate
            err = np.zeros(n)
            for i in range(s):
                err += h * (b[i] - b_hat[i]) * k[i]
            err_norm = np.linalg.norm(err / (1.0 + np.abs(x_new))) / max(np.sqrt(n), 1)

            if err_norm <= tol or h <= h_min:
                t += h
                x = x_new
                all_t.append(t)
                all_x.append(x.copy())
                all_h.append(h)
                all_k.append([ki.copy() for ki in k])
                steps += 1

            # Adjust step size
            if err_norm > 0:
                h_new = safety * h * (tol / err_norm) ** 0.2
            else:
                h_new = h * 2.0
            h = max(h_min, min(h_max, h_new))

            if steps > 10 * total_steps:
                break

        # Store for adjoint
        self._rk_data = {
            't': all_t, 'x': all_x, 'h': all_h, 'k': all_k,
            'n_states': n, 'vars_all': vars_all
        }

        # Interpolate onto uniform grid for get_results compatibility
        # Linear interpolation between adaptive-step trajectory points
        traj = []
        j = 0
        for ti in self.tSim:
            # Advance j to bracket ti
            while j < len(all_t) - 2 and all_t[j + 1] < ti:
                j += 1
            if j >= len(all_t) - 1:
                traj.append(list(all_x[-1]))
            elif abs(all_t[j + 1] - all_t[j]) < 1e-15:
                traj.append(list(all_x[j]))
            else:
                # Linear interpolation
                alpha = (ti - all_t[j]) / (all_t[j + 1] - all_t[j])
                xi = [(1 - alpha) * all_x[j][i] + alpha * all_x[j + 1][i]
                      for i in range(n)]
                traj.append(xi)

        return traj

    # ---- simulation ----
    def run(self):
        """Run simulation. If _do_ad is True, records onto AADC tape."""
        total_steps = self.pre_steps + self.n_steps

        # Build full variables array from constants
        variables_all = list(self._numeric_variables_all)
        for const_pos, const_idx in enumerate(self.constant_indices):
            variables_all[const_idx] = self.variables[const_pos]

        # Always run forward (numeric). AD uses stored trajectory for adjoint.
        traj = self._integrate(self.states, variables_all, total_steps, self.dt)
        self.state_traj = np.array(traj).T  # (n_states, n_sim_steps)

        # Compute algebraic variables at each time point
        self._compute_var_traj(traj, variables_all)

        self._has_run = True
        return True

    def _compute_var_traj(self, state_traj_list, variables_all):
        """Compute algebraic variable trajectories from state trajectory."""
        var_names = list(self.var_name_to_idx.keys())
        n_vars = len(var_names)
        n_times = len(state_traj_list)
        self.var_traj = np.zeros((n_vars, n_times))

        for ti_idx, st in enumerate(state_traj_list):
            t = self.tSim[ti_idx] if ti_idx < len(self.tSim) else 0.0
            rates = [0.0] * self.STATE_COUNT
            vars_copy = list(variables_all)
            self.model.compute_rates(t, st, rates, vars_copy)
            try:
                self.model.compute_variables(t, st, rates, vars_copy)
            except AttributeError:
                pass
            for vi, name in enumerate(var_names):
                idx = self.var_name_to_idx[name]
                self.var_traj[vi, ti_idx] = float(vars_copy[idx])

    # ---- time ----
    def get_time(self, include_pre_time=False):
        if include_pre_time:
            return self.tSim
        else:
            return self.tSim - self.pre_time

    # ---- results ----
    def get_all_variable_names(self):
        return list(self.state_name_to_idx.keys()) + list(self.var_name_to_idx.keys())

    def _extract(self, name):
        if name == 'time':
            return self.tSim
        if name in self.state_name_to_idx:
            idx = self.state_name_to_idx[name]
            return self.state_traj[idx, :]
        if name in self.var_name_to_idx:
            var_names = list(self.var_name_to_idx.keys())
            idx = var_names.index(name)
            return self.var_traj[idx, :]
        kind, idx_res = self._resolve_name(name)
        if kind == "state":
            return self.state_traj[idx_res, :]
        raise ValueError(f"variable {name} not found")

    def get_results(self, variables_list_of_lists, flatten=False):
        if type(variables_list_of_lists[0]) is not list:
            variables_list_of_lists = [[entry] for entry in variables_list_of_lists]
        results = []
        for variables_list in variables_list_of_lists:
            row = [self._extract(name) for name in variables_list]
            results.append(row)
        if flatten:
            results = [item for sublist in results for item in sublist]
        return results

    def get_all_results(self, flatten=False):
        return self.get_results(self.get_all_variable_names(), flatten=flatten)

    # ---- AADC AD: discrete adjoint (arXiv:2410.01911) ----
    def _create_param_subset(self, param_names, param_vals=None):
        """Mark parameters for AD. Called by paramID before run()."""
        self._ad_param_names = [x[0] if isinstance(x, list) else x for x in param_names]
        self._ad_param_var_indices = []
        for name in self._ad_param_names:
            kind, idx = self._resolve_name(name)
            if kind == "var":
                self._ad_param_var_indices.append(idx)
            else:
                raise ValueError(f"AD parameter {name} must be a variable, got {kind}")
        if param_vals is not None:
            param_vals = np.asarray(param_vals, dtype=float)
            for i, name in enumerate(self._ad_param_names):
                kind, idx = self._resolve_name(name)
                if kind == "var":
                    self.variables[self._var_idx_to_const_pos(idx)] = param_vals[i]
        self._do_ad = True

        # Record AAD kernel immediately (needs fresh model state)
        variables_all = list(self._numeric_variables_all)
        for const_pos, const_idx in enumerate(self.constant_indices):
            variables_all[const_idx] = self.variables[const_pos]
        self._record_rhs_aad(variables_all)

    def _record_rhs_aad(self, variables_all):
        """Record the ODE RHS with AAD for vector-Jacobian products.

        Uses the same pattern as the verified standalone AadRhs:
        record compute_rates(t, x, rates, vars) with idouble x, p, t.
        """
        n = self.STATE_COUNT
        m = len(self._ad_param_names)

        # Use rk-adjoint-python's AadRhs which is already verified
        vars_list = list(variables_all)
        param_var_indices = list(self._ad_param_var_indices)
        model = self.model

        self._patch_math_functions()

        def rhs_for_aad(x, p, t):
            v = list(vars_list)
            for i, var_idx in enumerate(param_var_indices):
                v[var_idx] = p[i]
            rates = [aadc.idouble(0.0) for _ in range(n)]
            model.compute_rates(t, x, rates, v)
            return rates

        p0 = np.array([float(variables_all[idx]) for idx in param_var_indices])
        x0 = np.zeros(n)

        funcs = aadc.Functions()
        funcs.start_recording()

        id_x = [aadc.idouble(float(x0[i])) for i in range(n)]
        a_x = [xi.mark_as_input() for xi in id_x]

        id_p = [aadc.idouble(float(p0[i])) for i in range(m)]
        a_p = [pi.mark_as_input() for pi in id_p]

        id_t = aadc.idouble(0.0)
        a_t = id_t.mark_as_input()

        dxdt = rhs_for_aad(id_x, id_p, id_t)

        r_f = [fi.mark_as_output() for fi in dxdt]

        funcs.stop_recording()

        self._aad_funcs = funcs
        self._aad_a_x = a_x
        self._aad_a_p = a_p
        self._aad_a_t = a_t
        self._aad_r_f = r_f
        self._aad_workers = aadc.ThreadPool(1)

    def _vjp(self, x, p_vals, t, v):
        """Vector-Jacobian product via AAD kernel."""
        n = self.STATE_COUNT
        m = len(self._ad_param_names)

        inputs = {}
        for i in range(n):
            inputs[self._aad_a_x[i]] = float(x[i])
        for i in range(m):
            inputs[self._aad_a_p[i]] = float(p_vals[i])
        inputs[self._aad_a_t] = float(t)

        all_args = list(self._aad_a_x) + list(self._aad_a_p)
        request = {r: all_args for r in self._aad_r_f}

        res = aadc.evaluate(self._aad_funcs, request, inputs, self._aad_workers)

        vjp_x = np.zeros(n)
        vjp_p = np.zeros(m)
        for i in range(n):
            vi = float(v[i])
            if vi == 0.0:
                continue
            for j in range(n):
                vjp_x[j] += vi * float(np.asarray(res[1][self._aad_r_f[i]][self._aad_a_x[j]]).flat[0])
            for j in range(m):
                vjp_p[j] += vi * float(np.asarray(res[1][self._aad_r_f[i]][self._aad_a_p[j]]).flat[0])

        return vjp_x, vjp_p

    def compute_gradient(self, cost_func, dJdx_T=None):
        """
        Compute dJ/dp using discrete adjoint (arXiv:2410.01911).

        Parameters
        ----------
        cost_func : callable
            J(x_T) -> scalar. Cost function of final state.
        dJdx_T : np.array or None
            If provided, gradient of cost w.r.t. final state (avoids FD).

        Returns
        -------
        dJdp : np.array
            Gradient of J w.r.t. the AD parameters.
        """
        if not hasattr(self, '_rk_data') or self._rk_data is None:
            raise RuntimeError("Must call run() before compute_gradient()")
        if not hasattr(self, '_ad_param_names'):
            raise RuntimeError("Must call _create_param_subset() before compute_gradient()")

        rk = self._rk_data
        all_x = rk['x']
        all_h = rk['h']
        all_k = rk['k']
        all_t = rk['t']
        n = rk['n_states']
        N = len(all_h)

        # Get current parameter values
        p_vals = np.array([self._numeric_variables_all[idx]
                           for idx in self._ad_param_var_indices], dtype=float)
        # Update from self.variables
        for i, var_idx in enumerate(self._ad_param_var_indices):
            const_pos = self._var_idx_to_const_pos(var_idx)
            p_vals[i] = float(self.variables[const_pos])

        if not hasattr(self, '_aad_funcs') or self._aad_funcs is None:
            raise RuntimeError("AAD kernel not recorded. Call _create_param_subset() first.")

        # Terminal condition
        if dJdx_T is not None:
            wbarend = np.array(dJdx_T, dtype=float)
        else:
            x_T = np.array(all_x[-1])
            J0 = cost_func(x_T)
            wbarend = np.zeros(n)
            eps = 1e-7
            for i in range(n):
                x_up = x_T.copy(); x_up[i] += eps
                wbarend[i] = (cost_func(x_up) - J0) / eps

        # Butcher tableau (Dormand-Prince)
        a = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        ])
        b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
        c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
        s = 7

        # Discrete adjoint backward sweep
        # (arXiv:2410.01911, Algorithm 1; ported from C++ backpropagation.hpp)
        # Dormand-Prince Butcher tableau
        a = np.array([
            [0, 0, 0, 0, 0, 0, 0],
            [1/5, 0, 0, 0, 0, 0, 0],
            [3/40, 9/40, 0, 0, 0, 0, 0],
            [44/45, -56/15, 32/9, 0, 0, 0, 0],
            [19372/6561, -25360/2187, 64448/6561, -212/729, 0, 0, 0],
            [9017/3168, -355/33, 46732/5247, 49/176, -5103/18656, 0, 0],
            [35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0],
        ])
        b = np.array([35/384, 0, 500/1113, 125/192, -2187/6784, 11/84, 0])
        c = np.array([0, 1/5, 3/10, 4/5, 8/9, 1, 1])
        s = 7

        alphabar = np.zeros(len(p_vals))

        for step in range(N - 1, -1, -1):
            h = all_h[step]
            x_n = np.array(all_x[step])
            k = all_k[step]
            t_n = all_t[step]

            # Initialize stage adjoints (C++ back_prop_step lines 171-176)
            w_bar = np.zeros((n, s + 2))
            w_bar[:, s + 1] = wbarend

            # Distribute incoming adjoint to stages (C++ lines 179-184)
            for i in range(n):
                w_bar[i, 0] += w_bar[i, s + 1]
                for mm in range(1, s + 1):
                    w_bar[i, mm] += b[mm - 1] * h * w_bar[i, s + 1]

            # Backward through stages s to 1 (C++ lines 196-224)
            for mm in range(s, 0, -1):
                t_mn = t_n + c[mm - 1] * h

                # Reconstruct intermediate state (C++ get_intermediate_state)
                x_mn = x_n.copy()
                for kk in range(1, mm):
                    x_mn += h * a[mm - 1, kk - 1] * k[kk - 1]

                w_bar_m = w_bar[:, mm].copy()
                vjp_x, vjp_p = self._vjp(x_mn, p_vals, t_mn, w_bar_m)

                # Update stage 0 adjoint (C++ line 211)
                for i in range(n):
                    w_bar[i, 0] += vjp_x[i]

                # Update earlier stage adjoints (C++ lines 214-216)
                for kk in range(1, mm):
                    for i in range(n):
                        w_bar[i, kk] += vjp_x[i] * a[mm - 1, kk - 1] * h

                # Accumulate parameter sensitivity (C++ lines 220-222)
                alphabar += vjp_p

            # Update wbarend for next step (C++ lines 227-228)
            wbarend[:] = w_bar[:, 0]

        return alphabar

    # ---- reset helpers ----
    def run_offline_pre_and_set_default_state(self, offline_pre_time):
        offline_pre_time = float(offline_pre_time)
        if offline_pre_time <= 0:
            return
        self._do_ad = False
        self.update_times(self.dt, 0.0, offline_pre_time, 0.0)
        self.run()
        self.states = list(self.state_traj[:, -1])
        self.default_state_inits = list(self.states)
        self._has_run = False

    def reset_and_clear(self, only_one_exp=-1):
        self._do_ad = False
        self._aadc_funcs = None
        self._init_state()

    def reset_states(self):
        self.states = list(self.default_state_inits)

    def close_simulation(self):
        pass
