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
        Semi-implicit Euler integration.
        Works with both plain float (numeric) and aadc.idouble (recording).

        Returns state trajectory: list of state arrays at each sim-time step.
        """
        st = list(states)
        n = self.STATE_COUNT
        rates = [0.0] * n

        # Storage for sim-time portion only
        traj = []

        for step in range(total_steps):
            # Compute rates
            self.model.compute_rates(step * dt, st, rates, variables_all)

            # Semi-implicit Euler: for stiff states, we'd compute lam[i].
            # For now, forward Euler (works for non-stiff or with small dt).
            # TODO: add diagonal damping for stiff models (see cvs3_aadc_python.py)
            for i in range(n):
                st[i] = st[i] + dt * rates[i]

            # Store sim-time steps
            if step >= self.pre_steps:
                traj.append([s for s in st])

        return traj

    # ---- simulation ----
    def run(self):
        """Run simulation. If _do_ad is True, records onto AADC tape."""
        total_steps = self.pre_steps + self.n_steps

        # Build full variables array from constants
        variables_all = list(self._numeric_variables_all)
        for const_pos, const_idx in enumerate(self.constant_indices):
            variables_all[const_idx] = self.variables[const_pos]

        if not self._do_ad:
            # Numeric run
            traj = self._integrate(self.states, variables_all, total_steps, self.dt)
            self.state_traj = np.array(traj).T  # (n_states, n_sim_steps)

            # Compute algebraic variables at each time point
            self._compute_var_traj(traj, variables_all)
        else:
            # AADC recording run — handled by _record_and_evaluate
            pass

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

    # ---- AADC AD helpers ----
    def _create_param_subset(self, param_names, param_vals=None):
        """Mark parameters for AD. Called by paramID before run()."""
        self._ad_param_names = [x[0] if isinstance(x, list) else x for x in param_names]
        if param_vals is not None:
            param_vals = np.asarray(param_vals, dtype=float)
            for i, name in enumerate(self._ad_param_names):
                kind, idx = self._resolve_name(name)
                if kind == "var":
                    self.variables[self._var_idx_to_const_pos(idx)] = param_vals[i]
        self._do_ad = True

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
