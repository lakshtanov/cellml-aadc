# AADC for CellML ODE Model Calibration

Benchmark and examples for using [AADC](https://matlogica.com) (Automatic Adjoint Differentiation Compiler)
as an AD backend for physiological ODE models from the
[circulatory_autogen](https://github.com/physiomelinks/circulatory_autogen) project
(University of Auckland).

**The problem:** CasADI (their current AD backend) uses symbolic expression graphs.
When the model contains `if/else` (e.g. heart valve opening/closing logic),
CasADI calls `SX.__bool__()` on a symbolic variable and crashes.

**The solution:** AADC records the actual execution and handles conditionals
via `aadc.iif()`, which evaluates both branches and selects at replay time.
Gradients are exact, Hessian and HMC sampling work out of the box.

## Prerequisites

1. **AADC Python package** — contact [MatLogica](https://matlogica.com) for trial license
2. **Python 3.11+** with `numpy`
3. **AADC C++ library** (optional, for C++ benchmarks)
4. **g++ with AVX2** (optional, for C++ benchmarks)

## Install AADC Python

```bash
pip install aadc    # or follow MatLogica install instructions
```

Verify:
```bash
python -c "import aadc; print('OK')"
```

## Run Python examples

### 1. `compare_casadi_vs_aadc.py` — start here

Runs the same two models (Lotka-Volterra and 3-compartment cardiovascular)
with both CasADI and AADC, side by side. Shows that CasADI produces
identical gradients on Lotka-Volterra but crashes on 3-compartment
due to conditional valve logic. AADC handles both.

```bash
python compare_casadi_vs_aadc.py
```

### 2. `cvs3_aadc_python.py` — gradient benchmark

Records the full 27-state cardiovascular model (2,200 time steps,
semi-implicit Euler, all valve conditionals via `aadc.iif()`) onto
the AADC tape and benchmarks gradient evaluation. Recording takes ~3s
(one-time), then each gradient costs ~6ms.

```bash
python cvs3_aadc_python.py
```

### 3. `example_usage.py` — gradient, Hessian, batch

Shows the practical workflow in 4 steps: record kernel, evaluate
single gradient, compute the full 4×4 Hessian via FD of gradient
(84ms), and batch-evaluate 100 parameter sets at once (0.5ms/eval).
This is what you need for calibration and uncertainty quantification.

```bash
python example_usage.py
```

### 4. `example_hmc.py` — Hamiltonian Monte Carlo

Runs full Bayesian posterior sampling for 2 calibration parameters
(q_lv_init, C_aortic) using HMC with leapfrog integrator. Starts
from a perturbed point, recovers true parameters within 1σ.
200 samples in 7.7s, 96% acceptance rate.

```bash
python example_hmc.py
```

### Expected output: `compare_casadi_vs_aadc.py`

```
MODEL 1: Lotka-Volterra (2 states, 4 params)
  CasADI:  Cost=3.319408e-01  Gradient=[0.859, 0.240, -0.465, 2.503]  0.2 ms
  AADC:    Cost=3.319408e-01  Gradient=[0.859, 0.240, -0.465, 2.503]  0.3 ms
  Cost match: True.  Max gradient diff: 5.00e-16

MODEL 2: 3-Compartment Cardiovascular (27 states, conditional valve logic)
  CasADI:  *** CRASHES: Cannot compute the truth value of a CasADi SXElem
  AADC:    Cost=1.357e-07  Gradient: 4 values  6.2 ms/eval  WORKS
```

### Expected output: `cvs3_aadc_python.py`

```
Recording AADC kernel from Python...
Recording: 3.0s

Cost = 1.357312e-07
Gradient:
  dC/dq_lv = 7.325680e-06
  dC/dC_ao = -6.201263e-01
  dC/dE_lv_A = -3.061596e-16
  dC/dE_lv_B = -4.325641e-15

Benchmark: 6.52 ms/eval (gradient included)
           153 evals/s
```

### Expected output: `example_usage.py`

```
Step 1: Recording AADC kernel...
  Done in 3.1s (one-time cost)

Step 2: Gradient evaluation
  Cost = 1.357312e-07
  Gradient:
    dC/dq_lv_init = 7.325680e-06
    dC/dC_aortic = -6.201263e-01
  Time: 10.0 ms/eval

Step 3: Hessian via FD of gradient
  Hessian (4x4) computed in 84 ms

Step 4: Batch gradient evaluation (for HMC)
  100 gradient evaluations in 52 ms (0.5 ms/eval)
```

### Expected output: `example_hmc.py`

```
HMC Results (200 samples in 7.7s)
Acceptance rate: 96%
Samples/second: 26.0

Posterior statistics:
  q_lv_init: mean=1.90e-03 ± 3.07e-04  (true=2.00e-03)
  C_aortic:  mean=1.07e-08 ± 2.43e-09  (true=1.20e-08)
```

## Build and run C++ benchmarks (optional)

The C++ version of the same 3-compartment model runs ~2x faster than Python
(3.3ms vs 6.5ms per gradient) and supports multi-thread + AVX vectorization
for throughput up to 5,000 gradient evaluations per second.

Set `AADC` path in `Makefile` to your AADC C++ installation, then:

```bash
make                # build lotka_bench and cvs3_aadc
make bench          # run all C++ benchmarks
./cvs3_aadc --threads 8 --iters 50   # 3-compartment only
```

### Expected output: `cvs3_aadc`

```
Forward (1 thr, 1 lane):  1.231 ms
AD (1 thr, 1 lane):       3.325 ms  ratio: 2.7x fwd
AD (8 thr × 4 AVX = 32):  0.200 ms/eval  (4998 evals/s)

Gradient: dC/dq_lv_init=7.325680e-06, dC/dC_aortic=-6.201263e-01, ...

FD gradient verification:
  dC/dq_lv_init: AD/FD ratio=1.000000
  dC/dC_aortic:  AD/FD ratio=0.999996

Hessian (4x4) computed in 26.55 ms
  Symmetry: max |H[i,j]-H[j,i]|/avg = 1.73e-04
```

## Files

**Python (main examples):**

| File | What it does |
|---|---|
| `compare_casadi_vs_aadc.py` | **Start here.** Runs both CasADI and AADC on both models side by side. Shows CasADI crash vs AADC success. |
| `cvs3_aadc_python.py` | Full 3-compartment cardiovascular model (27 states, 4 calibration params) ported to AADC. All valve conditionals use `aadc.iif()`. Semi-implicit Euler for stiff ODE. |
| `example_usage.py` | Practical 4-step workflow: record kernel → gradient → Hessian → batch. Copy this as a starting template for your own model. |
| `example_hmc.py` | Hamiltonian Monte Carlo: Bayesian posterior sampling for 2 parameters with leapfrog integrator and Metropolis accept/reject. |

**C++ (optional, for maximum performance):**

| File | What it does |
|---|---|
| `cvs3_aadc.cpp` | Same 3-compartment model in C++ with AADC. Multi-thread, AVX, Hessian. 0.2 ms/eval at 8 threads. |
| `lotka_bench.cpp` | Lotka-Volterra (2 states) C++ benchmark. 1,323× faster than CasADI AD. |

**Baselines:**

| File | What it does |
|---|---|
| `run_casadi_bench.py` | CasADI baseline for Lotka-Volterra. Crashes on 3-compartment. |
| `Makefile` | `make` builds C++. `make bench` runs all C++ benchmarks. |

---

## Integration with circulatory_autogen

`circulatory_autogen` has a clean backend architecture — all solvers go through
a common `SimulationHelper` interface in `src/solver_wrappers/`. This means AADC
can be plugged in as a peer to CasADI/Myokit/OpenCOR and propagate to the entire
pipeline: paramID, HMC, sensitivity analysis, and the 12 LABOURS platform.

### Architecture

```
paramID / HMC / sensitivity_analysis
        ↓
  get_simulation_helper(solver="aadc_semi_implicit")
        ↓
  ┌──────────────────────────────────────────┐
  │ SimulationHelper  (common interface)     │
  │   .run()  .get_results()  .set_param_vals│
  └────┬────────┬────────┬────────┬──────────┘
       │        │        │        │
   OpenCOR   Myokit    CasADI   AADC ← new
   (CVODE)   (CVODE)  (cvodes)  (semi-implicit + iif)
```

### Installation (3 steps)

**Step 1.** Copy `aadc_solver_helper.py` into circulatory_autogen:
```bash
cp aadc_solver_helper.py  <circulatory_autogen>/src/solver_wrappers/
```

**Step 2.** Register the backend in `src/solver_wrappers/__init__.py`:
```python
try:
    from solver_wrappers.aadc_solver_helper import SimulationHelper as AadcSimulationHelper
except Exception:
    AadcSimulationHelper = None
```

**Step 3.** Add to `get_simulation_helper()` in the same file:
```python
elif solver == 'aadc_semi_implicit':
    if AadcSimulationHelper is not None:
        return AadcSimulationHelper(model_path, dt, sim_time, solver_info, pre_time=pre_time)
    else:
        raise RuntimeError("AADC solver requested but aadc is not installed")
```

### Usage

In your experiment config, set:
```yaml
solver:
  solver: aadc_semi_implicit
```

That's it. Any CellML model processed by circulatory_autogen now gets
AADC gradients — including models with conditionals where CasADI crashes.

### What aadc_solver_helper.py handles automatically

- **Conditionals**: patches `leq_func`, `geq_func`, `max` in the generated
  model to use `aadc.iif()` — no manual code changes needed
- **Math functions**: patches `exp`, `sin`, `cos`, `floor` etc. to AADC versions
- **ODE integration**: semi-implicit Euler (stiff-safe)
- **Same interface**: `run()`, `get_results()`, `set_param_vals()`, `get_init_param_vals()`,
  `reset_states()` — all match the existing backends

## Appendix: Standalone usage (without circulatory_autogen)

The Python examples in this repo work standalone — you don't need
circulatory_autogen installed. They demonstrate the AADC workflow
directly:

| Step | How | Where to look |
|---|---|---|
| Record kernel | `aadc.Functions()` + `start/stop_recording` | `example_usage.py` Step 1 |
| Get gradient | `aadc.evaluate()` | `example_usage.py` Step 2 |
| Hessian | FD of gradient | `example_usage.py` Step 3 |
| HMC | Leapfrog + Metropolis | `example_hmc.py` |
| Conditionals | `aadc.iif(cond, val_true, val_false)` | `cvs3_aadc_python.py` |
| Stiff ODE | `y += dt*f/(1+dt*lam)` | `cvs3_aadc_python.py` |
