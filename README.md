# AADC for CellML ODE Model Calibration

AADC as a drop-in replacement for CasADI for differentiating physiological ODE models.
CasADI crashes on models with conditional logic (valve open/close). AADC handles it.

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

Set `AADC` path in `Makefile`, then:

```bash
cd exp/CellML
make
make bench          # runs all: AADC C++ + CasADI Python baseline
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

| File | What it does |
|---|---|
| `compare_casadi_vs_aadc.py` | **Start here:** side-by-side CasADI vs AADC comparison |
| `cvs3_aadc_python.py` | Python: 3-compartment model + gradient benchmark |
| `example_usage.py` | Python: gradient, Hessian, batch evaluation |
| `example_hmc.py` | Python: HMC posterior sampling |
| `cvs3_aadc.cpp` | C++: same model, multi-thread, Hessian |
| `lotka_bench.cpp` | C++: Lotka-Volterra benchmark (1,323× vs CasADI) |
| `run_casadi_bench.py` | CasADI baseline (crashes on 3-compartment) |
| `Makefile` | `make` / `make bench` |

---

## Appendix: Integrating AADC with your own CellML model

### Step 1: Modify generated Python code

The CellML code generator produces `compute_rates()` with `if/else` and `max()`.
Three replacements needed:

| Pattern | Replace with | Example |
|---|---|---|
| `x if leq_func(a, b) else y` | `aadc.iif(a <= b, x, y)` | valve logic |
| `max(x, 0.0)` | `aadc.iif(x >= 0.0, x, 0.0)` | clamping |
| `floor(x)` | `math.floor(float(x))` | cardiac phase |

### Step 2: Write ODE stepper

For stiff models, use semi-implicit Euler with diagonal damping:

```python
for step in range(total_steps):
    rates, lam = compute_rates_and_damping(st, params)
    for i in range(N_STATES):
        st[i] += dt * rates[i] / (1.0 + dt * lam[i])
```

`lam[i] = -∂f_i/∂y_i` — the diagonal Jacobian coefficient.
See `cvs3_aadc_python.py` for complete implementation.

### Step 3: Record kernel + evaluate

```python
import aadc

# Record (once, ~3s)
funcs = aadc.Functions()
funcs.start_recording()
param = aadc.idouble(value)
a_param = param.mark_as_input()
cost = run_model(param)
r_cost = cost.mark_as_output()
funcs.stop_recording()

# Evaluate (fast, ~6ms)
res = aadc.evaluate(funcs, {r_cost: [a_param]}, {a_param: value}, aadc.ThreadPool(4))
gradient = float(np.asarray(res[1][r_cost][a_param]).flat[0])
```

See `example_usage.py` for Hessian and batch evaluation.
See `example_hmc.py` for HMC sampling.
