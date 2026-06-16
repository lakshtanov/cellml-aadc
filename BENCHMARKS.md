# Benchmark Results

## CasADI vs AADC

The 3-compartment cardiovascular model (27 states) contains conditional valve logic
(heart valves open/close based on pressure differences using `if/else`).

**Why CasADI crashes:** CasADI builds a symbolic expression graph (SX). When
the model evaluates `chi * 2.0 if leq_func(chi, 0.5) else 0.0`, Python calls
`SX.__bool__()` to decide which branch to take. But `chi` is a symbolic variable
with no concrete value — `__bool__()` is undefined, so CasADI raises
`RuntimeError: Cannot compute the truth value of a CasADi SXElem`.
This is a fundamental limitation: CasADI cannot trace through `if/else` at all.

**Why AADC works:** AADC uses a tape-based approach. During recording, `idouble`
variables carry concrete values, so `if/else` executes normally (the condition
has a definite truth value). For parametric replay where the branch might change,
`aadc.iif(condition, val_true, val_false)` records both branches on the tape
and selects the correct one at evaluation time. The gradient through `iif` is
the gradient of whichever branch was taken.

```python
# CasADI: CRASHES
chi_final = chi * 2.0 if leq_func(chi, 0.5) else 0.0

# AADC: WORKS
chi_final = aadc.iif(chi <= 0.5, chi * 2.0, 0.0)
```

| Metric | CasADI | AADC |
|---|---|---|
| LV gradient (2 states, 4 params) | 0.2 ms | 0.19 ms |
| **CVS gradient (27 states, 2 params)** | **CRASHES** | **6.2 ms** |
| CVS batch 100 evals (4 threads) | CRASHES | 0.48 ms/eval |
| CVS Hessian 4×4 | CRASHES | 84 ms |
| CVS HMC 1000 samples | IMPOSSIBLE | ~2.5 s |

On Lotka-Volterra (no conditionals), CasADI and AADC perform identically (~0.2 ms).

**Note:** CasADI does not crash because of model size (27 states vs 2).
It crashes because of `if/else` in the code. A 1000-state model without
conditionals would work fine in CasADI. A 2-state model with one `if`
would crash. Most realistic physiological models (valve logic, activation
thresholds, clamping) contain conditionals — so in practice CasADI fails
on most CellML cardiovascular models.

## AADC Tape-Based Gradient Scaling

3-compartment cardiovascular model (27 states, 2 calibration parameters).
Tape records the full RK4 integration, one reverse pass gives all gradients.

| Steps | Record (one-time) | Single gradient | Batch 100 (4 threads) |
|---|---|---|---|
| 220 | 0.5 s | 0.6 ms | 0.06 ms/eval |
| **2,200** (default) | **3.1 s** | **6.2 ms** | **0.48 ms/eval** |
| 5,500 | 7.2 s | 30.5 ms | 1.18 ms/eval |
| 11,000 | 14.0 s | 54.3 ms | 2.48 ms/eval |
| 22,000 | 27.7 s | 104.9 ms | 5.91 ms/eval |

Scaling is linear in the number of steps. Record time is one-time cost.
Batch evaluation amortizes Python overhead via SIMD (AVX2, 4 lanes)
and multi-threading.

## Gradient Methods

`compute_gradient()` auto-selects the best method:

| Method | How it works | Speed | Memory | When to use |
|---|---|---|---|---|
| **tape** (default) | Record full ODE on AADC tape, one reverse pass | **0.5 ms** | O(steps) | < 50K steps |
| **adjoint** | Discrete adjoint per RK stage ([arXiv:2410.01911](https://arxiv.org/abs/2410.01911)) | 65 ms | O(1 step) | > 50K steps |

Tape uses RK4 fixed-step (recordable on AADC tape).
Adjoint uses adaptive RK45 Dormand-Prince (not recordable, needs per-stage VJP).
Both produce correct gradients for their respective discretizations.

The tape method is **136× faster** than discrete adjoint for typical models.

## HMC Performance

Hamiltonian Monte Carlo for Bayesian posterior sampling of calibration parameters.

| Configuration | Time | Samples/s |
|---|---|---|
| Sequential (1 thread) | 33 s / 1000 samples | 30 |
| Batch 100 (4 threads) | **2.5 s / 1000 samples** | **400** |

With CasADI: **impossible** (gradient crashes on conditional models).

## Discrete Adjoint (arXiv:2410.01911)

Python implementation of Martins & Lakshtanov (2025). Used as fallback
for very long integrations where tape memory is prohibitive.

| Model | Steps | Forward | Adjoint | Ratio |
|---|---|---|---|---|
| Lotka-Volterra (2 states) | 188 (adaptive) | 37 ms | 65 ms | 1.8× |

Adjoint/forward ratio 1.8× is close to the theoretical optimum (2-3×).
The per-stage VJP uses a single AADC reverse pass (optimized: `v^T f` scalar trick).

## Parallelization Scaling

3-compartment CVS, 27 states, 2200 steps, 2 calibration parameters.
Tape-based gradient with varying thread count and batch size.

| Threads | Batch 1 | Batch 10 | Batch 100 | Peak evals/s |
|---|---|---|---|---|
| 1 | 6.9 ms | 1.8 ms/eval | 1.4 ms/eval | 732 |
| 2 | 12.0 ms | 1.5 ms/eval | 0.80 ms/eval | 1,257 |
| 4 | 11.5 ms | 1.0 ms/eval | 0.49 ms/eval | 2,050 |
| **8** | 10.1 ms | **0.97 ms/eval** | **0.30 ms/eval** | **3,324** |

Single evaluation has overhead (~10ms) from Python → C++ call.
Batching amortizes this: 100 evaluations cost 30ms total at 8 threads.
SIMD (AVX2, 4 lanes) provides additional parallelism within each thread.

## Application Benchmarks

Using batch evaluation (tape-based, 4 threads):

| Application | Evaluations | Time | Per eval |
|---|---|---|---|
| Hessian (2×2) | 4 | **9 ms** | 2.3 ms |
| HMC (1000 samples, 5 leapfrog) | 5,000 | **2.4 s** | 0.48 ms |
| Ensemble calibration (1000 patients) | 1,000 | **396 ms** | 0.40 ms |

With CasADI: all three are **impossible** (gradient crashes on conditional models).

## Validation

All gradients verified against central finite differences:

| Test | AD/FD ratio |
|---|---|
| LV: AADC vs CasADI | gradients identical (diff < 1e-10) |
| LV: AADC vs FD | 1.000000 |
| CVS: AADC tape vs FD | 1.000000 |
| CVS: Discrete adjoint vs FD | 1.000000 |
| CVS: Hessian symmetry | < 1% |

## Why Runtime AD Matters for Multi-Organ Models

The 12 LABOURS project aims to build patient-specific digital twins
combining multiple organ models (heart, lungs, kidneys, vasculature...)
each written in CellML, each potentially with conditional logic.

**Compile-time AD (CasADI approach):**
- Must assemble the full symbolic graph of all coupled organs before computing
- Every new patient configuration (which organs, which connections) = rebuild the graph
- Any `if/else` in any sub-model breaks the entire symbolic graph
- Adding or removing an organ requires re-deriving the adjoint

**Runtime AD (AADC approach):**
- Run the coupled model → tape records automatically → gradient ready
- Change patient configuration → re-run → new tape, no recompilation
- Conditionals in any sub-model handled transparently via `iif()`
- Add/remove organs at runtime with no AD infrastructure changes

For clinical use (calibrate model to this patient's data, now), runtime AD
eliminates the compile step between "change the model" and "get the gradient".
This is the difference between a research tool and a clinical tool.

## Hardware

All benchmarks on Intel Xeon Silver 4114 (10 cores), single socket.
Python 3.11, AADC trial license.
