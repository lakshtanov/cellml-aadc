# Gradient Architecture: Auto-Selection

## Three Gradient Methods

### 1. Tape (full ODE on AADC tape)

**How**: Record entire ODE integration (RK4 fixed-step) on AADC tape.
One forward + one reverse = all gradients.

**Speed**: 0.008 ms/eval (119K evals/s with AVX + threads)

**Accuracy**: RK4 fixed-step. Not stiff-safe. First/fourth order.

**Memory**: O(N_states × total_steps). ~4 KB/step for 27 states.

**When to use**:
- Model is not stiff (explicit RK4 stable at given dt)
- Total steps < 50K (tape fits in memory, < 200 MB)
- Calibration, HMC, sensitivity analysis (gradient direction is what matters)
- Maximum throughput needed (batch evaluation)

### 2. ComputeBlock(CVODES)

**How**: CVODES (BDF, implicit, adaptive) as AADC ComputeBlock.
Forward: CVODES integrates, AADC provides Jacobian.
Reverse: CVODES adjoint (CVodeB) integrates backward, AADC provides VJP.

**Speed**: ~340 ms (forward 168 ms + reverse ~168 ms)

**Accuracy**: BDF adaptive, stiff-safe. Same accuracy as CVODES standalone.

**Memory**: O(checkpoints) — CVODES internal checkpointing, not full tape.

**When to use**:
- Model is stiff (stiffness ratio > 1e6)
- Accuracy is critical (uncertainty quantification, clinical decisions)
- Long integrations (tape would be too large)

### 3. Discrete Adjoint (rk-adjoint-python)

**How**: Adaptive RK45 forward, store trajectory + stages.
Backward: walk through stages, AADC VJP per stage.

**Speed**: ~65 ms (Python), ~1 ms (C++ VectorizedAdjoint)

**Accuracy**: Adaptive RK45, not stiff-safe but handles moderate stiffness.

**Memory**: O(N_steps) for trajectory storage.

**When to use**:
- Python-only environment (no C++ CVODES)
- Moderate stiffness (RK45 converges)
- Fallback when tape too large but CVODES not available

## Auto-Selection Logic

```
compute_gradient(method='auto'):

  1. Estimate stiffness (if not cached):
     - Record RHS with AADC
     - Evaluate Jacobian at initial state
     - Compute eigenvalues
     - stiffness_ratio = |λ_min / λ_max|

  2. Check memory:
     - tape_size = N_states × total_steps × 4 KB
     - tape_fits = (tape_size < 200 MB)

  3. Select method:
     if stiffness_ratio > 1e6:
         method = 'cvodes'        # stiff → need implicit solver
     elif tape_fits:
         method = 'tape'          # not stiff, fits → fastest
     else:
         method = 'adjoint'       # not stiff, too large → per-stage
```

## Decision Table

| Stiff? | Tape fits? | Method | Speed | Accuracy |
|---|---|---|---|---|
| No | Yes | **tape** | 0.008 ms | RK4 |
| No | No | **adjoint** | 65 ms (Python) | adaptive RK45 |
| Yes | — | **cvodes** | ~340 ms | BDF adaptive |

## User Override

```python
# Auto (default)
h.compute_gradient(cost_func, method='auto')

# Force specific method
h.compute_gradient(cost_func, method='tape')     # fastest
h.compute_gradient(cost_func, method='cvodes')   # most accurate
h.compute_gradient(cost_func, method='adjoint')  # adaptive RK45
```

## Config via solver_info

```yaml
solver: aadc_semi_implicit
solver_info:
  gradient_method: auto          # auto | tape | cvodes | adjoint
  stiffness_threshold: 1e6      # stiffness ratio above this → cvodes
  max_tape_mb: 200               # tape size limit for auto
  threads: 4                     # multi-thread for batch evaluation
  tol: 1e-8                      # ODE tolerance (adaptive methods)
```

## Implementation Status

| Method | Forward | Gradient | Status |
|---|---|---|---|
| tape | RK4 on tape | tape reverse | ✅ Done |
| adjoint | adaptive RK45 | per-stage VJP | ✅ Done |
| cvodes | CVODES BDF | forward Jacobian only | ⚠️ Partial |
| cvodes | CVODES BDF | **ComputeBlock adjoint** | ❌ TODO |

### What's needed for ComputeBlock(CVODES)

1. AADC ComputeBlock wrapping CVODES forward integration
   - Input: parameters p
   - Output: final state x(T) or cost J
   - Inside: CVODES BDF with AADC Jacobian (already working)

2. ComputeBlock adjoint callback
   - CVODES adjoint (CVodeAdjInit, CVodeB) integrates backward
   - At each step: AADC provides v^T ∂f/∂x and v^T ∂f/∂p
   - Output: dJ/dp

3. Integration with compute_gradient() API
   - method='cvodes' triggers ComputeBlock path
   - method='auto' selects based on stiffness estimate

### Estimated CVODES ComputeBlock performance

- Forward: 168 ms (CVODES BDF, already measured)
- Reverse: ~168 ms (CVODES adjoint, similar cost to forward)
- Total gradient: ~340 ms
- Batch: parallel independent integrations, 24 ms/eval at 8 threads
- HMC 1000 samples: ~340s sequential, ~42s at 8 threads
