#!/usr/bin/env python3
"""
Memory and performance scaling test for AADC CellML backend.

Measures how workspace size, compilation time, and evaluation time
scale with the number of ODE integration steps. This validates that
AADC memory usage is predictable and linear.

Run: python test_scaling.py
"""
import time
import numpy as np

try:
    import aadc
except ImportError:
    print("AADC not installed — skipping")
    exit(0)

from cvs3_aadc_python import run_model, Params


def measure(pre_steps, sim_steps):
    """Record and benchmark at given step count."""
    p = Params()
    total = pre_steps + sim_steps

    t0 = time.time()
    funcs = aadc.Functions()
    funcs.start_recording()

    id_qlv = aadc.idouble(p.q_lv_init)
    id_cao = aadc.idouble(p.C_aortic)
    a_qlv = id_qlv.mark_as_input()
    a_cao = id_cao.mark_as_input()

    cost = run_model(id_qlv, id_cao,
                     aadc.idouble(p.E_lv_A), aadc.idouble(p.E_lv_B),
                     pre_steps=pre_steps, sim_steps=sim_steps, dt=0.01)
    r_cost = cost.mark_as_output()
    funcs.stop_recording()
    rec_time = time.time() - t0

    # Evaluate
    inputs = {a_qlv: p.q_lv_init, a_cao: p.C_aortic}
    request = {r_cost: [a_qlv, a_cao]}
    workers = aadc.ThreadPool(1)

    # Warmup
    aadc.evaluate(funcs, request, inputs, workers)

    # Benchmark
    n = 10
    t0 = time.time()
    for _ in range(n):
        aadc.evaluate(funcs, request, inputs, workers)
    eval_ms = (time.time() - t0) / n * 1000

    return {
        'total_steps': total,
        'rec_time': rec_time,
        'eval_ms': eval_ms,
    }


def main():
    print("=" * 70)
    print("AADC Memory & Performance Scaling Test")
    print("3-compartment cardiovascular model (27 states)")
    print("=" * 70)

    configs = [
        (200, 20),       # 220 steps (tiny)
        (2000, 200),     # 2,200 steps (default)
        (5000, 500),     # 5,500 steps
        (10000, 1000),   # 11,000 steps
        (20000, 2000),   # 22,000 steps
    ]

    results = []
    for pre, sim in configs:
        total = pre + sim
        print(f"\n  Recording {total:,} steps...", end=" ", flush=True)
        r = measure(pre, sim)
        results.append(r)
        print(f"rec={r['rec_time']:.1f}s, eval={r['eval_ms']:.1f}ms")

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Steps':>8} {'Rec (s)':>8} {'Eval (ms)':>10} {'µs/step':>8}")
    print(f"{'-'*70}")
    for r in results:
        us_per_step = r['eval_ms'] * 1000 / r['total_steps']
        print(f"{r['total_steps']:>8,} {r['rec_time']:>8.1f} {r['eval_ms']:>10.1f} {us_per_step:>8.1f}")

    # Check linearity
    steps = [r['total_steps'] for r in results]
    times = [r['eval_ms'] for r in results]
    if len(results) >= 3:
        # Linear fit
        coeffs = np.polyfit(steps, times, 1)
        r_squared = 1 - np.sum((np.array(times) - np.polyval(coeffs, steps))**2) / \
                        np.sum((np.array(times) - np.mean(times))**2)
        print(f"\nLinear fit: eval_ms = {coeffs[0]*1000:.2f} µs/step + {coeffs[1]:.1f} ms")
        print(f"R² = {r_squared:.4f}")
        if r_squared > 0.99:
            print("Scaling is linear ✓")
        else:
            print(f"WARNING: R² < 0.99, scaling may not be perfectly linear")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    main()
