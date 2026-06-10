"""CasADI Lotka-Volterra benchmark — baseline for AADC comparison.
Run: python run_casadi_bench.py [--steps N] [--iters N]
"""
import casadi as ca
import numpy as np
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=500, help='ODE time steps (T = steps * dt)')
parser.add_argument('--iters', type=int, default=50, help='benchmark iterations')
parser.add_argument('--dt', type=float, default=0.01)
parser.add_argument('--alpha', type=float, default=4.5, help='test alpha')
parser.add_argument('--beta', type=float, default=0.25, help='test beta')
parser.add_argument('--delta', type=float, default=0.18, help='test delta')
parser.add_argument('--gamma', type=float, default=3.2, help='test gamma')
args = parser.parse_args()

N = args.steps
dt = args.dt
test_params = np.array([args.alpha, args.beta, args.delta, args.gamma])
true_params = np.array([5.0, 0.2, 0.2, 3.0])
x0_val = np.array([20.0, 10.0])

print(f"CasADI Lotka-Volterra Benchmark")
print(f"  Steps: {N}, dt: {dt}, T: {N*dt}")
print(f"  True params:  {true_params}")
print(f"  Test params:  {test_params}")
print(f"  Bench iters:  {args.iters}")

# Setup
x = ca.SX.sym('x', 2)
p = ca.SX.sym('p', 4)
rhs = ca.vertcat(p[0]*x[0] - p[1]*x[0]*x[1], p[2]*x[0]*x[1] - p[3]*x[1])
dae = {'x': x, 'p': p, 'ode': rhs}
integrator = ca.integrator('F', 'cvodes', dae, 0, dt)

# Observed data
def simulate_np(params, x0=x0_val, N_steps=N):
    traj = np.zeros((N_steps+1, 2)); traj[0] = x0
    xk = ca.DM(x0)
    for k in range(N_steps):
        xk = integrator(x0=xk, p=params)['xf']
        traj[k+1] = np.array(xk).flatten()
    return traj

obs_data = simulate_np(true_params)

def cost_numpy(params):
    traj = simulate_np(params)
    return float(np.mean((traj - obs_data)**2))

# Build CasADI symbolic cost + gradient
print("\nBuilding symbolic graph...")
t0 = time.time()
p_sym = ca.SX.sym('p', 4)
x_sym = ca.SX.sym('x0', 2)
xk = x_sym; total_cost = 0
for k in range(N):
    xk = integrator(x0=xk, p=p_sym)['xf']
    diff = xk - ca.DM(obs_data[k+1])
    total_cost += ca.dot(diff, diff)
total_cost = total_cost / (N * 2)

cost_fn = ca.Function('cost_f', [p_sym, x_sym], [total_cost])
grad_fn = ca.Function('grad_f', [p_sym, x_sym], [ca.gradient(total_cost, p_sym)])
print(f"Build time: {time.time()-t0:.2f}s")

c = float(cost_fn(test_params, x0_val))
print(f"Cost at test params: {c:.4f}")

# Benchmark forward
NB = args.iters
t0 = time.time()
for _ in range(NB):
    _ = float(cost_fn(test_params, x0_val))
t1 = time.time()
fwd_ms = (t1-t0)/NB*1000

# Benchmark AD
t0 = time.time()
for _ in range(NB):
    g_ad = np.array(grad_fn(test_params, x0_val)).flatten()
t1 = time.time()
ad_ms = (t1-t0)/NB*1000

# Benchmark FD
t0 = time.time()
for _ in range(NB):
    g_fd = np.zeros(4); f0 = cost_numpy(test_params); eps = 1e-7
    for i in range(4):
        pp = test_params.copy(); pp[i] += eps
        g_fd[i] = (cost_numpy(pp) - f0) / eps
t1 = time.time()
fd_ms = (t1-t0)/NB*1000

print(f"\n=== RESULTS ===")
print(f"Forward:    {fwd_ms:.1f} ms   ({1000/fwd_ms:.0f} evals/s)")
print(f"CasADI AD:  {ad_ms:.1f} ms   ({1000/ad_ms:.0f} evals/s)  ratio: {ad_ms/fwd_ms:.1f}x fwd")
print(f"FD (4p):    {fd_ms:.1f} ms   ({1000/fd_ms:.0f} evals/s)")
print(f"AD speedup vs FD: {fd_ms/ad_ms:.1f}x")
print(f"\nAD gradient:  [{', '.join(f'{v:.4f}' for v in g_ad)}]")
print(f"FD gradient:  [{', '.join(f'{v:.4f}' for v in g_fd)}]")
print(f"Match: {np.allclose(g_ad, g_fd, rtol=0.01)}")
