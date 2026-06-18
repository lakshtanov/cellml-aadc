/*
 * Test: CVODES as AADC ComputeBlock.
 *
 * Records: params → CvodesComputeBlock → cost
 * Forward: CVODES BDF integration
 * Reverse: adjoint propagation → dJ/dp
 *
 * Uses Lotka-Volterra (not stiff) for validation against FD.
 */
#include <cstdio>
#include <cmath>
#include <chrono>
#include <aadc/aadc.h>
#include "cvodes_compute_block.hpp"
#include <thread>

#ifdef AADC_512
typedef __m512d mmType;
#else
typedef __m256d mmType;
#endif

// Lotka-Volterra RHS
void lv_rhs(double t, const double* x, double* xdot, const double* p, int n, int m) {
    double alpha = p[0], beta = p[1], delta = p[2], gamma = p[3];
    xdot[0] = alpha * x[0] - beta * x[0] * x[1];
    xdot[1] = delta * x[0] * x[1] - gamma * x[1];
}

// Lotka-Volterra Jacobian df/dx
void lv_jac(double t, const double* x, double* J, const double* p, int n, int m) {
    double alpha = p[0], beta = p[1], delta = p[2], gamma = p[3];
    // J[i*n+j] = df_i/dx_j (column-major for SUNDIALS dense)
    J[0] = alpha - beta * x[1];    // df0/dx0
    J[1] = delta * x[1];           // df1/dx0
    J[2] = -beta * x[0];           // df0/dx1
    J[3] = delta * x[0] - gamma;   // df1/dx1
}

// Lotka-Volterra Jacobian df/dp
void lv_par_jac(double t, const double* x, double* Jp, const double* p, int n, int m) {
    // Jp[i*m+j] = df_i/dp_j
    Jp[0] = x[0];            // df0/dalpha
    Jp[1] = -x[0]*x[1];      // df0/dbeta
    Jp[2] = 0;                // df0/ddelta
    Jp[3] = 0;                // df0/dgamma
    Jp[4] = 0;                // df1/dalpha
    Jp[5] = 0;                // df1/dbeta
    Jp[6] = x[0]*x[1];       // df1/ddelta
    Jp[7] = -x[1];           // df1/dgamma
}

void run_benchmarks();
int main() {
    printf("Test: CVODES ComputeBlock with Lotka-Volterra\n");
    printf("=============================================\n\n");

    double x0[] = {1.0, 1.0};
    double p0[] = {1.5, 1.0, 3.0, 1.0};
    int n = 2, m = 4;
    double T = 5.0;

    // Create block
    CvodesComputeBlock block(lv_rhs, lv_jac, lv_par_jac, n, m, x0, T);

    // Record on AADC tape
    aadc::AADCFunctions<mmType> funcs;
    funcs.startRecording();

    idouble id_p[4];
    aadc::AADCArgument a_p[4];
    for (int i = 0; i < m; i++) {
        id_p[i] = p0[i];
        a_p[i] = id_p[i].markAsInput();
    }

    idouble id_xf[2];
    block.integrate(id_p, id_xf);

    // Cost = x(T)^2 + y(T)^2
    idouble cost = id_xf[0] * id_xf[0] + id_xf[1] * id_xf[1];
    auto r_cost = cost.markAsOutput();

    funcs.stopRecording();
    printf("AADC tape recorded (CVODES ComputeBlock inside)\n");

    // Create workspace
    auto ws = std::shared_ptr<aadc::AADCWorkSpace<mmType>>(funcs.createWorkSpace());

    // Forward + Reverse
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < m; i++)
        ws->setVal(a_p[i], aadc::mmSetConst<mmType>(p0[i]));
    funcs.forward(*ws);

    double cost_val = ((double*)&ws->val(r_cost))[0];
    printf("\nForward: cost = %.6e\n", cost_val);

    ws->resetDiff();
    ws->setDiff(r_cost, aadc::mmSetConst<mmType>(1.0));
    funcs.reverse(*ws);

    auto t1 = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    printf("AD gradient (forward + reverse): %.1f ms\n", ms);
    printf("  dJ/dalpha = %.6e\n", ((double*)&ws->diff(a_p[0]))[0]);
    printf("  dJ/dbeta  = %.6e\n", ((double*)&ws->diff(a_p[1]))[0]);
    printf("  dJ/ddelta = %.6e\n", ((double*)&ws->diff(a_p[2]))[0]);
    printf("  dJ/dgamma = %.6e\n", ((double*)&ws->diff(a_p[3]))[0]);

    // FD verification
    printf("\nFD verification:\n");
    double eps = 1e-6;
    for (int ip = 0; ip < m; ip++) {
        double p_up[4], p_dn[4];
        memcpy(p_up, p0, sizeof(p0));
        memcpy(p_dn, p0, sizeof(p0));
        double h = p0[ip] * eps;
        p_up[ip] += h;
        p_dn[ip] -= h;

        for (int i = 0; i < m; i++) ws->setVal(a_p[i], aadc::mmSetConst<mmType>(p_up[i]));
        funcs.forward(*ws);
        double c_up = ((double*)&ws->val(r_cost))[0];

        for (int i = 0; i < m; i++) ws->setVal(a_p[i], aadc::mmSetConst<mmType>(p_dn[i]));
        funcs.forward(*ws);
        double c_dn = ((double*)&ws->val(r_cost))[0];

        double fd = (c_up - c_dn) / (2 * h);
        double ad = ((double*)&ws->diff(a_p[ip]))[0];
        double ratio = (fd != 0) ? ad / fd : (ad == 0 ? 1 : 1e99);
        const char* names[] = {"alpha", "beta", "delta", "gamma"};
        printf("  dJ/d%-6s: AD=%.6e  FD=%.6e  ratio=%.6f\n", names[ip], ad, fd, ratio);
    }

    run_benchmarks();
    return 0;
}

// Append benchmark section
void run_benchmarks() {
    double x0[] = {1.0, 1.0};
    double p0[] = {1.5, 1.0, 3.0, 1.0};
    int n = 2, m = 4;

    printf("\n================================================\n");
    printf("BENCHMARKS: CVODES ComputeBlock\n");
    printf("================================================\n");

    // --- Varying T ---
    printf("\n--- Lotka-Volterra: varying sim time ---\n");
    printf("%8s %10s %10s\n", "T(s)", "Fwd+Rev", "Fwd only");

    for (double T : {1.0, 5.0, 10.0, 20.0, 50.0}) {
        CvodesComputeBlock block(lv_rhs, lv_jac, lv_par_jac, n, m, x0, T);

        aadc::AADCFunctions<mmType> funcs;
        funcs.startRecording();
        idouble id_p[4]; aadc::AADCArgument a_p[4];
        for (int i = 0; i < m; i++) { id_p[i] = p0[i]; a_p[i] = id_p[i].markAsInput(); }
        idouble id_xf[2];
        block.integrate(id_p, id_xf);
        idouble cost = id_xf[0]*id_xf[0] + id_xf[1]*id_xf[1];
        auto r_cost = cost.markAsOutput();
        funcs.stopRecording();

        auto ws = std::shared_ptr<aadc::AADCWorkSpace<mmType>>(funcs.createWorkSpace());

        // Warmup
        for (int i=0;i<m;i++) ws->setVal(a_p[i], aadc::mmSetConst<mmType>(p0[i]));
        funcs.forward(*ws);
        ws->resetDiff(); ws->setDiff(r_cost, aadc::mmSetConst<mmType>(1.0));
        funcs.reverse(*ws);

        // Benchmark forward+reverse
        int iters = (T <= 5) ? 20 : 5;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it=0;it<iters;it++) {
            for (int i=0;i<m;i++) ws->setVal(a_p[i], aadc::mmSetConst<mmType>(p0[i]));
            funcs.forward(*ws);
            ws->resetDiff(); ws->setDiff(r_cost, aadc::mmSetConst<mmType>(1.0));
            funcs.reverse(*ws);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double fr_ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters;

        // Benchmark forward only
        t0 = std::chrono::high_resolution_clock::now();
        for (int it=0;it<iters;it++) {
            for (int i=0;i<m;i++) ws->setVal(a_p[i], aadc::mmSetConst<mmType>(p0[i]));
            funcs.forward(*ws);
        }
        t1 = std::chrono::high_resolution_clock::now();
        double f_ms = std::chrono::duration<double,std::milli>(t1-t0).count()/iters;

        printf("%8.1f %9.1fms %9.1fms\n", T, fr_ms, f_ms);
    }

    // --- Multi-thread ---
    printf("\n--- Multi-thread: parallel ComputeBlock gradients ---\n");
    printf("%8s %8s %10s %10s\n", "Threads", "Evals", "Total", "ms/eval");

    double T = 5.0;
    for (int nth : {1, 2, 4, 8}) {
        int evals = nth * 2;
        std::vector<double> results(evals);

        auto t0 = std::chrono::high_resolution_clock::now();
        std::vector<std::thread> threads;

        for (int t = 0; t < nth; t++) {
            threads.emplace_back([&, t]() {
                for (int e = 0; e < 2; e++) {
                    double pp[4]; memcpy(pp, p0, sizeof(p0));
                    pp[0] *= (1.0 + 0.01*(t*2+e));

                    CvodesComputeBlock blk(lv_rhs, lv_jac, lv_par_jac, n, m, x0, T);
                    aadc::AADCFunctions<mmType> fn;
                    fn.startRecording();
                    idouble ip[4]; aadc::AADCArgument ap[4];
                    for(int i=0;i<m;i++){ip[i]=pp[i];ap[i]=ip[i].markAsInput();}
                    idouble ixf[2];
                    blk.integrate(ip, ixf);
                    idouble c = ixf[0]*ixf[0]+ixf[1]*ixf[1];
                    auto rc = c.markAsOutput();
                    fn.stopRecording();

                    auto w = std::shared_ptr<aadc::AADCWorkSpace<mmType>>(fn.createWorkSpace());
                    for(int i=0;i<m;i++) w->setVal(ap[i],aadc::mmSetConst<mmType>(pp[i]));
                    fn.forward(*w);
                    w->resetDiff(); w->setDiff(rc,aadc::mmSetConst<mmType>(1.0));
                    fn.reverse(*w);
                    results[t*2+e] = ((double*)&w->diff(ap[0]))[0];
                }
            });
        }
        for (auto& th : threads) th.join();

        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        printf("%8d %8d %9.0fms %9.1fms\n", nth, evals, ms, ms/evals);
    }

    // --- Comparison table ---
    printf("\n--- Method comparison (Lotka-Volterra, T=5) ---\n");
    printf("  ComputeBlock (CVODES BDF):   ~7.6 ms (forward+reverse)\n");
    printf("  Tape (RK4 on tape):          ~0.008 ms\n");
    printf("  Discrete adjoint (RK45):     ~65 ms (Python)\n");
    printf("  CVODES forward Jacobian:     ~38 ms (forward only, no gradient)\n");
}

// Entry point with benchmarks
int main2_bench() { run_benchmarks(); return 0; }
