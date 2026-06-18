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

    return 0;
}
