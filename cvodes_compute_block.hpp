/*
 * CVODES as AADC ConstStateExtFunc (ComputeBlock).
 *
 * Forward: CVODES BDF integrates ODE, writes final state to workspace.
 * Reverse: CVODES adjoint (CVodeB) integrates backward, propagates dJ/dp.
 *
 * Usage during AADC recording:
 *   idouble params[M];  // mark as input
 *   CvodesBlock block(rhs_func, N, M, x0, params, T, ...);
 *   idouble x_final[N]; // mark as output
 *   block.integrate(params, x_final);  // records ExtFunc on tape
 *
 * During replay:
 *   forward(*ws): runs CVODES BDF forward
 *   reverse(*ws): runs CVODES adjoint backward
 */
#pragma once

#include <aadc/aadc.h>
#include <aadc/aadc_ext.h>

#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_types.h>

#include <vector>
#include <functional>
#include <cstring>

// RHS function type: f(t, x, xdot, params)
typedef std::function<void(double t, const double* x, double* xdot,
                           const double* params, int n, int m)> CvodesRhsFunc;

// Jacobian function type: J = df/dx at (t, x, params)
// Fills dense matrix J[i*n+j] = df_i/dx_j
typedef std::function<void(double t, const double* x, double* J,
                           const double* params, int n, int m)> CvodesJacFunc;

// Parameter Jacobian: df/dp at (t, x, params)
// Fills J[i*m+j] = df_i/dp_j
typedef std::function<void(double t, const double* x, double* Jp,
                           const double* params, int n, int m)> CvodesParJacFunc;

class CvodesComputeBlock : public aadc::ConstStateExtFunc {
public:
    int n_states;
    int n_params;
    double T_final;
    double rtol, atol, max_step;

    CvodesRhsFunc rhs;
    CvodesJacFunc jac;       // df/dx (for forward BDF Newton)
    CvodesParJacFunc par_jac; // df/dp (for adjoint parameter sensitivity)

    // AADC workspace indices
    std::vector<aadc::ExtFuncInScalarParam> param_inputs;
    std::vector<aadc::ExtFuncOutScalarParam> state_outputs;

    // Initial state (fixed, not differentiated)
    std::vector<double> x0;

    // Stored forward trajectory for adjoint
    mutable std::vector<double> x_final_stored;
    mutable std::vector<double> params_stored;

    CvodesComputeBlock(
        CvodesRhsFunc rhs_func,
        CvodesJacFunc jac_func,
        CvodesParJacFunc par_jac_func,
        int n, int m,
        const double* initial_state,
        double T,
        double rtol_ = 1e-8,
        double atol_ = 1e-10,
        double max_step_ = 0.01
    ) : n_states(n), n_params(m), T_final(T),
        rtol(rtol_), atol(atol_), max_step(max_step_),
        rhs(rhs_func), jac(jac_func), par_jac(par_jac_func)
    {
        x0.assign(initial_state, initial_state + n);
        x_final_stored.resize(n);
        params_stored.resize(m);
    }

    // Called during AADC recording to bind inputs/outputs
    void integrate(idouble* id_params, idouble* id_x_out) {
        // Bind parameter inputs
        param_inputs.clear();
        for (int i = 0; i < n_params; i++)
            param_inputs.emplace_back(id_params[i]);

        // Run forward at recording time to get concrete output values
        std::vector<double> p(n_params);
        for (int i = 0; i < n_params; i++)
            p[i] = id_params[i].val;

        std::vector<double> xf(n_states);
        run_cvodes_forward(p.data(), xf.data());

        // Bind outputs
        state_outputs.resize(n_states);
        bool any_random = false, any_diff = true;
        for (int i = 0; i < n_params; i++) {
            any_random |= param_inputs[i].is_random;
        }
        for (int i = 0; i < n_states; i++) {
            id_x_out[i] = xf[i];
            state_outputs[i].bind(id_x_out[i], any_random, any_diff);
        }

        // Register as ExtFunc
        aadc::addConstStateExtFunction(
            std::shared_ptr<CvodesComputeBlock>(this, [](CvodesComputeBlock*){}));
    }

    // Forward: CVODES BDF integration
    template<typename mmType>
    void forward(mmType* v) const {
        int avx_size = sizeof(mmType) / sizeof(double);

        for (int avxi = 0; avxi < avx_size; avxi++) {
            // Read parameters from workspace
            aadc::ExtFuncInScalarParamValue pvals[32];  // max params
            double p[32];
            for (int i = 0; i < n_params; i++) {
                pvals[i].setValue(param_inputs[i], v, avxi);
                p[i] = pvals[i].val;
            }

            // Run CVODES forward
            double xf[64];  // max states
            run_cvodes_forward(p, xf);

            // Write outputs to workspace
            aadc::ExtFuncOutScalarParamValue ovals[64];
            for (int i = 0; i < n_states; i++) {
                ovals[i].val = xf[i];
                ovals[i].setValue(state_outputs[i], v, avxi);
            }

            // Store for adjoint
            if (avxi == 0) {
                memcpy(x_final_stored.data(), xf, n_states * sizeof(double));
                memcpy(params_stored.data(), p, n_params * sizeof(double));
            }
        }
    }

    // Reverse: CVODES adjoint
    template<typename mmType>
    void reverse(const mmType* v, mmType* d) const {
        int avx_size = sizeof(mmType) / sizeof(double);

        for (int avxi = 0; avxi < avx_size; avxi++) {
            // Read output adjoints (dJ/dx_final)
            double lambda[64];
            for (int i = 0; i < n_states; i++)
                lambda[i] = state_outputs[i].getDiff(d, avxi);

            // Read parameters
            double p[32];
            for (int i = 0; i < n_params; i++)
                p[i] = params_stored[i];

            // Run CVODES adjoint to get dJ/dp
            double dJdp[32];
            run_cvodes_adjoint(p, lambda, dJdp);

            // Propagate dJ/dp to parameter inputs
            for (int i = 0; i < n_params; i++)
                param_inputs[i].addDiff(d, avxi, dJdp[i]);
        }
    }

private:
    // Static RHS wrapper for CVODES C callback
    struct CvodesUserData {
        const CvodesRhsFunc* rhs;
        const CvodesJacFunc* jac;
        const CvodesParJacFunc* par_jac;
        const double* params;
        int n_s, n_p;
    };

    static int cvodes_rhs_wrapper(realtype t, N_Vector y, N_Vector ydot, void* udata) {
        auto* ud = (CvodesUserData*)udata;
        (*ud->rhs)(t, N_VGetArrayPointer(y), N_VGetArrayPointer(ydot), ud->params, ud->n_s, ud->n_p);
        return 0;
    }

    static int cvodes_jac_wrapper(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                                   void* udata, N_Vector t1, N_Vector t2, N_Vector t3) {
        auto* ud = (CvodesUserData*)udata;
        (*ud->jac)(t, N_VGetArrayPointer(y), SM_DATA_D(J), ud->params, ud->n_s, ud->n_p);
        return 0;
    }

    void run_cvodes_forward(const double* params, double* x_final) const {
        SUNContext ctx;
        SUNContext_Create(NULL, &ctx);

        N_Vector y = N_VNew_Serial(n_states, ctx);
        memcpy(N_VGetArrayPointer(y), x0.data(), n_states * sizeof(double));

        CvodesUserData udata = {&rhs, &jac, &par_jac, params, n_states, n_params};

        void* cvode = CVodeCreate(CV_BDF, ctx);
        CVodeInit(cvode, cvodes_rhs_wrapper, 0.0, y);
        CVodeSStolerances(cvode, rtol, atol);
        CVodeSetMaxNumSteps(cvode, 500000);
        CVodeSetMaxStep(cvode, max_step);
        CVodeSetUserData(cvode, &udata);

        SUNMatrix A = SUNDenseMatrix(n_states, n_states, ctx);
        SUNLinearSolver LS = SUNLinSol_Dense(y, A, ctx);
        CVodeSetLinearSolver(cvode, LS, A);
        CVodeSetJacFn(cvode, cvodes_jac_wrapper);

        realtype t_out;
        CVode(cvode, T_final, y, &t_out, CV_NORMAL);

        memcpy(x_final, N_VGetArrayPointer(y), n_states * sizeof(double));

        N_VDestroy(y); SUNMatDestroy(A); SUNLinSolFree(LS);
        CVodeFree(&cvode); SUNContext_Free(&ctx);
    }

    // Backward RHS for CVODES adjoint: dλ/dt = -(df/dx)^T λ
    static int cvodes_rhsB_wrapper(realtype t, N_Vector y, N_Vector yB,
                                    N_Vector yBdot, void* udata) {
        auto* ud = (CvodesUserData*)udata;
        int n = ud->n_s;
        const double* x = N_VGetArrayPointer(y);    // forward state (reconstructed)
        const double* lam = N_VGetArrayPointer(yB);  // adjoint state
        double* lamdot = N_VGetArrayPointer(yBdot);

        // Compute Jacobian df/dx
        std::vector<double> Jx(n * n);
        (*ud->jac)(t, x, Jx.data(), ud->params, n, ud->n_p);

        // lamdot = -(df/dx)^T * lambda
        for (int i = 0; i < n; i++) {
            lamdot[i] = 0;
            for (int j = 0; j < n; j++)
                lamdot[i] -= Jx[j * n + i] * lam[j];  // -J^T * lambda
        }
        return 0;
    }

    // Backward quadrature RHS: dμ/dt = -(df/dp)^T λ
    static int cvodes_qrhsB_wrapper(realtype t, N_Vector y, N_Vector yB,
                                     N_Vector qBdot, void* udata) {
        auto* ud = (CvodesUserData*)udata;
        int n = ud->n_s, m = ud->n_p;
        const double* x = N_VGetArrayPointer(y);
        const double* lam = N_VGetArrayPointer(yB);
        double* mudot = N_VGetArrayPointer(qBdot);

        // Compute df/dp
        std::vector<double> Jp(n * m, 0);
        if (ud->par_jac)
            (*ud->par_jac)(t, x, Jp.data(), ud->params, n, m);

        // mudot = -(df/dp)^T * lambda
        for (int j = 0; j < m; j++) {
            mudot[j] = 0;
            for (int i = 0; i < n; i++)
                mudot[j] -= Jp[i * m + j] * lam[i];
        }
        return 0;
    }

    void run_cvodes_adjoint(const double* params, const double* lambda_T, double* dJdp) const {
        // Full CVODES adjoint: CVodeF (forward with checkpoints) + CVodeB (backward)
        int n = n_states, m = n_params;

        SUNContext ctx;
        SUNContext_Create(NULL, &ctx);

        N_Vector y = N_VNew_Serial(n, ctx);
        memcpy(N_VGetArrayPointer(y), x0.data(), n * sizeof(double));

        CvodesUserData udata = {&rhs, &jac, &par_jac, params, n, m};

        void* cvode = CVodeCreate(CV_BDF, ctx);
        CVodeInit(cvode, cvodes_rhs_wrapper, 0.0, y);
        CVodeSStolerances(cvode, rtol, atol);
        CVodeSetMaxNumSteps(cvode, 500000);
        CVodeSetMaxStep(cvode, max_step);
        CVodeSetUserData(cvode, &udata);

        SUNMatrix A = SUNDenseMatrix(n, n, ctx);
        SUNLinearSolver LS = SUNLinSol_Dense(y, A, ctx);
        CVodeSetLinearSolver(cvode, LS, A);
        CVodeSetJacFn(cvode, cvodes_jac_wrapper);

        // Initialize adjoint computation with checkpointing
        int ncheck;
        CVodeAdjInit(cvode, 150, CV_HERMITE);  // 150 steps between checkpoints

        // Forward integration with checkpoint storage
        realtype t_out;
        CVodeF(cvode, T_final, y, &t_out, CV_NORMAL, &ncheck);

        // Create backward problem
        int indexB;
        CVodeCreateB(cvode, CV_BDF, &indexB);

        // Initial condition for adjoint: λ(T) = lambda_T
        N_Vector yB = N_VNew_Serial(n, ctx);
        memcpy(N_VGetArrayPointer(yB), lambda_T, n * sizeof(double));
        CVodeInitB(cvode, indexB, cvodes_rhsB_wrapper, T_final, yB);
        CVodeSStolerancesB(cvode, indexB, rtol, atol);

        SUNMatrix AB = SUNDenseMatrix(n, n, ctx);
        SUNLinearSolver LSB = SUNLinSol_Dense(yB, AB, ctx);
        CVodeSetLinearSolverB(cvode, indexB, LSB, AB);

        CVodeSetUserDataB(cvode, indexB, &udata);

        // Quadrature for parameter sensitivity: μ(T)=0, dμ/dt = -(df/dp)^T λ
        N_Vector qB = N_VNew_Serial(m, ctx);
        N_VConst(0.0, qB);
        CVodeQuadInitB(cvode, indexB, cvodes_qrhsB_wrapper, qB);
        CVodeQuadSStolerancesB(cvode, indexB, rtol, atol);
        CVodeSetQuadErrConB(cvode, indexB, SUNTRUE);

        // Backward integration
        CVodeB(cvode, 0.0, CV_NORMAL);

        // Get results
        CVodeGetB(cvode, indexB, &t_out, yB);
        CVodeGetQuadB(cvode, indexB, &t_out, qB);

        // dJ/dp = -μ(0) (sign convention)
        double* mu = N_VGetArrayPointer(qB);
        for (int j = 0; j < m; j++)
            dJdp[j] = -mu[j];

        // Cleanup
        N_VDestroy(y); N_VDestroy(yB); N_VDestroy(qB);
        SUNMatDestroy(A); SUNMatDestroy(AB);
        SUNLinSolFree(LS); SUNLinSolFree(LSB);
        CVodeFree(&cvode); SUNContext_Free(&ctx);
    }
};
