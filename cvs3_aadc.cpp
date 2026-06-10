/* 3-Compartment Cardiovascular Model — AADC port.
 * Ported from circulatory_autogen generated 3compartment.py
 * 27 states, 143 algebraic variables, 4 calibration parameters.
 *
 * Key feature: conditionals (valve logic) handled via iIf.
 * CasADI crashes on this model; AADC handles it.
 *
 * Integration: semi-implicit Euler (IMEX).
 *   Stiff states (flow through inductances, valve flows) are treated
 *   with implicit diagonal damping: y += dt*f/(1 + dt*λ).
 *   Non-stiff states (charges, volumes, timing) use forward Euler.
 *
 * Usage: ./cvs3_aadc [--threads N] [--iters N] [--steps N]
 *   --steps: ODE steps for sim_time (default: 200 = 2s at dt=0.01)
 *   --pre_steps: ODE steps for pre_time (default: 2000 = 20s)
 *   --threads: worker threads (default: 1)
 *   --iters: benchmark iterations (default: 100)
 */
#include <cstdio>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <thread>
#include <aadc/aadc.h>

#ifdef AADC_512
typedef __m512d mmType;
#else
typedef __m256d mmType;
#endif

inline mmType mm_lane0(double v){
    mmType r=aadc::mmSetConst<mmType>(0.0);((double*)&r)[0]=v;return r;}

// ========== Model Parameters (constants) ==========
struct CVSParams {
    double R_pvn = 1333000.0;
    double C_pvn = 0.0000000060015;
    double I_pvn = 0.000001;
    double q_C_init_pvn = 0.0001;
    double q_us_0_pvn = 0.0;
    double delta_q_us_pvn = 0.0;
    double u_ext_pvn = 0.0;
    double delta_C_pvn = 0.0;
    double q_0_par = 0.0;
    double R_par = 10664000.0;
    double C_par = 3.09077e-10;
    double I_par = 0.000001;
    double u_0_par = 1463.0;
    double u_ext_par = 0.0;
    double q_0_aortic = 6.94e-06;
    double R_aortic = 1000000.0;
    double C_aortic = 0.000000012028;  // calibration param
    double I_aortic = 10000.0;
    double u_0_aortic = 13300.0;
    double u_ext_aortic = 0.0;
    double R_t_sys = 110000000.0;
    double C_t_sys = 0.0000001;
    double q_us_sys = 0.00245;
    double q_init_sys = 0.00245;
    double u_ext_sys = 0.0;
    double R_venous = 1114600.0;
    double C_venous = 0.000001;
    double I_venous = 0.01;
    double q_C_init_venous = 0.0013;
    double q_us_0_venous = 0.0;
    double delta_q_us_venous = 0.0;
    double u_ext_venous = 0.0;
    double delta_C_venous = 0.0;
    double rho = 1050.0;
    double T_period = 1.0;
    double q_ra_us = 0.000004;
    double q_rv_us = 0.00001;
    double q_la_us = 0.000004;
    double q_lv_us = 0.000005;
    double q_ra_init = 0.000004;
    double q_rv_init = 0.00001;
    double q_la_init = 0.000004;
    double q_lv_init = 0.002;  // calibration param
    double t_ac = 0.17, t_ar = 0.17, t_astart = 0.8;
    double t_vc = 0.30, t_vr = 0.15, t_vstart = 0.0;
    double E_ra_A = 7998000.0, E_ra_B = 9331000.0;
    double E_rv_A = 73315000.0, E_rv_B = 6665000.0;
    double E_la_A = 9331000.0, E_la_B = 11997000.0;
    double E_lv_A = 366575000.0;  // calibration param
    double E_lv_B = 10664000.0;   // calibration param
    double k_vo_trv = 0.3, k_vo_puv = 0.2, k_vo_miv = 0.3, k_vo_aov = 0.04;
    double k_vc_trv = 0.4, k_vc_puv = 0.2, k_vc_miv = 0.4, k_vc_aov = 0.04;
    double m_rg_trv = 0.0, m_rg_puv = 0.0, m_rg_miv = 0.0, m_rg_aov = 0.0;
    double m_st_trv = 1.0, m_st_puv = 1.0, m_st_miv = 1.0, m_st_aov = 1.0;
    double l_eff = 0.01;
    double a_nn_trv = 0.0009, a_nn_puv = 0.0004, a_nn_miv = 0.0006, a_nn_aov = 0.000314;
    double eps_1 = 0.07, eps_2 = 0.02, eps_m4 = 1e-14, eps_m2 = 1e-14;
    double I_t_sys = 1e-6;
};

// State indices
enum S {
    V_PVN=0, V_PAR, QC_PVN, QC_PAR, V_PUV,
    CHI_A, CHI_V, S_HEART, ZETA_TRV, ZETA_PUV,
    ZETA_MIV, ZETA_AOV, V_TRV, V_MIV, V_AOV,
    Q_RA, Q_RV, Q_LA, Q_LV, V_VENOUS,
    QCD_AORTIC, QC_AORTIC, V_AORTIC, V_SYS, VT_SYS,
    Q_SYS, QC_VENOUS, N_STATES // = 27
};

static const char* state_names[] = {
    "V_PVN", "V_PAR", "QC_PVN", "QC_PAR", "V_PUV",
    "CHI_A", "CHI_V", "S_HEART", "ZETA_TRV", "ZETA_PUV",
    "ZETA_MIV", "ZETA_AOV", "V_TRV", "V_MIV", "V_AOV",
    "Q_RA", "Q_RV", "Q_LA", "Q_LV", "V_VENOUS",
    "QCD_AORTIC", "QC_AORTIC", "V_AORTIC", "V_SYS", "VT_SYS",
    "Q_SYS", "QC_VENOUS"
};

// Compute ODE rates and diagonal damping coefficients.
// rates[i] = f_i(y)
// lambda[i] = -∂f_i/∂y_i (diagonal of negative Jacobian, for stiff states)
// Semi-implicit update: y_i += dt * rates[i] / (1 + dt * lambda[i])
template<typename T>
void compute_rates_and_damping(const T* st, T* rates, T* lambda,
                               const CVSParams& p,
                               T q_lv_init, T C_aortic, T E_lv_A, T E_lv_B) {
    const double PI = 3.14159265358979;

    // Initialize lambda to 0 (forward Euler for non-stiff states)
    for (int i = 0; i < N_STATES; i++) lambda[i] = T(0.0);

    // Viscoelastic coefficients
    T r_v_pvn = 0.01 / p.C_pvn;
    T r_v_par = 0.01 / p.C_par;
    T r_v_aortic = 0.01 / C_aortic;
    T r_v_sys = 0.01 / p.C_t_sys;
    T r_v_venous = 0.01 / p.C_venous;

    // Computed constants
    T q_us_wcont_pvn = p.q_us_0_pvn * (1.0 - p.delta_q_us_pvn);
    T c_wcont_pvn = p.C_pvn * (1.0 - p.delta_C_pvn);
    T t_astart_norm = p.t_astart / p.T_period;
    T t_vstart_norm = p.t_vstart / p.T_period;
    T q_us_wcont_venous = p.q_us_0_venous * (1.0 - p.delta_q_us_venous);
    T c_wcont_venous = p.C_venous * (1.0 - p.delta_C_venous);

    // Pulmonary venous pressure
    T q_c_pvn = st[QC_PVN] + p.q_us_0_pvn - q_us_wcont_pvn;
    T u_c_pvn = q_c_pvn / c_wcont_pvn;
    T u_pvn = u_c_pvn + p.u_ext_pvn + r_v_pvn * (st[V_PAR] - st[V_PVN]);

    // Cardiac activation (atrial)
    T chi_afloor = st[CHI_A] - floor(st[CHI_A]);
    T chi_afloor_final = iIf(chi_afloor <= 0.5, chi_afloor * 2.0, T(0.0));
    T e_a = 0.5 * (1.0 - cos(2.0 * PI * chi_afloor_final));

    T u_la = (e_a * p.E_la_A + p.E_la_B) * (st[Q_LA] - p.q_la_us);

    // V_PVN: rate = (u_pvn - u_la - R_pvn * V_PVN) / I_pvn
    // Self-coupling: -(R_pvn + r_v_pvn) / I_pvn
    rates[V_PVN] = (u_pvn - u_la - p.R_pvn * st[V_PVN]) / p.I_pvn;
    lambda[V_PVN] = (p.R_pvn + r_v_pvn) / p.I_pvn;

    rates[QC_PVN] = st[V_PAR] - st[V_PVN];

    // Pulmonary arterial pressure
    T u_c_par = st[QC_PAR] / p.C_par;
    T u_par = p.u_0_par + u_c_par + p.u_ext_par + r_v_par * (st[V_PUV] - st[V_PAR]);

    // V_PAR: rate = (u_par - u_pvn - R_par * V_PAR) / I_par
    // Self-coupling through u_par (-r_v_par*V_PAR) and -u_pvn (-r_v_pvn*V_PAR via u_pvn)
    // Wait: u_pvn includes +r_v_pvn*(V_PAR - V_PVN), so -u_pvn has -r_v_pvn*V_PAR
    rates[V_PAR] = (u_par - u_pvn - p.R_par * st[V_PAR]) / p.I_par;
    lambda[V_PAR] = (p.R_par + r_v_par + r_v_pvn) / p.I_par;

    rates[QC_PAR] = st[V_PUV] - st[V_PAR];

    // Heart timing
    T mt = st[S_HEART] - floor(st[S_HEART]);

    // Atrial activation rate
    T chi_af = chi_afloor;
    T rate_chi_a;
    {
        T r1 = 0.25 / p.t_ac;
        T r2 = 0.25 / p.t_ar;
        T r3 = 0.5 / (p.T_period - p.t_ac - p.t_ar);
        T trig = iIf(mt >= t_astart_norm,
                     iIf(mt <= t_astart_norm + p.eps_1,
                         iIf(chi_af <= 0.25, r1, T(0.0)),
                         T(0.0)),
                     T(0.0));
        T ongoing = iIf(chi_af > p.eps_2,
                       iIf(chi_af <= 0.25, r1, T(0.0)),
                       T(0.0));
        T relax = iIf(chi_af >= 0.25,
                     iIf(chi_af < 0.5, r2, T(0.0)),
                     T(0.0));
        T diast = iIf(chi_af >= 0.5, r3, T(0.0));
        rate_chi_a = trig + ongoing + relax + diast;
    }
    rates[CHI_A] = rate_chi_a;

    // Ventricular activation
    T chi_vfloor = st[CHI_V] - floor(st[CHI_V]);
    T chi_vfloor_final = iIf(chi_vfloor <= 0.5, chi_vfloor * 2.0, T(0.0));
    T e_v = 0.5 * (1.0 - cos(2.0 * PI * chi_vfloor_final));

    T rate_chi_v;
    {
        T r1 = 0.25 / p.t_vc;
        T r2 = 0.25 / p.t_vr;
        T r3 = 0.5 / (p.T_period - p.t_vc - p.t_vr);
        T trig = iIf(mt >= t_vstart_norm,
                     iIf(mt <= t_vstart_norm + p.eps_1,
                         iIf(chi_vfloor <= 0.25, r1, T(0.0)),
                         T(0.0)),
                     T(0.0));
        T ongoing = iIf(chi_vfloor > p.eps_2,
                       iIf(chi_vfloor <= 0.25, r1, T(0.0)),
                       T(0.0));
        T relax = iIf(chi_vfloor >= 0.25,
                     iIf(chi_vfloor < 0.5, r2, T(0.0)),
                     T(0.0));
        T diast = iIf(chi_vfloor >= 0.5, r3, T(0.0));
        rate_chi_v = trig + ongoing + relax + diast;
    }
    rates[CHI_V] = rate_chi_v;
    rates[S_HEART] = 1.0 / p.T_period;

    // Chamber pressures
    T u_rv = (e_v * p.E_rv_A + p.E_rv_B) * (st[Q_RV] - p.q_rv_us);
    T u_ra = (e_a * p.E_ra_A + p.E_ra_B) * (st[Q_RA] - p.q_ra_us);
    T u_lv = (e_v * E_lv_A + E_lv_B) * (st[Q_LV] - p.q_lv_us);

    // Aortic root pressure
    T u_c_aortic = st[QC_AORTIC] / (C_aortic / 2.0);
    T u_aortic = p.u_0_aortic + u_c_aortic + p.u_ext_aortic +
                 2.0 * r_v_aortic * (st[V_AOV] - st[V_AORTIC]);

    // Valve opening dynamics (zeta)
    // rate_opening = (1 - ζ) * k_vo * Δu → λ = k_vo * Δu (self-damping)
    // rate_closing = ζ * k_vc * Δu (Δu<0) → λ = k_vc * |Δu|
    auto do_zeta = [&](int idx, T u_up, T u_down, double k_vo, double k_vc) {
        T du = u_up - u_down;
        T du_abs = iIf(du >= 0.0, du, -du);
        rates[idx] = iIf(u_up >= u_down,
            (1.0 - st[idx]) * k_vo * du,
            st[idx] * k_vc * du);
        lambda[idx] = iIf(u_up >= u_down, k_vo * du_abs, k_vc * du_abs);
    };

    do_zeta(ZETA_TRV, u_ra, u_rv, p.k_vo_trv, p.k_vc_trv);
    do_zeta(ZETA_PUV, u_rv, u_par, p.k_vo_puv, p.k_vc_puv);
    do_zeta(ZETA_MIV, u_la, u_lv, p.k_vo_miv, p.k_vc_miv);
    do_zeta(ZETA_AOV, u_lv, u_aortic, p.k_vo_aov, p.k_vc_aov);

    // Valve flow computation helper
    auto compute_valve = [&](int zeta_idx, double m_st, double a_nn, double m_rg) {
        T zeta = iIf(st[zeta_idx] >= 0.0, st[zeta_idx], T(0.0));
        T a_eff = (m_st * a_nn - m_rg * a_nn) * zeta + m_rg * a_nn;
        T l = p.rho * p.l_eff / (a_eff + p.eps_m2);
        T b = p.rho / (2.0 * a_eff * a_eff + p.eps_m4);
        return std::make_pair(l, b);
    };

    // Valve flows: rate = (-b*v*|v| + Δu) / l
    // Linearized damping: λ = 2*b*|v| / l
    auto do_valve_flow = [&](int v_idx, int zeta_idx,
                             double m_st, double a_nn, double m_rg,
                             T u_up, T u_down) {
        auto [l, b] = compute_valve(zeta_idx, m_st, a_nn, m_rg);
        T v = st[v_idx];
        T v_fabs = iIf(v >= 0.0, v, -v);
        rates[v_idx] = (-b * v * v_fabs + u_up - u_down) / l;
        lambda[v_idx] = 2.0 * b * v_fabs / l;
    };

    do_valve_flow(V_TRV, ZETA_TRV, p.m_st_trv, p.a_nn_trv, p.m_rg_trv, u_ra, u_rv);
    do_valve_flow(V_PUV, ZETA_PUV, p.m_st_puv, p.a_nn_puv, p.m_rg_puv, u_rv, u_par);
    do_valve_flow(V_MIV, ZETA_MIV, p.m_st_miv, p.a_nn_miv, p.m_rg_miv, u_la, u_lv);
    do_valve_flow(V_AOV, ZETA_AOV, p.m_st_aov, p.a_nn_aov, p.m_rg_aov, u_lv, u_aortic);

    // Chamber volumes
    rates[Q_RA] = st[V_VENOUS] - st[V_TRV];
    rates[Q_RV] = st[V_TRV] - st[V_PUV];
    rates[Q_LA] = st[V_PVN] - st[V_MIV];
    rates[Q_LV] = st[V_MIV] - st[V_AOV];

    // Aortic root dynamics
    T u_c_d_aortic = st[QCD_AORTIC] / (C_aortic / 2.0);
    T u_d_aortic = p.u_0_aortic + u_c_d_aortic + p.u_ext_aortic +
                   2.0 * r_v_aortic * (st[V_AORTIC] - st[V_SYS]);

    // V_AORTIC: self-coupling through u_aortic (-2*r_v*V_AORTIC) and
    //           -u_d_aortic (-2*r_v*V_AORTIC), total -4*r_v - R
    rates[V_AORTIC] = (u_aortic - u_d_aortic - p.R_aortic * st[V_AORTIC]) / p.I_aortic;
    lambda[V_AORTIC] = (p.R_aortic + 4.0 * r_v_aortic) / p.I_aortic;

    rates[QC_AORTIC] = st[V_AOV] - st[V_AORTIC];
    rates[QCD_AORTIC] = st[V_AORTIC] - st[V_SYS];

    // Systemic
    rates[Q_SYS] = st[V_SYS] - st[VT_SYS];
    T u_c_sys = (st[Q_SYS] - p.q_us_sys) / p.C_t_sys;
    T u_sys = u_c_sys + p.u_ext_sys + r_v_sys * (st[V_SYS] - st[VT_SYS]);

    // V_SYS: self-coupling: -(2*r_v_aortic + r_v_sys + R_t_sys/2) / I_t_sys
    rates[V_SYS] = (u_d_aortic - u_sys - st[V_SYS] * p.R_t_sys / 2.0) / p.I_t_sys;
    lambda[V_SYS] = (p.R_t_sys / 2.0 + 2.0 * r_v_aortic + r_v_sys) / p.I_t_sys;

    // Venous
    T q_c_venous = st[QC_VENOUS] + p.q_us_0_venous - q_us_wcont_venous;
    T u_c_venous = q_c_venous / c_wcont_venous;
    T v_venous_in = st[VT_SYS];
    T u_venous = u_c_venous + p.u_ext_venous + r_v_venous * (v_venous_in - st[V_VENOUS]);

    // VT_SYS: self-coupling: -(R_t_sys/2 + r_v_sys + r_v_venous) / I_t_sys
    rates[VT_SYS] = (u_sys - u_venous - st[VT_SYS] * p.R_t_sys / 2.0) / p.I_t_sys;
    lambda[VT_SYS] = (p.R_t_sys / 2.0 + r_v_sys + r_v_venous) / p.I_t_sys;

    // V_VENOUS: self-coupling: -(R_venous + r_v_venous) / I_venous
    rates[V_VENOUS] = (u_venous - u_ra - p.R_venous * st[V_VENOUS]) / p.I_venous;
    lambda[V_VENOUS] = (p.R_venous + r_v_venous) / p.I_venous;

    rates[QC_VENOUS] = v_venous_in - st[V_VENOUS];
}

// ========== Plain double version for verification ==========
void run_double_verification(const CVSParams& p, int total_steps, double dt) {
    printf("=== Plain double verification (semi-implicit Euler) ===\n");

    double st[N_STATES] = {0};
    st[QC_PVN] = p.q_C_init_pvn;
    st[Q_RA] = p.q_ra_init;
    st[Q_RV] = p.q_rv_init;
    st[Q_LA] = p.q_la_init;
    st[Q_LV] = p.q_lv_init;
    st[QC_VENOUS] = p.q_C_init_venous;
    st[Q_SYS] = p.q_init_sys;

    double rates[N_STATES], lambda[N_STATES];
    double st_prev[N_STATES];

    auto t0 = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < total_steps; step++) {
        for (int i = 0; i < N_STATES; i++) st_prev[i] = st[i];

        compute_rates_and_damping(st, rates, lambda, p,
            p.q_lv_init, p.C_aortic, p.E_lv_A, p.E_lv_B);

        for (int i = 0; i < N_STATES; i++)
            st[i] += dt * rates[i] / (1.0 + dt * lambda[i]);

        // Clamp zeta to [0, 1]
        for (int z : {ZETA_TRV, ZETA_PUV, ZETA_MIV, ZETA_AOV}) {
            if (st[z] < 0.0) st[z] = 0.0;
            if (st[z] > 1.0) st[z] = 1.0;
        }

        // Check for NaN/Inf
        double t = (step + 1) * dt;
        bool any_nan = false;
        for (int i = 0; i < N_STATES; i++)
            if (std::isnan(st[i]) || std::isinf(st[i])) { any_nan = true; break; }

        if (any_nan) {
            printf("  t=%.4f (step %d): *** NaN/Inf detected! ***\n", t, step+1);
            printf("  Previous state (before update):\n");
            for (int i = 0; i < N_STATES; i++)
                printf("    %-14s = %12.6e  rate=%12.6e  lam=%12.6e  delta=%12.6e\n",
                       state_names[i], st_prev[i], rates[i], lambda[i],
                       dt * rates[i] / (1.0 + dt * lambda[i]));
            printf("  Current state (after update):\n");
            for (int i = 0; i < N_STATES; i++)
                printf("    %-14s = %12.6e\n", state_names[i], st[i]);
            return;
        }

        if (step < 5 || step % 500 == 499 || step == total_steps - 1) {
            double chi_v = st[CHI_V] - floor(st[CHI_V]);
            double chi_vf = (chi_v <= 0.5) ? chi_v * 2.0 : 0.0;
            double e_v = 0.5 * (1.0 - cos(2.0 * 3.14159265 * chi_vf));
            double u_lv = (e_v * p.E_lv_A + p.E_lv_B) * (st[Q_LV] - p.q_lv_us);
            printf("  t=%6.2f: Q_LV=%.4e u_LV=%10.1f V_PVN=%10.4e V_AOV=%10.4e\n",
                   t, st[Q_LV], u_lv, st[V_PVN], st[V_AOV]);
        }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    // Final state summary
    printf("\n  Final state (t=%.1fs):\n", total_steps * dt);
    for (int i = 0; i < N_STATES; i++)
        printf("    %-14s = %12.6e\n", state_names[i], st[i]);

    // Compute LV pressure at final time
    double chi_v = st[CHI_V] - floor(st[CHI_V]);
    double chi_vf = (chi_v <= 0.5) ? chi_v * 2.0 : 0.0;
    double e_v = 0.5 * (1.0 - cos(2.0 * 3.14159265 * chi_vf));
    double u_lv = (e_v * p.E_lv_A + p.E_lv_B) * (st[Q_LV] - p.q_lv_us);
    printf("\n  LV pressure: %.1f Pa = %.1f mmHg\n", u_lv, u_lv / 133.322);
    printf("  Cost (Q_LV²): %.6e\n", st[Q_LV] * st[Q_LV]);
    printf("  Elapsed: %.3f ms\n\n", elapsed * 1000.0);
}

int main(int argc, char* argv[]) {
    setbuf(stdout, NULL);

    int pre_steps = 2000, sim_steps = 200, n_threads = 1, n_iters = 100;
    double dt = 0.01;
    bool skip_aadc = false;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--steps" && i+1 < argc) sim_steps = atoi(argv[++i]);
        else if (a == "--pre_steps" && i+1 < argc) pre_steps = atoi(argv[++i]);
        else if (a == "--threads" && i+1 < argc) n_threads = atoi(argv[++i]);
        else if (a == "--iters" && i+1 < argc) n_iters = atoi(argv[++i]);
        else if (a == "--dt" && i+1 < argc) dt = atof(argv[++i]);
        else if (a == "--double-only") skip_aadc = true;
        else if (a == "--help") {
            printf("Usage: %s [--steps N] [--pre_steps N] [--threads N] [--iters N] [--dt F] [--double-only]\n", argv[0]);
            return 0;
        }
    }

    const int AVX_BATCH = aadc::mmSize<mmType>();
    CVSParams params;
    int total_steps = pre_steps + sim_steps;

    printf("3-Compartment CVS Model — AADC Benchmark\n");
    printf("  States: %d, dt: %.4f, Method: semi-implicit Euler\n", N_STATES, dt);
    printf("  Pre-time: %.1fs (%d steps), Sim-time: %.1fs (%d steps)\n",
           pre_steps * dt, pre_steps, sim_steps * dt, sim_steps);
    printf("  Threads: %d, AVX batch: %d, Iters: %d\n\n", n_threads, AVX_BATCH, n_iters);

    // Step 1: verify with plain doubles
    run_double_verification(params, total_steps, dt);

    if (skip_aadc) return 0;

    // ========== Record AADC kernel ==========
    printf("Recording AADC kernel...\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    aadc::AADCFunctions<mmType> funcs;
    funcs.startRecording();

    // Calibration parameters as inputs
    idouble id_q_lv_init = params.q_lv_init;
    idouble id_C_aortic = params.C_aortic;
    idouble id_E_lv_A = params.E_lv_A;
    idouble id_E_lv_B = params.E_lv_B;

    auto a_qlv = id_q_lv_init.markAsInput();
    auto a_cao = id_C_aortic.markAsInput();
    auto a_elva = id_E_lv_A.markAsInput();
    auto a_elvb = id_E_lv_B.markAsInput();

    // Initial states
    idouble st[N_STATES];
    for (int i = 0; i < N_STATES; i++) st[i] = 0.0;
    st[QC_PVN] = params.q_C_init_pvn;
    st[Q_RA] = params.q_ra_init;
    st[Q_RV] = params.q_rv_init;
    st[Q_LA] = params.q_la_init;
    st[Q_LV] = id_q_lv_init;  // calibration param
    st[QC_VENOUS] = params.q_C_init_venous;
    st[Q_SYS] = params.q_init_sys;

    // Semi-implicit Euler integration
    idouble rates[N_STATES], lam[N_STATES];

    for (int step = 0; step < total_steps; step++) {
        compute_rates_and_damping(st, rates, lam, params,
            id_q_lv_init, id_C_aortic, id_E_lv_A, id_E_lv_B);

        for (int i = 0; i < N_STATES; i++)
            st[i] = st[i] + dt * rates[i] / (1.0 + dt * lam[i]);

        // Clamp zeta to [0, 1] using iIf
        for (int z : {ZETA_TRV, ZETA_PUV, ZETA_MIV, ZETA_AOV}) {
            st[z] = iIf(st[z] >= 0.0, st[z], idouble(0.0));
            st[z] = iIf(st[z] <= 1.0, st[z], idouble(1.0));
        }
    }

    // Cost: LV volume squared (placeholder)
    idouble cost = st[Q_LV] * st[Q_LV];
    auto r_cost = cost.markAsOutput();

    funcs.stopRecording();
    auto t1 = std::chrono::high_resolution_clock::now();
    printf("Compiled: %.1fs, %lu blocks, fwd %.1f MB, rev %.1f MB, ws %.1f MB\n\n",
           std::chrono::duration<double>(t1 - t0).count(),
           (unsigned long)funcs.getNumCodeBlocks(),
           funcs.getCodeSizeFwd() / 1e6, funcs.getCodeSizeRev() / 1e6,
           funcs.getWorkSpaceMemUse() / 1e6);

    // Benchmark — create workspaces for all threads
    std::vector<std::shared_ptr<aadc::AADCWorkSpace<mmType>>> wss(n_threads);
    for (int t = 0; t < n_threads; t++)
        wss[t] = std::shared_ptr<aadc::AADCWorkSpace<mmType>>(funcs.createWorkSpace());

    aadc::AADCArgument args[4] = {a_qlv, a_cao, a_elva, a_elvb};
    double pvals[4] = {params.q_lv_init, params.C_aortic, params.E_lv_A, params.E_lv_B};
    const char* pnames[] = {"dC/dq_lv_init", "dC/dC_aortic", "dC/dE_lv_A", "dC/dE_lv_B"};

    // AVX-filled parameter values
    mmType p_avx[4];
    for (int i = 0; i < 4; i++) p_avx[i] = aadc::mmSetConst<mmType>(pvals[i]);

    // Forward (1 thread, 1 lane)
    auto bt0 = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < n_iters; b++) {
        for (int i = 0; i < 4; i++) wss[0]->setVal(args[i], mm_lane0(pvals[i]));
        funcs.forward(*wss[0]);
    }
    auto bt1 = std::chrono::high_resolution_clock::now();
    double fwd_ms = std::chrono::duration<double, std::milli>(bt1 - bt0).count() / n_iters;
    double cval = ((double*)&wss[0]->val(r_cost))[0];
    printf("Forward (1 thr, 1 lane):  %.3f ms  cost=%.6e\n", fwd_ms, cval);

    // AD (1 thread, 1 lane)
    bt0 = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < n_iters; b++) {
        for (int i = 0; i < 4; i++) wss[0]->setVal(args[i], mm_lane0(pvals[i]));
        funcs.forward(*wss[0]);
        wss[0]->resetDiff();
        wss[0]->setDiff(r_cost, mm_lane0(1.0));
        funcs.reverse(*wss[0]);
    }
    bt1 = std::chrono::high_resolution_clock::now();
    double ad1_ms = std::chrono::duration<double, std::milli>(bt1 - bt0).count() / n_iters;
    printf("AD (1 thr, 1 lane):       %.3f ms  ratio: %.1fx fwd\n", ad1_ms, ad1_ms / fwd_ms);

    // AD (1 thread, AVX lanes)
    bt0 = std::chrono::high_resolution_clock::now();
    for (int b = 0; b < n_iters; b++) {
        for (int i = 0; i < 4; i++) wss[0]->setVal(args[i], p_avx[i]);
        funcs.forward(*wss[0]);
        wss[0]->resetDiff();
        wss[0]->setDiff(r_cost, aadc::mmSetConst<mmType>(1.0));
        funcs.reverse(*wss[0]);
    }
    bt1 = std::chrono::high_resolution_clock::now();
    double ad_avx_ms = std::chrono::duration<double, std::milli>(bt1 - bt0).count() / n_iters;
    int avx_evals = AVX_BATCH;
    printf("AD (1 thr, %d AVX):       %.3f ms  (%.3f ms/eval, %.0f evals/s)\n",
           AVX_BATCH, ad_avx_ms, ad_avx_ms / avx_evals,
           1000.0 * avx_evals / ad_avx_ms);

    // AD multi-thread × AVX
    if (n_threads > 1) {
        bt0 = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < n_iters; b++) {
            std::vector<std::thread> threads;
            for (int t = 1; t < n_threads; t++) {
                threads.emplace_back([&, t]() {
                    auto& ws = *wss[t];
                    for (int i = 0; i < 4; i++) ws.setVal(args[i], p_avx[i]);
                    funcs.forward(ws);
                    ws.resetDiff();
                    ws.setDiff(r_cost, aadc::mmSetConst<mmType>(1.0));
                    funcs.reverse(ws);
                });
            }
            {
                auto& ws = *wss[0];
                for (int i = 0; i < 4; i++) ws.setVal(args[i], p_avx[i]);
                funcs.forward(ws);
                ws.resetDiff();
                ws.setDiff(r_cost, aadc::mmSetConst<mmType>(1.0));
                funcs.reverse(ws);
            }
            for (auto& t : threads) t.join();
        }
        bt1 = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(bt1 - bt0).count();
        int total_evals = n_iters * n_threads * AVX_BATCH;
        printf("AD (%d thr × %d AVX = %d): %.3f ms total, %.4f ms/eval  (%.0f evals/s)\n",
               n_threads, AVX_BATCH, n_threads * AVX_BATCH,
               total_ms, total_ms / total_evals, 1000.0 * total_evals / total_ms);
    }

    // Gradient values
    printf("\nGradient: [");
    for (int i = 0; i < 4; i++) {
        // Re-run single-lane AD for clean gradient
        for (int j = 0; j < 4; j++) wss[0]->setVal(args[j], mm_lane0(pvals[j]));
        funcs.forward(*wss[0]);
    }
    wss[0]->resetDiff();
    wss[0]->setDiff(r_cost, mm_lane0(1.0));
    funcs.reverse(*wss[0]);
    for (int i = 0; i < 4; i++) {
        double g = ((double*)&wss[0]->diff(args[i]))[0];
        printf("%s=%.6e%s", pnames[i], g, i < 3 ? ", " : "");
    }
    printf("]\n");

    // FD verification
    printf("\nFD gradient verification:\n");
    double eps_fd = 1e-6;
    for (int ip = 0; ip < 4; ip++) {
        double p_up[4], p_dn[4];
        for (int j = 0; j < 4; j++) { p_up[j] = pvals[j]; p_dn[j] = pvals[j]; }
        double h = pvals[ip] * eps_fd;
        if (h == 0) h = eps_fd;
        p_up[ip] += h;
        p_dn[ip] -= h;

        for (int j = 0; j < 4; j++) wss[0]->setVal(args[j], mm_lane0(p_up[j]));
        funcs.forward(*wss[0]);
        double c_up = ((double*)&wss[0]->val(r_cost))[0];

        for (int j = 0; j < 4; j++) wss[0]->setVal(args[j], mm_lane0(p_dn[j]));
        funcs.forward(*wss[0]);
        double c_dn = ((double*)&wss[0]->val(r_cost))[0];

        double fd = (c_up - c_dn) / (2.0 * h);
        double ad = ((double*)&wss[0]->diff(args[ip]))[0];
        double ratio = (fd != 0.0) ? ad / fd : (ad == 0.0 ? 1.0 : 1e99);
        printf("  %s: AD=%.6e FD=%.6e ratio=%.6f\n", pnames[ip], ad, fd, ratio);
    }

    // Hessian via FD of AD gradient
    printf("\nHessian (FD of AD gradient, eps=1e-5):\n");
    double eps_h = 1e-5;
    double hess[4][4];
    auto bt_h0 = std::chrono::high_resolution_clock::now();

    for (int ip = 0; ip < 4; ip++) {
        double p_up[4], p_dn[4];
        for (int j = 0; j < 4; j++) { p_up[j] = pvals[j]; p_dn[j] = pvals[j]; }
        double h = pvals[ip] * eps_h;
        if (h == 0) h = eps_h;
        p_up[ip] += h;
        p_dn[ip] -= h;

        // Gradient at p + h*e_ip
        for (int j = 0; j < 4; j++) wss[0]->setVal(args[j], mm_lane0(p_up[j]));
        funcs.forward(*wss[0]);
        wss[0]->resetDiff();
        wss[0]->setDiff(r_cost, mm_lane0(1.0));
        funcs.reverse(*wss[0]);
        double grad_up[4];
        for (int j = 0; j < 4; j++) grad_up[j] = ((double*)&wss[0]->diff(args[j]))[0];

        // Gradient at p - h*e_ip
        for (int j = 0; j < 4; j++) wss[0]->setVal(args[j], mm_lane0(p_dn[j]));
        funcs.forward(*wss[0]);
        wss[0]->resetDiff();
        wss[0]->setDiff(r_cost, mm_lane0(1.0));
        funcs.reverse(*wss[0]);
        double grad_dn[4];
        for (int j = 0; j < 4; j++) grad_dn[j] = ((double*)&wss[0]->diff(args[j]))[0];

        // H[ip][j] = (grad_up[j] - grad_dn[j]) / (2h)
        for (int j = 0; j < 4; j++)
            hess[ip][j] = (grad_up[j] - grad_dn[j]) / (2.0 * h);
    }

    auto bt_h1 = std::chrono::high_resolution_clock::now();
    double hess_ms = std::chrono::duration<double, std::milli>(bt_h1 - bt_h0).count();

    // Print Hessian matrix
    printf("  %16s %16s %16s %16s\n", "q_lv_init", "C_aortic", "E_lv_A", "E_lv_B");
    for (int i = 0; i < 4; i++) {
        printf("  ");
        for (int j = 0; j < 4; j++)
            printf("%16.4e ", hess[i][j]);
        printf("  ← %s\n", pnames[i]);
    }

    // Symmetry check
    double max_asym = 0;
    for (int i = 0; i < 4; i++)
        for (int j = i+1; j < 4; j++) {
            double avg = 0.5 * (fabs(hess[i][j]) + fabs(hess[j][i]));
            if (avg > 0) max_asym = std::max(max_asym, fabs(hess[i][j] - hess[j][i]) / avg);
        }
    printf("  Symmetry: max |H[i,j]-H[j,i]|/avg = %.2e\n", max_asym);
    printf("  Time: %.2f ms (= 8 AD evaluations)\n", hess_ms);

    printf("\nCasADI reference: CRASHES on this model (conditional valve logic)\n");
    printf("  → Hessian impossible with CasADI (cannot even compute gradient)\n");
    return 0;
}
