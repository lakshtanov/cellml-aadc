/*
 * 3-Compartment Cardiovascular Model — CVODES + AADC
 *
 * Forward integration: CVODES (BDF, implicit, adaptive, stiff-safe)
 * Jacobian: AADC (exact, no finite differences)
 *
 * This is the "Approach 2" from the Auckland collaboration:
 * keep CVODES for the forward solve, use AADC for the Jacobian.
 *
 * Build:
 *   g++ -O2 -mavx2 -std=c++17 \
 *     -I$AADC/include -I$SUNDIALS/include \
 *     cvs3_cvodes.cpp \
 *     -L$AADC/lib -laadc-avx2 \
 *     -L$SUNDIALS/lib64 -lsundials_cvodes -lsundials_nvecserial \
 *     -lpthread -Wl,-rpath,$AADC/lib:$SUNDIALS/lib64 \
 *     -o cvs3_cvodes
 */
#include <cstdio>
#include <cmath>
#include <cstring>
#include <chrono>
#include <vector>

// AADC
#include <aadc/aadc.h>

// SUNDIALS / CVODES
#include <cvodes/cvodes.h>
#include <nvector/nvector_serial.h>
#include <sunmatrix/sunmatrix_dense.h>
#include <sunlinsol/sunlinsol_dense.h>
#include <sundials/sundials_types.h>

#ifdef AADC_512
typedef __m512d mmType;
#else
typedef __m256d mmType;
#endif

// ========== Model constants ==========
static const int N = 27;  // states

// Same parameters as cvs3_aadc.cpp
struct CVSParams {
    double R_pvn=1333000, C_pvn=6.0015e-9, I_pvn=1e-6;
    double q_C_init_pvn=1e-4, q_us_0_pvn=0, delta_q_us_pvn=0, u_ext_pvn=0, delta_C_pvn=0;
    double R_par=10664000, C_par=3.09077e-10, I_par=1e-6;
    double u_0_par=1463, u_ext_par=0;
    double R_aortic=1e6, C_aortic=1.2028e-8, I_aortic=10000;
    double u_0_aortic=13300, u_ext_aortic=0;
    double R_t_sys=1.1e8, C_t_sys=1e-7, q_us_sys=0.00245, q_init_sys=0.00245, u_ext_sys=0;
    double R_venous=1114600, C_venous=1e-6, I_venous=0.01;
    double q_C_init_venous=0.0013, q_us_0_venous=0, delta_q_us_venous=0, u_ext_venous=0, delta_C_venous=0;
    double rho=1050, T_period=1;
    double q_ra_us=4e-6, q_rv_us=1e-5, q_la_us=4e-6, q_lv_us=5e-6;
    double q_ra_init=4e-6, q_rv_init=1e-5, q_la_init=4e-6, q_lv_init=0.002;
    double t_ac=0.17, t_ar=0.17, t_astart=0.8;
    double t_vc=0.30, t_vr=0.15, t_vstart=0.0;
    double E_ra_A=7998000, E_ra_B=9331000;
    double E_rv_A=73315000, E_rv_B=6665000;
    double E_la_A=9331000, E_la_B=11997000;
    double E_lv_A=366575000, E_lv_B=10664000;
    double k_vo_trv=0.3, k_vo_puv=0.2, k_vo_miv=0.3, k_vo_aov=0.04;
    double k_vc_trv=0.4, k_vc_puv=0.2, k_vc_miv=0.4, k_vc_aov=0.04;
    double m_rg_trv=0, m_rg_puv=0, m_rg_miv=0, m_rg_aov=0;
    double m_st_trv=1, m_st_puv=1, m_st_miv=1, m_st_aov=1;
    double l_eff=0.01;
    double a_nn_trv=9e-4, a_nn_puv=4e-4, a_nn_miv=6e-4, a_nn_aov=3.14e-4;
    double eps_1=0.07, eps_2=0.02, eps_m4=1e-14, eps_m2=1e-14;
    double I_t_sys=1e-6;
};

static CVSParams gp;  // global params

// State indices (same as cvs3_aadc.cpp)
enum S {
    V_PVN=0, V_PAR, QC_PVN, QC_PAR, V_PUV,
    CHI_A, CHI_V, S_HEART, ZETA_TRV, ZETA_PUV,
    ZETA_MIV, ZETA_AOV, V_TRV, V_MIV, V_AOV,
    Q_RA, Q_RV, Q_LA, Q_LV, V_VENOUS,
    QCD_AORTIC, QC_AORTIC, V_AORTIC, V_SYS, VT_SYS,
    Q_SYS, QC_VENOUS
};

// ========== RHS for CVODES ==========
static int cvs_rhs(realtype t, N_Vector y, N_Vector ydot, void* user_data) {
    const double *st = N_VGetArrayPointer(y);
    double *rates = N_VGetArrayPointer(ydot);
    const double PI = 3.14159265358979;
    const CVSParams &p = gp;

    double r_v_pvn = 0.01/p.C_pvn, r_v_par = 0.01/p.C_par;
    double r_v_aortic = 0.01/p.C_aortic, r_v_sys = 0.01/p.C_t_sys;
    double r_v_venous = 0.01/p.C_venous;

    // Pulmonary venous
    double q_c_pvn = st[QC_PVN] + p.q_us_0_pvn;
    double u_c_pvn = q_c_pvn / p.C_pvn;
    double u_pvn = u_c_pvn + r_v_pvn*(st[V_PAR]-st[V_PVN]);

    double chi_af = st[CHI_A] - floor(st[CHI_A]);
    double chi_af_final = (chi_af <= 0.5) ? chi_af*2.0 : 0.0;
    double e_a = 0.5*(1.0 - cos(2*PI*chi_af_final));
    double u_la = (e_a*p.E_la_A + p.E_la_B)*(st[Q_LA]-p.q_la_us);

    rates[V_PVN] = (u_pvn - u_la - p.R_pvn*st[V_PVN])/p.I_pvn;
    rates[QC_PVN] = st[V_PAR] - st[V_PVN];

    double u_c_par = st[QC_PAR]/p.C_par;
    double u_par = p.u_0_par + u_c_par + r_v_par*(st[V_PUV]-st[V_PAR]);
    rates[V_PAR] = (u_par - u_pvn - p.R_par*st[V_PAR])/p.I_par;
    rates[QC_PAR] = st[V_PUV] - st[V_PAR];

    double mt = st[S_HEART] - floor(st[S_HEART]);
    double t_astart_norm = p.t_astart/p.T_period;
    double t_vstart_norm = p.t_vstart/p.T_period;

    // Atrial activation rate
    double r1a = 0.25/p.t_ac, r2a = 0.25/p.t_ar;
    double r3a = 0.5/(p.T_period - p.t_ac - p.t_ar);
    double trig_a = (mt>=t_astart_norm && mt<=t_astart_norm+p.eps_1 && chi_af<=0.25) ? r1a : 0;
    double ongoing_a = (chi_af>p.eps_2 && chi_af<=0.25) ? r1a : 0;
    double relax_a = (chi_af>=0.25 && chi_af<0.5) ? r2a : 0;
    double diast_a = (chi_af>=0.5) ? r3a : 0;
    rates[CHI_A] = trig_a + ongoing_a + relax_a + diast_a;

    double chi_vf = st[CHI_V] - floor(st[CHI_V]);
    double chi_vf_final = (chi_vf<=0.5) ? chi_vf*2.0 : 0.0;
    double e_v = 0.5*(1.0-cos(2*PI*chi_vf_final));
    double r1v=0.25/p.t_vc, r2v=0.25/p.t_vr;
    double r3v = 0.5/(p.T_period-p.t_vc-p.t_vr);
    double trig_v = (mt>=t_vstart_norm && mt<=t_vstart_norm+p.eps_1 && chi_vf<=0.25) ? r1v : 0;
    double ongoing_v = (chi_vf>p.eps_2 && chi_vf<=0.25) ? r1v : 0;
    double relax_v = (chi_vf>=0.25 && chi_vf<0.5) ? r2v : 0;
    double diast_v = (chi_vf>=0.5) ? r3v : 0;
    rates[CHI_V] = trig_v + ongoing_v + relax_v + diast_v;
    rates[S_HEART] = 1.0/p.T_period;

    double u_rv = (e_v*p.E_rv_A+p.E_rv_B)*(st[Q_RV]-p.q_rv_us);
    double u_ra = (e_a*p.E_ra_A+p.E_ra_B)*(st[Q_RA]-p.q_ra_us);
    double u_lv = (e_v*p.E_lv_A+p.E_lv_B)*(st[Q_LV]-p.q_lv_us);

    double u_c_aortic = st[QC_AORTIC]/(p.C_aortic/2);
    double u_aortic = p.u_0_aortic + u_c_aortic + 2*r_v_aortic*(st[V_AOV]-st[V_AORTIC]);

    // Valve zeta
    auto valve_zeta = [](double u_up, double u_dn, double zeta, double k_vo, double k_vc) {
        return (u_up>=u_dn) ? (1-zeta)*k_vo*(u_up-u_dn) : zeta*k_vc*(u_up-u_dn);
    };
    rates[ZETA_TRV] = valve_zeta(u_ra,u_rv,st[ZETA_TRV],p.k_vo_trv,p.k_vc_trv);
    rates[ZETA_PUV] = valve_zeta(u_rv,u_par,st[ZETA_PUV],p.k_vo_puv,p.k_vc_puv);
    rates[ZETA_MIV] = valve_zeta(u_la,u_lv,st[ZETA_MIV],p.k_vo_miv,p.k_vc_miv);
    rates[ZETA_AOV] = valve_zeta(u_lv,u_aortic,st[ZETA_AOV],p.k_vo_aov,p.k_vc_aov);

    // Valve flows
    auto valve_flow = [&](int zi, double ms, double an, double mr, double u_up, double u_dn) {
        double zeta = (st[zi]>=0) ? st[zi] : 0;
        double a_eff = (ms*an-mr*an)*zeta + mr*an;
        double l = p.rho*p.l_eff/(a_eff+p.eps_m2);
        double b = p.rho/(2*a_eff*a_eff+p.eps_m4);
        double v_abs = (st[zi]>=0) ? fabs(st[zi]) : 0;
        // placeholder — use the actual state for flow
        return 0.0;
    };
    // Direct valve flow rates
    auto do_valve = [&](int vi, int zi, double ms, double an, double mr, double u_up, double u_dn) {
        double zeta = (st[zi]>=0) ? st[zi] : 0;
        double a_eff = (ms*an-mr*an)*zeta + mr*an;
        double l = p.rho*p.l_eff/(a_eff+p.eps_m2);
        double b = p.rho/(2*a_eff*a_eff+p.eps_m4);
        double v = st[vi], v_fabs = fabs(v);
        rates[vi] = (-b*v*v_fabs + u_up - u_dn)/l;
    };
    do_valve(V_TRV, ZETA_TRV, p.m_st_trv, p.a_nn_trv, p.m_rg_trv, u_ra, u_rv);
    do_valve(V_PUV, ZETA_PUV, p.m_st_puv, p.a_nn_puv, p.m_rg_puv, u_rv, u_par);
    do_valve(V_MIV, ZETA_MIV, p.m_st_miv, p.a_nn_miv, p.m_rg_miv, u_la, u_lv);
    do_valve(V_AOV, ZETA_AOV, p.m_st_aov, p.a_nn_aov, p.m_rg_aov, u_lv, u_aortic);

    rates[Q_RA] = st[V_VENOUS] - st[V_TRV];
    rates[Q_RV] = st[V_TRV] - st[V_PUV];
    rates[Q_LA] = st[V_PVN] - st[V_MIV];
    rates[Q_LV] = st[V_MIV] - st[V_AOV];

    double u_c_d = st[QCD_AORTIC]/(p.C_aortic/2);
    double u_d = p.u_0_aortic + u_c_d + 2*r_v_aortic*(st[V_AORTIC]-st[V_SYS]);
    rates[V_AORTIC] = (u_aortic-u_d-p.R_aortic*st[V_AORTIC])/p.I_aortic;
    rates[QC_AORTIC] = st[V_AOV]-st[V_AORTIC];
    rates[QCD_AORTIC] = st[V_AORTIC]-st[V_SYS];

    rates[Q_SYS] = st[V_SYS]-st[VT_SYS];
    double u_c_sys = (st[Q_SYS]-p.q_us_sys)/p.C_t_sys;
    double u_sys = u_c_sys + r_v_sys*(st[V_SYS]-st[VT_SYS]);
    rates[V_SYS] = (u_d-u_sys-st[V_SYS]*p.R_t_sys/2)/p.I_t_sys;

    double q_c_v = st[QC_VENOUS] + p.q_us_0_venous;
    double u_c_v = q_c_v/p.C_venous;
    double u_ven = u_c_v + r_v_venous*(st[VT_SYS]-st[V_VENOUS]);
    rates[VT_SYS] = (u_sys-u_ven-st[VT_SYS]*p.R_t_sys/2)/p.I_t_sys;
    rates[V_VENOUS] = (u_ven-u_ra-p.R_venous*st[V_VENOUS])/p.I_venous;
    rates[QC_VENOUS] = st[VT_SYS]-st[V_VENOUS];

    return 0;
}

// ========== AADC Jacobian for CVODES ==========
// Global AADC objects (recorded once)
static aadc::AADCFunctions<mmType>* g_funcs = nullptr;
static std::shared_ptr<aadc::AADCWorkSpace<mmType>> g_ws;
static std::vector<aadc::AADCArgument> g_a_x;
static std::vector<aadc::AADCResult> g_r_f;

// Record RHS with AADC (once)
void record_aadc_rhs() {
    static aadc::AADCFunctions<mmType> funcs;
    funcs.startRecording();

    idouble id_x[N];
    g_a_x.resize(N);
    for (int i = 0; i < N; i++) {
        id_x[i] = 0.0;
        g_a_x[i] = id_x[i].markAsInput();
    }

    // Compute RHS with idouble (same logic as cvs_rhs but with iIf)
    const double PI = 3.14159265358979;
    const CVSParams &p = gp;
    idouble r_v_pvn = 0.01/p.C_pvn, r_v_par = 0.01/p.C_par;
    idouble r_v_aortic = 0.01/p.C_aortic, r_v_sys = 0.01/p.C_t_sys;
    idouble r_v_venous = 0.01/p.C_venous;

    idouble q_c_pvn = id_x[QC_PVN] + p.q_us_0_pvn;
    idouble u_c_pvn = q_c_pvn / p.C_pvn;
    idouble u_pvn = u_c_pvn + r_v_pvn*(id_x[V_PAR]-id_x[V_PVN]);

    idouble chi_af = id_x[CHI_A] - floor(id_x[CHI_A]);
    idouble chi_af_final = iIf(chi_af <= 0.5, chi_af*2.0, idouble(0.0));
    idouble e_a = 0.5*(1.0 - cos(2*PI*chi_af_final));
    idouble u_la = (e_a*p.E_la_A + p.E_la_B)*(id_x[Q_LA]-p.q_la_us);

    idouble rates_id[N];
    rates_id[V_PVN] = (u_pvn - u_la - p.R_pvn*id_x[V_PVN])/p.I_pvn;
    rates_id[QC_PVN] = id_x[V_PAR] - id_x[V_PVN];

    idouble u_c_par = id_x[QC_PAR]/p.C_par;
    idouble u_par = p.u_0_par + u_c_par + r_v_par*(id_x[V_PUV]-id_x[V_PAR]);
    rates_id[V_PAR] = (u_par - u_pvn - p.R_par*id_x[V_PAR])/p.I_par;
    rates_id[QC_PAR] = id_x[V_PUV] - id_x[V_PAR];

    // Timing (passive floor)
    rates_id[CHI_A] = 0.0; // simplified — timing doesn't affect Jacobian structure
    rates_id[CHI_V] = 0.0;
    rates_id[S_HEART] = 1.0/p.T_period;

    idouble chi_vf = id_x[CHI_V] - floor(id_x[CHI_V]);
    idouble chi_vf_final = iIf(chi_vf <= 0.5, chi_vf*2.0, idouble(0.0));
    idouble e_v = 0.5*(1.0-cos(2*PI*chi_vf_final));

    idouble u_rv = (e_v*p.E_rv_A+p.E_rv_B)*(id_x[Q_RV]-p.q_rv_us);
    idouble u_ra = (e_a*p.E_ra_A+p.E_ra_B)*(id_x[Q_RA]-p.q_ra_us);
    idouble u_lv = (e_v*p.E_lv_A+p.E_lv_B)*(id_x[Q_LV]-p.q_lv_us);

    idouble u_c_aortic = id_x[QC_AORTIC]/(p.C_aortic/2);
    idouble u_aortic = p.u_0_aortic + u_c_aortic + 2.0*r_v_aortic*(id_x[V_AOV]-id_x[V_AORTIC]);

    // Valve zeta with iIf
    auto izeta = [&](int idx, idouble u_up, idouble u_dn, double kvo, double kvc) {
        return iIf(u_up >= u_dn,
            (1.0-id_x[idx])*kvo*(u_up-u_dn),
            id_x[idx]*kvc*(u_up-u_dn));
    };
    rates_id[ZETA_TRV] = izeta(ZETA_TRV,u_ra,u_rv,p.k_vo_trv,p.k_vc_trv);
    rates_id[ZETA_PUV] = izeta(ZETA_PUV,u_rv,u_par,p.k_vo_puv,p.k_vc_puv);
    rates_id[ZETA_MIV] = izeta(ZETA_MIV,u_la,u_lv,p.k_vo_miv,p.k_vc_miv);
    rates_id[ZETA_AOV] = izeta(ZETA_AOV,u_lv,u_aortic,p.k_vo_aov,p.k_vc_aov);

    // Valve flows
    auto ivalve = [&](int vi, int zi, double ms, double an, double mr, idouble u_up, idouble u_dn) {
        idouble zeta = iIf(id_x[zi]>=0.0, id_x[zi], idouble(0.0));
        idouble a_eff = (ms*an-mr*an)*zeta + mr*an;
        idouble l = p.rho*p.l_eff/(a_eff+p.eps_m2);
        idouble b = p.rho/(2.0*a_eff*a_eff+p.eps_m4);
        idouble v = id_x[vi];
        idouble v_fabs = iIf(v>=0.0, v, -v);
        rates_id[vi] = (-b*v*v_fabs + u_up - u_dn)/l;
    };
    ivalve(V_TRV,ZETA_TRV,p.m_st_trv,p.a_nn_trv,p.m_rg_trv,u_ra,u_rv);
    ivalve(V_PUV,ZETA_PUV,p.m_st_puv,p.a_nn_puv,p.m_rg_puv,u_rv,u_par);
    ivalve(V_MIV,ZETA_MIV,p.m_st_miv,p.a_nn_miv,p.m_rg_miv,u_la,u_lv);
    ivalve(V_AOV,ZETA_AOV,p.m_st_aov,p.a_nn_aov,p.m_rg_aov,u_lv,u_aortic);

    rates_id[Q_RA] = id_x[V_VENOUS]-id_x[V_TRV];
    rates_id[Q_RV] = id_x[V_TRV]-id_x[V_PUV];
    rates_id[Q_LA] = id_x[V_PVN]-id_x[V_MIV];
    rates_id[Q_LV] = id_x[V_MIV]-id_x[V_AOV];

    idouble u_c_d = id_x[QCD_AORTIC]/(p.C_aortic/2);
    idouble u_d = p.u_0_aortic + u_c_d + 2.0*r_v_aortic*(id_x[V_AORTIC]-id_x[V_SYS]);
    rates_id[V_AORTIC] = (u_aortic-u_d-p.R_aortic*id_x[V_AORTIC])/p.I_aortic;
    rates_id[QC_AORTIC] = id_x[V_AOV]-id_x[V_AORTIC];
    rates_id[QCD_AORTIC] = id_x[V_AORTIC]-id_x[V_SYS];

    rates_id[Q_SYS] = id_x[V_SYS]-id_x[VT_SYS];
    idouble u_c_sys = (id_x[Q_SYS]-p.q_us_sys)/p.C_t_sys;
    idouble u_sys = u_c_sys + r_v_sys*(id_x[V_SYS]-id_x[VT_SYS]);
    rates_id[V_SYS] = (u_d-u_sys-id_x[V_SYS]*p.R_t_sys/2)/p.I_t_sys;

    idouble q_c_v = id_x[QC_VENOUS];
    idouble u_c_v = q_c_v/p.C_venous;
    idouble u_ven = u_c_v + r_v_venous*(id_x[VT_SYS]-id_x[V_VENOUS]);
    rates_id[VT_SYS] = (u_sys-u_ven-id_x[VT_SYS]*p.R_t_sys/2)/p.I_t_sys;
    rates_id[V_VENOUS] = (u_ven-u_ra-p.R_venous*id_x[V_VENOUS])/p.I_venous;
    rates_id[QC_VENOUS] = id_x[VT_SYS]-id_x[V_VENOUS];

    g_r_f.resize(N);
    for (int i = 0; i < N; i++)
        g_r_f[i] = rates_id[i].markAsOutput();

    funcs.stopRecording();
    g_funcs = &funcs;
    g_ws = funcs.createWorkSpace();

    printf("AADC RHS kernel recorded for Jacobian\n");
}

// CVODES dense Jacobian callback using AADC
static int cvs_jac(realtype t, N_Vector y, N_Vector fy, SUNMatrix J,
                    void* user_data, N_Vector tmp1, N_Vector tmp2, N_Vector tmp3) {
    const double *st = N_VGetArrayPointer(y);

    // Set state inputs
    for (int i = 0; i < N; i++) {
        auto& v = g_ws->val(g_a_x[i]);
        for (int k = 0; k < (int)(sizeof(mmType)/sizeof(double)); k++)
            ((double*)&v)[k] = st[i];
    }

    // For each output, seed = 1 and reverse to get row of Jacobian
    for (int i = 0; i < N; i++) {
        g_funcs->forward(*g_ws);
        g_ws->resetDiff();
        auto& d = g_ws->diff(g_r_f[i]);
        for (int k = 0; k < (int)(sizeof(mmType)/sizeof(double)); k++)
            ((double*)&d)[k] = 1.0;
        g_funcs->reverse(*g_ws);

        for (int j = 0; j < N; j++) {
            double val = ((double*)&g_ws->diff(g_a_x[j]))[0];
            SM_ELEMENT_D(J, i, j) = val;
        }
    }

    return 0;
}

int main() {
    printf("3-Compartment CVS: CVODES (BDF) + AADC Jacobian\n");
    printf("================================================\n\n");

    // Record AADC kernel
    record_aadc_rhs();

    // CVODES setup
    SUNContext ctx;
    SUNContext_Create(NULL, &ctx);

    N_Vector y = N_VNew_Serial(N, ctx);
    double *yd = N_VGetArrayPointer(y);
    memset(yd, 0, N*sizeof(double));
    yd[QC_PVN] = gp.q_C_init_pvn;
    yd[Q_RA] = gp.q_ra_init;
    yd[Q_RV] = gp.q_rv_init;
    yd[Q_LA] = gp.q_la_init;
    yd[Q_LV] = gp.q_lv_init;
    yd[QC_VENOUS] = gp.q_C_init_venous;
    yd[Q_SYS] = gp.q_init_sys;

    void *cvode = CVodeCreate(CV_BDF, ctx);
    CVodeInit(cvode, cvs_rhs, 0.0, y);
    CVodeSStolerances(cvode, 1e-8, 1e-10);
    CVodeSetMaxNumSteps(cvode, 500000);
    CVodeSetMaxStep(cvode, 0.01);

    SUNMatrix A = SUNDenseMatrix(N, N, ctx);
    SUNLinearSolver LS = SUNLinSol_Dense(y, A, ctx);
    CVodeSetLinearSolver(cvode, LS, A);
    CVodeSetJacFn(cvode, cvs_jac);

    // Integrate
    printf("Integrating t=0 to t=22 (BDF + AADC Jacobian)...\n");
    auto t0 = std::chrono::high_resolution_clock::now();

    realtype t_out;
    int flag = CVode(cvode, 22.0, y, &t_out, CV_NORMAL);

    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    if (flag >= 0) {
        printf("SUCCESS: t=%.1f, elapsed=%.2fs\n", t_out, elapsed);
        printf("  Q_LV = %.6e\n", yd[Q_LV]);

        long nst, nfe, nje;
        CVodeGetNumSteps(cvode, &nst);
        CVodeGetNumRhsEvals(cvode, &nfe);
        CVodeGetNumJacEvals(cvode, &nje);
        printf("  Steps: %ld, RHS evals: %ld, Jacobian evals: %ld\n", nst, nfe, nje);
    } else {
        printf("FAILED: flag=%d at t=%.4f\n", flag, t_out);
    }

    // Cleanup
    N_VDestroy(y);
    SUNMatDestroy(A);
    SUNLinSolFree(LS);
    CVodeFree(&cvode);
    SUNContext_Free(&ctx);
    g_ws.reset();

    return 0;
}
