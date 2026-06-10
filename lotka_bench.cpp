/* Lotka-Volterra AADC benchmark driver.
 * Usage: ./lotka_bench [options]
 *   --steps N       ODE time steps (default: 500, T = N*0.01)
 *   --threads N     worker threads (default: 1)
 *   --iters N       benchmark iterations (default: 500)
 *   --alpha F       test alpha (default: 4.5)
 *   --beta F        test beta (default: 0.25)
 *   --delta F       test delta (default: 0.18)
 *   --gamma F       test gamma (default: 3.2)
 *   --true_alpha F  true alpha for obs data (default: 5.0)
 *   --true_beta F   true beta (default: 0.2)
 *   --true_delta F  true delta (default: 0.2)
 *   --true_gamma F  true gamma (default: 3.0)
 *   --x0 F          initial prey (default: 20.0)
 *   --y0 F          initial predator (default: 10.0)
 *   --dt F          time step (default: 0.01)
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

void simulate_double(double a,double b,double d,double g,
    double x0,double y0,double dt,int N,
    std::vector<double>&tx,std::vector<double>&ty){
    tx.resize(N+1);ty.resize(N+1);tx[0]=x0;ty[0]=y0;
    double x=x0,y=y0;
    for(int i=0;i<N;i++){
        double k1x=a*x-b*x*y,k1y=d*x*y-g*y;
        double x1=x+.5*dt*k1x,y1=y+.5*dt*k1y;
        double k2x=a*x1-b*x1*y1,k2y=d*x1*y1-g*y1;
        double x2=x+.5*dt*k2x,y2=y+.5*dt*k2y;
        double k3x=a*x2-b*x2*y2,k3y=d*x2*y2-g*y2;
        double x3=x+dt*k3x,y3=y+dt*k3y;
        double k4x=a*x3-b*x3*y3,k4y=d*x3*y3-g*y3;
        x+=dt/6*(k1x+2*k2x+2*k3x+k4x);
        y+=dt/6*(k1y+2*k2y+2*k3y+k4y);
        tx[i+1]=x;ty[i+1]=y;}}

int main(int argc,char*argv[]){
    setbuf(stdout,NULL);
    // Defaults
    int N_steps=500, n_threads=1, n_iters=500;
    double dt=0.01, x0=20.0, y0=10.0;
    double true_p[4]={5.0,0.2,0.2,3.0};
    double test_p[4]={4.5,0.25,0.18,3.2};

    // Parse CLI
    for(int i=1;i<argc;i++){
        std::string a=argv[i];
        if(a=="--steps"&&i+1<argc) N_steps=atoi(argv[++i]);
        else if(a=="--threads"&&i+1<argc) n_threads=atoi(argv[++i]);
        else if(a=="--iters"&&i+1<argc) n_iters=atoi(argv[++i]);
        else if(a=="--dt"&&i+1<argc) dt=atof(argv[++i]);
        else if(a=="--x0"&&i+1<argc) x0=atof(argv[++i]);
        else if(a=="--y0"&&i+1<argc) y0=atof(argv[++i]);
        else if(a=="--alpha"&&i+1<argc) test_p[0]=atof(argv[++i]);
        else if(a=="--beta"&&i+1<argc) test_p[1]=atof(argv[++i]);
        else if(a=="--delta"&&i+1<argc) test_p[2]=atof(argv[++i]);
        else if(a=="--gamma"&&i+1<argc) test_p[3]=atof(argv[++i]);
        else if(a=="--true_alpha"&&i+1<argc) true_p[0]=atof(argv[++i]);
        else if(a=="--true_beta"&&i+1<argc) true_p[1]=atof(argv[++i]);
        else if(a=="--true_delta"&&i+1<argc) true_p[2]=atof(argv[++i]);
        else if(a=="--true_gamma"&&i+1<argc) true_p[3]=atof(argv[++i]);
        else if(a=="--help"){
            printf("Usage: %s [--steps N] [--threads N] [--iters N] [--dt F]\n"
                   "  [--alpha F] [--beta F] [--delta F] [--gamma F]\n"
                   "  [--true_alpha F] [--true_beta F] [--true_delta F] [--true_gamma F]\n"
                   "  [--x0 F] [--y0 F]\n",argv[0]);
            return 0;}}

    const int AVX_BATCH=aadc::mmSize<mmType>();
    double T=N_steps*dt;

    printf("Lotka-Volterra AADC Benchmark\n");
    printf("  ODE: dx/dt = alpha*x - beta*x*y,  dy/dt = delta*x*y - gamma*y\n");
    printf("  Steps: %d,  dt: %.4f,  T: %.2f\n",N_steps,dt,T);
    printf("  True params:  [%.3f, %.3f, %.3f, %.3f]\n",true_p[0],true_p[1],true_p[2],true_p[3]);
    printf("  Test params:  [%.3f, %.3f, %.3f, %.3f]\n",test_p[0],test_p[1],test_p[2],test_p[3]);
    printf("  x0: %.1f,  y0: %.1f\n",x0,y0);
    printf("  Threads: %d,  AVX batch: %d,  Bench iters: %d\n\n",n_threads,AVX_BATCH,n_iters);

    // Generate observed data
    std::vector<double>obs_x,obs_y;
    simulate_double(true_p[0],true_p[1],true_p[2],true_p[3],x0,y0,dt,N_steps,obs_x,obs_y);

    // Record AADC kernel
    printf("Recording...\n");
    auto t0=std::chrono::high_resolution_clock::now();
    aadc::AADCFunctions<mmType>funcs;
    funcs.startRecording();

    idouble p[4]; aadc::AADCArgument pa[4];
    for(int i=0;i<4;i++){p[i]=test_p[i];pa[i]=p[i].markAsInput();}

    idouble x=x0,y=y0,cost=0;
    for(int i=0;i<N_steps;i++){
        idouble k1x=p[0]*x-p[1]*x*y, k1y=p[2]*x*y-p[3]*y;
        idouble x1=x+.5*dt*k1x,y1=y+.5*dt*k1y;
        idouble k2x=p[0]*x1-p[1]*x1*y1,k2y=p[2]*x1*y1-p[3]*y1;
        idouble x2=x+.5*dt*k2x,y2=y+.5*dt*k2y;
        idouble k3x=p[0]*x2-p[1]*x2*y2,k3y=p[2]*x2*y2-p[3]*y2;
        idouble x3=x+dt*k3x,y3=y+dt*k3y;
        idouble k4x=p[0]*x3-p[1]*x3*y3,k4y=p[2]*x3*y3-p[3]*y3;
        x+=dt/6*(k1x+2.0*k2x+2.0*k3x+k4x);
        y+=dt/6*(k1y+2.0*k2y+2.0*k3y+k4y);
        idouble dx=x-obs_x[i+1],dy=y-obs_y[i+1];
        cost+=dx*dx+dy*dy;}
    cost=cost/(double)(N_steps*2);
    auto r_cost=cost.markAsOutput();
    funcs.stopRecording();
    auto t1=std::chrono::high_resolution_clock::now();
    printf("Compiled: %.2fs, %lu blocks, fwd %.1f MB, rev %.1f MB, ws %.1f MB\n\n",
        std::chrono::duration<double>(t1-t0).count(),
        (unsigned long)funcs.getNumCodeBlocks(),
        funcs.getCodeSizeFwd()/1e6,funcs.getCodeSizeRev()/1e6,
        funcs.getWorkSpaceMemUse()/1e6);

    // Create workspaces
    std::vector<std::shared_ptr<aadc::AADCWorkSpace<mmType>>>wss(n_threads);
    for(int t=0;t<n_threads;t++)
        wss[t]=std::shared_ptr<aadc::AADCWorkSpace<mmType>>(funcs.createWorkSpace());

    mmType p_avx[4];
    for(int i=0;i<4;i++){
        double*vp=(double*)&p_avx[i];
        for(int a=0;a<AVX_BATCH;a++) vp[a]=test_p[i]+0.001*a;}

    // ========== Forward only ==========
    {
        auto&ws=*wss[0];
        auto bt0=std::chrono::high_resolution_clock::now();
        for(int b=0;b<n_iters;b++){
            for(int i=0;i<4;i++) ws.setVal(pa[i],mm_lane0(test_p[i]));
            funcs.forward(ws);}
        auto bt1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(bt1-bt0).count()/n_iters;
        double cv=((double*)&ws.val(r_cost))[0];
        printf("Forward (1 thr, 1 lane):  %.4f ms  cost=%.4f\n",ms,cv);
    }

    // ========== AD single thread, single lane ==========
    {
        auto&ws=*wss[0];
        auto bt0=std::chrono::high_resolution_clock::now();
        for(int b=0;b<n_iters;b++){
            for(int i=0;i<4;i++) ws.setVal(pa[i],mm_lane0(test_p[i]));
            funcs.forward(ws);
            ws.resetDiff();
            ws.setDiff(r_cost,mm_lane0(1.0));
            funcs.reverse(ws);}
        auto bt1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(bt1-bt0).count()/n_iters;
        double g[4];for(int i=0;i<4;i++)g[i]=((double*)&ws.diff(pa[i]))[0];
        printf("AD (1 thr, 1 lane):       %.4f ms  grad=[%.4f, %.4f, %.4f, %.4f]\n",
            ms,g[0],g[1],g[2],g[3]);
    }

    // ========== AD single thread, all AVX lanes ==========
    {
        auto&ws=*wss[0];
        auto bt0=std::chrono::high_resolution_clock::now();
        for(int b=0;b<n_iters;b++){
            for(int i=0;i<4;i++) ws.setVal(pa[i],p_avx[i]);
            funcs.forward(ws);
            ws.resetDiff();
            ws.setDiff(r_cost,aadc::mmSetConst<mmType>(1.0));
            funcs.reverse(ws);}
        auto bt1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(bt1-bt0).count()/n_iters;
        printf("AD (1 thr, %d AVX):       %.4f ms  = %.4f ms/eval  (%.0f evals/s)\n",
            AVX_BATCH,ms,ms/AVX_BATCH,1000.0*AVX_BATCH/ms);
    }

    // ========== AD multi-thread × AVX ==========
    if(n_threads>1){
        auto bt0=std::chrono::high_resolution_clock::now();
        for(int b=0;b<n_iters;b++){
            std::vector<std::thread>threads;
            for(int t=1;t<n_threads;t++){
                threads.emplace_back([&,t](){
                    auto&ws=*wss[t];
                    for(int i=0;i<4;i++) ws.setVal(pa[i],p_avx[i]);
                    funcs.forward(ws);ws.resetDiff();
                    ws.setDiff(r_cost,aadc::mmSetConst<mmType>(1.0));
                    funcs.reverse(ws);});}
            {auto&ws=*wss[0];
                for(int i=0;i<4;i++) ws.setVal(pa[i],p_avx[i]);
                funcs.forward(ws);ws.resetDiff();
                ws.setDiff(r_cost,aadc::mmSetConst<mmType>(1.0));
                funcs.reverse(ws);}
            for(auto&t:threads)t.join();}
        auto bt1=std::chrono::high_resolution_clock::now();
        double total=std::chrono::duration<double,std::milli>(bt1-bt0).count();
        int total_evals=n_iters*n_threads*AVX_BATCH;
        printf("AD (%d thr, %d AVX=%d):   %.1f ms total, %.4f ms/eval  (%.0f evals/s)\n",
            n_threads,AVX_BATCH,n_threads*AVX_BATCH,
            total,total/total_evals,1000.0*total_evals/total);
    }

    // ========== FD gradient ==========
    {
        auto&ws=*wss[0];
        double eps=1e-7;
        auto bt0=std::chrono::high_resolution_clock::now();
        double g_fd[4];
        for(int b=0;b<n_iters;b++){
            for(int i=0;i<4;i++) ws.setVal(pa[i],mm_lane0(test_p[i]));
            funcs.forward(ws);
            double c0=((double*)&ws.val(r_cost))[0];
            for(int i=0;i<4;i++){
                ws.setVal(pa[i],mm_lane0(test_p[i]+eps));
                funcs.forward(ws);
                g_fd[i]=(((double*)&ws.val(r_cost))[0]-c0)/eps;
                ws.setVal(pa[i],mm_lane0(test_p[i]));}}
        auto bt1=std::chrono::high_resolution_clock::now();
        double ms=std::chrono::duration<double,std::milli>(bt1-bt0).count()/n_iters;
        printf("FD (4 params):            %.4f ms  grad=[%.4f, %.4f, %.4f, %.4f]\n",
            ms,g_fd[0],g_fd[1],g_fd[2],g_fd[3]);
    }

    printf("\nCasADI reference: forward=36.1ms, AD=138.9ms (Python, CVODES, 1 thread)\n");
    return 0;
}
