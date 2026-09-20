// Native CUDA regression tests; no Python/PyTorch dependency.
#include "checkpoint.h"
#include <functional>
#include <numeric>
#include <random>

static void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
template<class T> static std::vector<T> read_gpu(const T* p, size_t n) {
    std::vector<T> v(n); CUDA_CHECK(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost)); return v;
}
template<class T> static T* upload(DeviceMemory& memory, const std::vector<T>& v) {
    T* p; memory.allocate(p, v.size() * sizeof(T));
    CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice)); return p;
}
static void near(double got, double ref, double atol, double rtol, const char* what) {
    if (!std::isfinite(got) || !std::isfinite(ref) || fabs(got-ref) > atol + rtol * fabs(ref)) {
        std::fprintf(stderr, "%s: got %.9g expected %.9g\n", what, got, ref);
        throw std::runtime_error(what);
    }
}
static void test_cell() {
    constexpr int B=2,T=5,D=3,N=B*T*D;
    std::vector<bf16> x(N), grad(N);
    for (int i=0;i<N;++i) { x[i]=f2bf(sinf(i*.71f)); grad[i]=f2bf(cosf(i*.37f)); }
    std::vector<float> initial{.3f,-.5f,.8f,-.4f,.7f,.2f}, decay{-.4f,.5f,2.f}, gate{.7f,-.3f,.4f};
    DeviceMemory memory;
    auto X=upload(memory,x), G=upload(memory,grad);
    auto C=upload(memory,initial), A=upload(memory,decay), Q=upload(memory,gate);
    float *S,*dX,*dA,*dQ;
    memory.allocate(S,N*4);memory.allocate(dX,N*4);memory.allocate(dA,D*4);memory.allocate(dQ,D*4);
    for (bool gated : {false,true}) {
        state_forward(X,S,A,C,B,T,D,gated?Q:nullptr);
        cell_backward(G,S,A,dX,dA,B,T,D,X,C,gated?Q:nullptr,dQ);
        auto state=read_gpu(S,N), dx=read_gpu(dX,N), da=read_gpu(dA,D), dq=read_gpu(dQ,D);
        auto objective=[&](const std::vector<double>& xx,const std::vector<double>& aa,const std::vector<double>& qq) {
            double result=0;
            for(int b=0;b<B;++b)for(int d=0;d<D;++d){
                double st=initial[b*D+d];
                for(int t=0;t<T;++t){int i=(b*T+t)*D+d;
                    double a=1/(1+exp(-aa[d]-(gated?qq[d]*xx[i]:0)));
                    st=a*st+(1-a)*xx[i];result+=st*bf2f(grad[i]);
                }
            }return result;
        };
        std::vector<double> xx(N),aa(decay.begin(),decay.end()),qq(gate.begin(),gate.end());
        for(int i=0;i<N;++i)xx[i]=bf2f(x[i]);
        for(int b=0;b<B;++b)for(int d=0;d<D;++d){double st=initial[b*D+d];
            for(int t=0;t<T;++t){int i=(b*T+t)*D+d;double a=1/(1+exp(-aa[d]-(gated?qq[d]*xx[i]:0)));
                st=a*st+(1-a)*xx[i];near(state[i],st,2e-6,2e-6,"cell forward");}}
        auto diff=[&](std::vector<double>& values,int i){double old=values[i],eps=1e-4;
            values[i]=old+eps;double plus=objective(xx,aa,qq);values[i]=old-eps;double minus=objective(xx,aa,qq);
            values[i]=old;return(plus-minus)/(2*eps);};
        for(int i=0;i<N;++i)near(dx[i],diff(xx,i),2e-5,2e-4,"cell input gradient");
        for(int i=0;i<D;++i){near(da[i],diff(aa,i),2e-5,2e-4,"cell decay gradient with carry");
            near(dq[i],diff(qq,i),2e-5,2e-4,"cell gate gradient");}
    }
    std::puts("PASS normalized/gated recurrence: forward and finite-difference gradients with nonzero carry");
}
static void test_router() {
    constexpr int N=5,E=4,K=2;
    DeviceMemory memory;
    std::vector<float> logits(N*E), sj(N*K,0);
    for(int i=0;i<N*E;++i)logits[i]=sinf(i*.43f)*2;
    auto L=upload(memory,logits), SJ=upload(memory,sj);
    int *idx,*counts;float *w,*p,*g;
    memory.allocate(idx,N*K*4);memory.allocate(counts,E*4);memory.allocate(w,N*K*4);
    memory.allocate(p,N*E*4);memory.allocate(g,N*E*4);
    CUDA_CHECK(cudaMemset(counts,0,E*4));
    topk_kernel<<<1,256>>>(L,idx,w,p,N,E,K);count_kernel<<<1,256>>>(idx,counts,N,K);
    auto cnt=read_gpu(counts,E);
    for(auto coef : {std::pair<float,float>{.7f,0.f},{0.f,.3f}}){
        router_bwd_kernel<<<1,256>>>(p,idx,SJ,g,N,E,K,L,counts,coef.first,coef.second);
        auto got=read_gpu(g,N*E);
        auto objective=[&](const std::vector<double>& z){double loss=0;
            for(int r=0;r<N;++r){double sum=0;for(int e=0;e<E;++e)sum+=exp(z[r*E+e]);
                double lse=log(sum);loss+=coef.second*lse*lse/N;
                for(int e=0;e<E;++e)loss+=coef.first*E/N*exp(z[r*E+e])/sum*cnt[e]/(N*K);
            }return loss;};
        std::vector<double> z(logits.begin(),logits.end());
        for(int i=0;i<N*E;++i){double old=z[i],eps=1e-4;z[i]=old+eps;double hi=objective(z);
            z[i]=old-eps;double lo=objective(z);z[i]=old;near(got[i],(hi-lo)/(2*eps),2e-6,2e-4,"router regularizer gradient");}
    }
    std::puts("PASS MoE auxiliary and z-loss finite-difference gradients");
}
static Cfg small_config(int experts=1) {
    Cfg c;c.dim=16;c.layers=2;c.batch=2;c.seqlen=7;c.experts=experts;c.topk=experts==1?1:2;
    c.warmup=0;c.lr=.005f;c.half_max=16;return c;
}
static void inputs(Model& m) {
    int n=m.c.batch*m.c.seqlen;std::vector<int> x(n),y(n),e(n,0);
    for(int i=0;i<n;++i){x[i]=65+i%3;y[i]=65+(i+1)%3;}
    CUDA_CHECK(cudaMemcpy(m.ids,x.data(),n*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.nxt,y.data(),n*4,cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(m.end,e.data(),n*4,cudaMemcpyHostToDevice));
}
static void test_model(int experts) {
    Model m;m.c=small_config(experts);build_model(m);StreamState state;build_state(state,m);inputs(m);
    float loss,ce;forward_window(m,state,loss,ce);
    auto before=read_gpu(m.L[1].S,m.c.batch*m.c.seqlen*m.c.dim);
    release_window(m);reset_state(state,m.c);
    auto& weight=m.params.at(m.L[0].exp[0]);
    auto saved=read_gpu(weight.master,weight.n),changed=saved;
    for(size_t i=0;i<changed.size();++i)changed[i]+=(i%5-2.0f)*.1f;
    CUDA_CHECK(cudaMemcpy(weight.master,changed.data(),weight.n*4,cudaMemcpyHostToDevice));
    copy_bf16_kernel<<<(weight.n+255)/256,256>>>(weight.master,weight.work,weight.n);
    forward_window(m,state,loss,ce);auto after=read_gpu(m.L[1].S,before.size());
    double delta=0;for(size_t i=0;i<before.size();++i)delta+=fabs(before[i]-after[i]);
    require(delta>.01,"higher recurrent layer ignores lower expert output");release_window(m);
    CUDA_CHECK(cudaMemcpy(weight.master,saved.data(),weight.n*4,cudaMemcpyHostToDevice));
    copy_bf16_kernel<<<(weight.n+255)/256,256>>>(weight.master,weight.work,weight.n);
    float first=0,last=0;
    for(int step=0;step<30;++step){reset_state(state,m.c);forward_window(m,state,loss,ce);
        if(step==0)first=ce;last=ce;backward_window(m,state);optimizer_step(m,step);release_window(m);}
    require(last<first*.8,"training does not reduce CE on repeated byte pattern");
    std::printf("PASS hierarchical %s model learns: CE %.4f -> %.4f\n",experts==1?"dense":"MoE",first,last);
    // Two separately owned states produce the same outputs from reset.
    StreamState other;build_state(other,m);reset_state(state,m.c);
    forward_window(m,state,loss,ce);auto output=read_gpu(m.logits,m.c.batch*m.c.seqlen*256);release_window(m);
    forward_window(m,other,loss,ce);auto output2=read_gpu(m.logits,output.size());release_window(m);
    for(size_t i=0;i<output.size();++i)near(bf2f(output[i]),bf2f(output2[i]),0,0,"independent stream state");
    // Exact resume: weights + Adam + progress + carry, followed by another update.
    std::string path="/tmp/tmt-test-"+std::to_string(getpid())+"-"+std::to_string(experts)+".ckpt";
    Progress progress;progress.step=30;progress.cursor=7;progress.carried=7;progress.data_size=1000;progress.data_hash=42;
    save_checkpoint(path,m,state,progress);
    Model restored;restored.c=checkpoint_config(path);build_model(restored);
    StreamState rs;build_state(rs,restored);Progress rp;load_checkpoint(path,restored,rs,rp);inputs(restored);
    require(rp.step==30&&rp.cursor==7&&rs.position==7,"checkpoint progress mismatch");
    forward_window(m,state,loss,ce);backward_window(m,state);optimizer_step(m,30);release_window(m);
    forward_window(restored,rs,loss,ce);backward_window(restored,rs);optimizer_step(restored,30);release_window(restored);
    for(size_t j=0;j<m.params.values.size();++j){auto a=read_gpu(m.params.at(j).master,m.params.at(j).n);
        auto b=read_gpu(restored.params.at(j).master,restored.params.at(j).n);
        for(size_t i=0;i<a.size();++i)near(a[i],b[i],1e-6,1e-5,"checkpoint resumed update");}
    {FILE* f=fopen(path.c_str(),"r+b");fseek(f,20,SEEK_SET);int byte=fgetc(f);fseek(f,20,SEEK_SET);fputc(byte^1,f);fclose(f);}
    bool rejected=false;try{verify_checkpoint(path);}catch(const std::exception&){rejected=true;}
    unlink(path.c_str());require(rejected,"corrupted checkpoint accepted");
    std::puts("PASS separate streams, checkpoint resume and corruption rejection");
}
static void test_mla_chunks() {
    Model a,b;a.c=small_config();a.c.layers=1;a.c.mla=1;a.c.mla_heads=2;
    a.c.mla_dh=4;a.c.mla_L=4;a.c.mla_R=6;a.c.mla_cache=17;a.c.mla_cc=3;
    b.c=a.c;b.c.mla_cc=64;build_model(a);build_model(b);
    StreamState sa,sb;build_state(sa,a);build_state(sb,b);inputs(a);inputs(b);
    for(int round=0;round<4;++round){float la,ca,lb,cb;
        forward_window(a,sa,la,ca);forward_window(b,sb,lb,cb);
        auto ya=read_gpu(a.logits,a.c.batch*a.c.seqlen*256),yb=read_gpu(b.logits,ya.size());
        double max_error=0;for(size_t i=0;i<ya.size();++i)max_error=std::max(max_error,(double)fabs(bf2f(ya[i])-bf2f(yb[i])));
        require(max_error<.035,"MLA output depends on chunk size");
        backward_window(a,sa);backward_window(b,sb);
        double max_grad_error=0,max_grad=0;
        for(size_t j=0;j<a.params.values.size();++j){auto x=read_gpu(a.params.at(j).grad,a.params.at(j).n);
            auto y=read_gpu(b.params.at(j).grad,b.params.at(j).n);
            for(size_t i=0;i<x.size();++i){max_grad_error=std::max(max_grad_error,(double)fabs(x[i]-y[i]));max_grad=std::max(max_grad,(double)fabs(y[i]));}}
        require(max_grad_error<.003+.04*max_grad,"MLA gradient depends on chunk size");
        require(sa.cache[0].head<=17&&sa.cache[0].base0+sa.cache[0].head==sa.position,"MLA cache lost absolute positions");
        if(round>=2)require(sa.cache[0].head==17&&sa.cache[0].base0>0,"MLA discards entire cache on overflow");
        release_window(a);release_window(b);
    }
    std::puts("PASS MLA partial chunks, multi-chunk gradients, RoPE positions and prefix eviction");
}
// Independent FP64 attention reference, including analytic backward. This is
// intentionally scalar code, separate from the CUDA layout/GEMM implementation.
static void test_mla_reference() {
    Model m;m.c=small_config();m.c.layers=1;m.c.mla=1;m.c.mla_heads=2;
    m.c.mla_dh=3;m.c.mla_L=4;m.c.mla_R=2;m.c.mla_cc=3;
    build_model(m);StreamState state;build_state(state,m);
    const int B=m.c.batch,T=m.c.seqlen,D=m.c.dim,H=m.c.mla_heads,F=m.c.mla_dh,R=m.c.mla_R,L=m.c.mla_L,N=B*T;
    const auto& p=m.ML[0].p;auto& keep=m.ML[0].keep;
    std::vector<double> x(N*D),dy(N*D);
    std::vector<bf16> xb(x.size()),dyb(dy.size());
    for(size_t i=0;i<x.size();++i){xb[i]=f2bf(sinf(i*.37f));x[i]=bf2f(xb[i]);dyb[i]=f2bf(cosf(i*.19f)*.1f);dy[i]=bf2f(dyb[i]);}
    auto weight=[&](size_t id){auto v=read_gpu(m.params.at(id).work,m.params.at(id).n);std::vector<double>w(v.size());
        for(size_t i=0;i<v.size();++i)w[i]=bf2f(v[i]);return w;};
    auto wq=weight(p.q),wk=weight(p.dkv),wr=weight(p.kr),wu=weight(p.uk),wv=weight(p.uv),wo=weight(p.o);
    auto linear=[](const std::vector<double>& a,const std::vector<double>& w,int rows,int out,int in){
        std::vector<double>y(rows*out,0);for(int r=0;r<rows;++r)for(int o=0;o<out;++o)for(int i=0;i<in;++i)y[r*out+o]+=a[r*in+i]*w[o*in+i];return y;};
    auto linear_backward=[](const std::vector<double>& a,const std::vector<double>& w,const std::vector<double>& grad,
                           int rows,int out,int in,std::vector<double>& da){
        std::vector<double>dw(out*in,0);if(da.empty())da.resize(rows*in,0);
        for(int r=0;r<rows;++r)for(int o=0;o<out;++o)for(int i=0;i<in;++i){dw[o*in+i]+=grad[r*out+o]*a[r*in+i];da[r*in+i]+=grad[r*out+o]*w[o*in+i];}return dw;};
    auto q=linear(x,wq,N,H*(F+R),D),lat=linear(x,wk,N,L,D),kr=linear(x,wr,N,R,D);
    auto key=linear(lat,wu,N,H*F,L),value=linear(lat,wv,N,H*F,L);
    // Nonzero positions exercise inverse query and key rotations.
    const int base=11;state.cache[0].base0=base;
    auto rotate=[&](std::vector<double>& a,int heads,int width,int offset,bool inverse){
        for(int n=0;n<N;++n)for(int h=0;h<heads;++h)for(int j=0;j<R;j+=2){
            double angle=(base+n%T)/pow(m.c.mla_theta,j/(double)R)*(inverse?-1:1),c=cos(angle),s=sin(angle);
            int i=(n*heads+h)*width+offset+j;double v=a[i],u=a[i+1];a[i]=c*v-s*u;a[i+1]=s*v+c*u;
        }};
    rotate(q,H,F+R,F,false);rotate(kr,1,R,0,false);
    std::vector<double>prob(B*H*T*T,0),out(N*H*F,0);
    double scale=1/sqrt((double)F);
    for(int b=0;b<B;++b)for(int h=0;h<H;++h)for(int t=0;t<T;++t){
        int n=b*T+t;double maxs=-1e30,sum=0;
        for(int k=0;k<=t;++k){int nk=b*T+k;double s=0;
            for(int d=0;d<F;++d)s+=q[(n*H+h)*(F+R)+d]*key[(nk*H+h)*F+d];
            for(int d=0;d<R;++d)s+=q[(n*H+h)*(F+R)+F+d]*kr[nk*R+d];
            int j=((b*H+h)*T+t)*T+k;prob[j]=s*scale;maxs=std::max(maxs,prob[j]);}
        for(int k=0;k<=t;++k){int j=((b*H+h)*T+t)*T+k;prob[j]=exp(prob[j]-maxs);sum+=prob[j];}
        for(int k=0;k<=t;++k){int j=((b*H+h)*T+t)*T+k;prob[j]/=sum;
            for(int d=0;d<F;++d)out[(n*H+h)*F+d]+=prob[j]*value[((b*T+k)*H+h)*F+d];}
    }
    auto y=linear(out,wo,N,D,H*F);std::vector<double>dout;
    auto dwo=linear_backward(out,wo,dy,N,D,H*F,dout);
    std::vector<double>dq(q.size(),0),dkr(kr.size(),0),dk(key.size(),0),dv(value.size(),0);
    for(int b=0;b<B;++b)for(int h=0;h<H;++h)for(int t=0;t<T;++t){int n=b*T+t;double dot=0;
        for(int d=0;d<F;++d)dot+=dout[(n*H+h)*F+d]*out[(n*H+h)*F+d];
        for(int k=0;k<=t;++k){int nk=b*T+k,j=((b*H+h)*T+t)*T+k;double dp=0;
            for(int d=0;d<F;++d){dp+=dout[(n*H+h)*F+d]*value[(nk*H+h)*F+d];dv[(nk*H+h)*F+d]+=prob[j]*dout[(n*H+h)*F+d];}
            double ds=scale*prob[j]*(dp-dot);
            for(int d=0;d<F;++d){dq[(n*H+h)*(F+R)+d]+=ds*key[(nk*H+h)*F+d];dk[(nk*H+h)*F+d]+=ds*q[(n*H+h)*(F+R)+d];}
            for(int d=0;d<R;++d){dq[(n*H+h)*(F+R)+F+d]+=ds*kr[nk*R+d];dkr[nk*R+d]+=ds*q[(n*H+h)*(F+R)+F+d];}
        }}
    rotate(dq,H,F+R,F,true);rotate(dkr,1,R,0,true);
    std::vector<double>dlat,dx;
    auto dwu=linear_backward(lat,wu,dk,N,H*F,L,dlat),dwv=linear_backward(lat,wv,dv,N,H*F,L,dlat);
    auto dwq=linear_backward(x,wq,dq,N,H*(F+R),D,dx),dwk=linear_backward(x,wk,dlat,N,L,D,dx),dwr=linear_backward(x,wr,dkr,N,R,D,dx);
    DeviceMemory memory;auto X=upload(memory,xb),DY=upload(memory,dyb);
    std::vector<long>positions(N);for(int n=0;n<N;++n)positions[n]=base+n%T;auto pos=upload(memory,positions);
    auto W=[&](size_t id){return m.params.at(id).work;};auto G=[&](size_t id){return m.params.at(id).grad;};
    mla_forward(X,N,W(p.q),W(p.dkv),W(p.kr),W(p.uk),W(p.uv),W(p.o),state.cache[0],keep,m.MW,m.M,pos,
                B,T,H,F,L,R,m.c.mla_cc,m.c.mla_cache,m.c.mla_theta,scale,D);
    mla_backward(DY,N,W(p.q),W(p.dkv),W(p.kr),W(p.uk),W(p.uv),W(p.o),G(p.q),G(p.dkv),G(p.kr),G(p.uk),G(p.uv),G(p.o),
                 state.cache[0],keep,m.MW,m.dH,X,pos,B,T,H,F,L,R,m.c.mla_cc,m.c.mla_theta,scale,D);
    auto check_bf=[&](const bf16* ptr,const std::vector<double>& ref,const char* name){auto got=read_gpu(ptr,ref.size());
        double norm=0,error=0;for(size_t i=0;i<ref.size();++i){norm+=ref[i]*ref[i];error+=pow(bf2f(got[i])-ref[i],2);}
        near(sqrt(error),0,1e-4+.025*sqrt(norm),0,name);};
    auto check_fp=[&](size_t id,const std::vector<double>& ref,const char* name){auto got=read_gpu(G(id),ref.size());
        double norm=0,error=0;for(size_t i=0;i<ref.size();++i){norm+=ref[i]*ref[i];error+=pow(got[i]-ref[i],2);}
        near(sqrt(error),0,1e-4+.03*sqrt(norm),0,name);};
    check_bf(m.M,y,"MLA CPU forward");check_bf(m.dH,dx,"MLA CPU input gradient");
    check_fp(p.q,dwq,"MLA Q gradient");check_fp(p.dkv,dwk,"MLA down KV gradient");check_fp(p.kr,dwr,"MLA RoPE gradient");
    check_fp(p.uk,dwu,"MLA up K gradient");check_fp(p.uv,dwv,"MLA up V gradient");check_fp(p.o,dwo,"MLA output gradient");
    std::puts("PASS MLA forward and ALL gradients against independent FP64 CPU reference");
}
static void compare_checkpoints(const std::string& first,const std::string& second) {
    Model a,b;a.c=checkpoint_config(first);b.c=checkpoint_config(second);
    require(config_text(a.c)==config_text(b.c),"checkpoint configs differ");
    build_model(a);build_model(b);StreamState sa,sb;build_state(sa,a);build_state(sb,b);
    Progress pa,pb;load_checkpoint(first,a,sa,pa);load_checkpoint(second,b,sb,pb);
    require(pa.step==pb.step&&pa.cursor==pb.cursor&&pa.epoch==pb.epoch&&pa.carried==pb.carried&&
            pa.data_size==pb.data_size&&pa.data_hash==pb.data_hash&&sa.position==sb.position,"checkpoint progress differs");
    auto compare=[&](float* x,float* y,long n){auto av=read_gpu(x,n),bv=read_gpu(y,n);
        for(long i=0;i<n;++i)near(av[i],bv[i],1e-7,1e-5,"resumed checkpoint values");};
    for(size_t j=0;j<a.params.values.size();++j){auto& x=a.params.at(j);auto& y=b.params.at(j);
        compare(x.master,y.master,x.n);compare(x.m,y.m,x.n);compare(x.v,y.v,x.n);}
    for(int l=0;l<a.c.layers;++l){compare(sa.carry[l],sb.carry[l],(long)a.c.batch*a.c.dim);
        if(!a.ML[l].use)continue;
        auto& x=sa.cache[l];auto& y=sb.cache[l];require(x.head==y.head&&x.base0==y.base0,"cache metadata differs");
        for(int stream=0;stream<a.c.batch;++stream)for(bool rope:{false,true}){
            int f=rope?x.R:x.L;auto av=read_gpu((rope?x.kr:x.lat)+(long)stream*x.Cmax*f,x.head*f);
            auto bv=read_gpu((rope?y.kr:y.lat)+(long)stream*y.Cmax*f,y.head*f);
            for(size_t i=0;i<av.size();++i)near(bf2f(av[i]),bf2f(bv[i]),1e-6,1e-5,"resumed cache values");
        }}
    std::puts("PASS checkpoint parameters, Adam moments, data progress and stream history match within FP32 tolerance");
}
int main(int argc,char** argv){try{
    if(argc==4 && std::string(argv[1])=="--compare") {compare_checkpoints(argv[2],argv[3]);return 0;}
test_cell();test_router();test_model(1);test_model(3);test_mla_chunks();test_mla_reference();
    CUDA_CHECK(cudaDeviceSynchronize());std::puts("ALL ARCHITECTURE TESTS PASSED");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
