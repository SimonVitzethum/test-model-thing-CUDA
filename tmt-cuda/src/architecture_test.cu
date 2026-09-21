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
// Hybrid traces: with one layer only layer 0's carry crosses the window
// boundary, and it depends only on decay, gate and embedding. Two T-windows with
// traces must therefore reproduce the full BPTT gradient of one 2T-window for
// ALL parameters; without traces they must not.
static void test_traces(int docsep=-1) {
    const int T=6;
    auto config=[&](int seqlen,int traces){Cfg c=small_config();c.layers=1;c.seqlen=seqlen;c.traces=traces;
        c.aux=0;c.zloss=0;c.docsep=docsep;return c;};
    std::vector<int> seq(2*(2*T+1));
    for(size_t i=0;i<seq.size();++i)seq[i]=65+(i*7+i/5)%11;
    auto feed=[&](Model& m,int offset){int B=m.c.batch,L=m.c.seqlen,n=B*L;std::vector<int> x(n),y(n),e(n,0);
        for(int b=0;b<B;++b)for(int t=0;t<L;++t){x[b*L+t]=seq[b*(2*T+1)+offset+t];y[b*L+t]=seq[b*(2*T+1)+offset+t+1];}
        CUDA_CHECK(cudaMemcpy(m.ids,x.data(),n*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.nxt,y.data(),n*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.end,e.data(),n*4,cudaMemcpyHostToDevice));};
    auto nonzero_gate=[](Model& m){auto& g=m.params.at(m.L[0].gate);std::vector<float> h(g.n);
        for(long d=0;d<g.n;++d)h[d]=.8f*sinf(d*1.3f+.4f);
        CUDA_CHECK(cudaMemcpy(g.master,h.data(),g.n*4,cudaMemcpyHostToDevice));
        copy_bf16_kernel<<<1,256>>>(g.master,g.work,g.n);};
    auto grads=[](Model& m){std::vector<std::vector<float>> g;for(auto& p:m.params.values)g.push_back(read_gpu(p.grad,p.n));return g;};
    float loss,ce;
    Model full;full.c=config(2*T,0);build_model(full);nonzero_gate(full);StreamState fs;build_state(fs,full);
    feed(full,0);forward_window(full,fs,loss,ce);backward_window(full,fs);auto ref=grads(full);
    auto two_windows=[&](int traces){Model m;m.c=config(T,traces);build_model(m);nonzero_gate(m);
        StreamState st;build_state(st,m);
        feed(m,0);forward_window(m,st,loss,ce);backward_window(m,st);auto g=grads(m);
        feed(m,T);forward_window(m,st,loss,ce);backward_window(m,st);auto h=grads(m);
        for(size_t j=0;j<g.size();++j)for(size_t i=0;i<g[j].size();++i)g[j][i]+=h[j][i];
        return g;};
    auto error=[&](const std::vector<std::vector<float>>& g,size_t j){double e=0,n=0;
        for(size_t i=0;i<g[j].size();++i){double r=2*ref[j][i];e+=pow(g[j][i]-r,2);n+=r*r;}
        return std::make_pair(sqrt(e),sqrt(n));};
    auto hybrid=two_windows(1),plain=two_windows(0);
    Model shape;shape.c=config(T,1);build_model(shape);
    for(size_t j=0;j<ref.size();++j){if(j==shape.tgt||j==shape.stop)continue;
        auto [e,n]=error(hybrid,j);near(e,0,1e-5+.02*n,0,"hybrid trace gradient vs full BPTT");}
    if(docsep>=0){std::printf("PASS hybrid traces stay exact with document resets (docsep=%d)\n",docsep);return;}
    double worst_hybrid=0,best_plain=1e30;
    for(size_t j:{shape.L[0].decay,shape.L[0].gate,shape.emb}){
        auto [eh,n]=error(hybrid,j);auto [ep,n2]=error(plain,j);(void)n2;
        require(ep>.05*n&&ep>10*eh,"traces do not change the cross-window gradient");
        worst_hybrid=std::max(worst_hybrid,eh/n);best_plain=std::min(best_plain,ep/n);}
    std::printf("PASS hybrid traces reproduce full BPTT across a window boundary (1 layer, all parameters; "
                "decay/gate/emb rel. error %.1e with traces vs >= %.2f without)\n",worst_hybrid,best_plain);
}
// Document reset: after a separator byte, outputs no longer depend on the
// previous document (a_t = 0 cuts carry and traces through the recurrence).
static void test_docsep() {
    Cfg c=small_config();c.traces=1;c.docsep=88;c.seqlen=9;
    Model m;m.c=c;build_model(m);StreamState st;build_state(st,m);
    const int B=c.batch,T=c.seqlen,sep=4;
    auto run=[&](int variant){std::vector<int> x(B*T),y(B*T),e(B*T,0);
        for(int b=0;b<B;++b)for(int t=0;t<T;++t){int v=t<sep?65+(t*3+variant*5+b)%7:(t==sep?88:66+(t+b)%5);
            x[b*T+t]=v;y[b*T+t]=66+(t+1)%5;}
        CUDA_CHECK(cudaMemcpy(m.ids,x.data(),B*T*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.nxt,y.data(),B*T*4,cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(m.end,e.data(),B*T*4,cudaMemcpyHostToDevice));
        reset_state(st,m.c);
        // A nonzero incoming carry from an earlier "document" must not leak either.
        std::vector<float> carry(B*c.dim);for(size_t i=0;i<carry.size();++i)carry[i]=sinf(i+variant*3.f);
        for(int l=0;l<c.layers;++l)CUDA_CHECK(cudaMemcpy(st.carry[l],carry.data(),carry.size()*4,cudaMemcpyHostToDevice));
        float loss,ce;forward_window(m,st,loss,ce);auto out=read_gpu(m.logits,B*T*256);release_window(m);return out;};
    auto a=run(0),b=run(1);double before=0;
    for(int bb=0;bb<B;++bb)for(int t=0;t<T;++t)for(int k=0;k<256;++k){size_t i=((size_t)bb*T+t)*256+k;
        if(t>=sep)near(bf2f(a[i]),bf2f(b[i]),0,0,"output after separator depends on previous document");
        else before+=fabs(bf2f(a[i])-bf2f(b[i]));}
    require(before>0,"test inputs do not differ before the separator");
    std::puts("PASS document reset: outputs after a separator are independent of the previous document");
}
// trace_decay is defined per byte: after 2T bytes the traces must be the same
// whether they were advanced in windows of T bytes or of one byte.
static void test_trace_decay() {
    const int T=6;
    auto traces=[&](int seqlen){Cfg c=small_config();c.traces=1;c.trace_decay=.9f;c.seqlen=seqlen;c.aux=0;c.zloss=0;
        Model m;m.c=c;build_model(m);StreamState st;build_state(st,m);
        for(int w=0;w<2*T/seqlen;++w){int B=c.batch,n=B*seqlen;std::vector<int> x(n),y(n),e(n,0);
            for(int b=0;b<B;++b)for(int t=0;t<seqlen;++t){int p=w*seqlen+t;x[b*seqlen+t]=65+(p*7+b)%9;y[b*seqlen+t]=65+(p*7+b+7)%9;}
            CUDA_CHECK(cudaMemcpy(m.ids,x.data(),n*4,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(m.nxt,y.data(),n*4,cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(m.end,e.data(),n*4,cudaMemcpyHostToDevice));
            float loss,ce;forward_window(m,st,loss,ce);backward_window(m,st);release_window(m);}
        long bd=(long)c.batch*c.dim;std::vector<float> all=read_gpu(st.temb,256*bd);
        for(int l=0;l<c.layers;++l){auto d=read_gpu(st.tdec[l],bd),g=read_gpu(st.tgate[l],bd);
            all.insert(all.end(),d.begin(),d.end());all.insert(all.end(),g.begin(),g.end());}
        return all;};
    auto a=traces(T),b=traces(1);double e=0,n=0;
    for(size_t i=0;i<a.size();++i){e+=pow(a[i]-b[i],2);n+=b[i]*b[i];}
    near(sqrt(e),0,1e-6+2e-3*sqrt(n),0,"trace_decay depends on the window length");
    std::puts("PASS trace_decay is per byte: traces after 2T bytes match for windows of T and of 1 byte");
}
// MoE backward against the dense path: two experts with identical weights,
// both selected (normalized top-k weights sum to 1), compute exactly the dense
// layer. The summed expert gradients and all other gradients must match.
static void test_moe_backward() {
    Cfg dense=small_config(1),moe=small_config(2);moe.topk=2;dense.aux=moe.aux=0;dense.zloss=moe.zloss=0;
    Model a,b;a.c=dense;b.c=moe;build_model(a);build_model(b);
    for(int l=0;l<dense.layers;++l){auto& w=a.params.at(a.L[l].exp[0]);
        for(int e=0;e<2;++e){auto& v=b.params.at(b.L[l].exp[e]);
            CUDA_CHECK(cudaMemcpy(v.master,w.master,w.n*4,cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(v.work,w.work,w.n*2,cudaMemcpyDeviceToDevice));}}
    // Parameters before the experts must be identical too (init order differs by expert count).
    auto copy=[&](size_t from,size_t to){auto& x=a.params.at(from);auto& y=b.params.at(to);
        CUDA_CHECK(cudaMemcpy(y.master,x.master,x.n*4,cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(y.work,x.work,x.n*2,cudaMemcpyDeviceToDevice));};
    copy(a.emb,b.emb);copy(a.dec,b.dec);copy(a.stop,b.stop);copy(a.tgt,b.tgt);
    for(int l=0;l<dense.layers;++l){copy(a.L[l].decay,b.L[l].decay);copy(a.L[l].gate,b.L[l].gate);
        copy(a.L[l].gamma,b.L[l].gamma);copy(a.L[l].beta,b.L[l].beta);}
    StreamState sa,sb;build_state(sa,a);build_state(sb,b);inputs(a);inputs(b);
    float la,ca,lb,cb;forward_window(a,sa,la,ca);forward_window(b,sb,lb,cb);
    near(cb,ca,1e-3,1e-3,"MoE with identical experts differs from dense forward");
    backward_window(a,sa);backward_window(b,sb);
    auto rel=[&](std::vector<float> x,std::vector<float> y,const char* what){double e=0,n=0;
        for(size_t i=0;i<x.size();++i){e+=pow(x[i]-y[i],2);n+=y[i]*y[i];}
        require(n>0,"reference gradient is zero");near(sqrt(e),0,1e-6+.03*sqrt(n),0,what);};
    for(int l=dense.layers-1;l>=0;--l){auto ref=read_gpu(a.params.at(a.L[l].exp[0]).grad,a.params.at(a.L[l].exp[0]).n);
        auto g0=read_gpu(b.params.at(b.L[l].exp[0]).grad,ref.size()),g1=read_gpu(b.params.at(b.L[l].exp[1]).grad,ref.size());
        for(size_t i=0;i<ref.size();++i)g0[i]+=g1[i];
        rel(g0,ref,"summed MoE expert gradients differ from dense");
        rel(read_gpu(b.params.at(b.L[l].decay).grad,dense.dim),read_gpu(a.params.at(a.L[l].decay).grad,dense.dim),
            "gradient below MoE layer differs from dense");}
    rel(read_gpu(b.params.at(b.emb).grad,256*dense.dim),read_gpu(a.params.at(a.emb).grad,256*dense.dim),
        "embedding gradient through MoE differs from dense");
    std::puts("PASS MoE backward: identical experts reproduce dense expert and input gradients");
}
// Fact memory: (1) with Wo = 0 the model is bit-identical to mem=0 even with a
// filled memory; (2) forward and all gradients of one memory layer, including
// the encoding scatter, against an independent FP64 CPU reference.
static void test_memory() {
    Cfg base=small_config();Cfg cm=base;cm.mem=1;cm.mem_len=9;cm.mem_heads=2;cm.mem_dh=4;cm.mem_every=1;
    {Model a,b;a.c=base;b.c=cm;build_model(a);build_model(b);StreamState sa,sb;build_state(sa,a);build_state(sb,b);
     inputs(a);inputs(b);std::vector<int> ids(cm.batch*cm.mem_len);for(size_t i=0;i<ids.size();++i)ids[i]=97+i%7;
     CUDA_CHECK(cudaMemcpy(b.MS.ids,ids.data(),ids.size()*4,cudaMemcpyHostToDevice));
     float l1,c1,l2,c2;forward_window(a,sa,l1,c1);forward_window(b,sb,l2,c2);
     auto x=read_gpu(a.logits,base.batch*base.seqlen*256),y=read_gpu(b.logits,x.size());
     for(size_t i=0;i<x.size();++i)near(bf2f(x[i]),bf2f(y[i]),0,0,"untrained memory changes the model");}
    const int B=2,T=5,M=6,H=2,F=3,D=8,HD=H*F,N=B*T;
    std::mt19937 rng(3);std::uniform_real_distribution<float> U(-1,1);
    auto rb=[&](int n,float s){std::vector<bf16> v(n);for(auto& e:v)e=f2bf(U(rng)*s);return v;};
    auto rf=[&](int n,float s,float o){std::vector<float> v(n);for(auto& e:v)e=o+U(rng)*s;return v;};
    auto X=rb(N*D,1),E=rb(256*D,.5f),Ep=rb(256*D,.5f),Pp=rb(M*D,.5f),Wq=rb(HD*D,.5f),Wk=rb(HD*D,.5f),Wv=rb(HD*D,.5f),Wo=rb(D*HD,.5f),dY=rb(N*D,1);
    auto gamma=rf(D,.3f,1),beta=rf(D,.3f,0),edec=rf(D,1.5f,0),egate=rf(D,.8f,0);
    // stream 0: slots 0..3 valid, 4..5 padding; stream 1: no memory at all.
    std::vector<int> ids{3,7,3,9,-1,-1,-1,-1,-1,-1,-1,-1};
    DeviceMemory mem;MemLayer L;MemShared S;
    auto dX_=upload(mem,X),dE=upload(mem,E),dEp=upload(mem,Ep),dPos=upload(mem,Pp),dWq=upload(mem,Wq),dWk=upload(mem,Wk),dWv=upload(mem,Wv),dWo=upload(mem,Wo),dDY=upload(mem,dY);
    auto dG=upload(mem,gamma),dBt=upload(mem,beta),dDec=upload(mem,edec),dGat=upload(mem,egate);S.ids=upload(mem,ids);
    bf16* Y;mem.allocate(Y,N*D*2);
    mem.allocate(S.enc,B*M*D*2);mem.allocate(S.enc0,B*M*D*2);mem.allocate(S.state,B*M*D*4);mem.allocate(S.dEnc0,B*M*D*4);mem.allocate(L.Xsnap,N*D*2);mem.allocate(L.Hn,N*D*2);mem.allocate(L.Q,N*HD*2);mem.allocate(L.O,N*HD*2);
    mem.allocate(L.K,B*M*HD*2);mem.allocate(L.V,B*M*HD*2);mem.allocate(L.mean,N*4);mem.allocate(L.rstd,N*4);mem.allocate(L.P,B*H*T*M*4);
    mem.allocate(S.dO,N*HD*2);mem.allocate(S.dQ,N*HD*2);mem.allocate(S.dK,B*M*HD*2);mem.allocate(S.dV,B*M*HD*2);mem.allocate(S.dHn,N*D*2);
    mem.allocate(S.dXln,N*D*2);mem.allocate(S.dEncB,B*M*D*2);mem.allocate(S.dS,B*H*T*M*4);mem.allocate(S.dEnc,B*M*D*4);
    CUDA_CHECK(cudaMemset(S.dEnc,0,B*M*D*4));
    float *gG,*gB,*gQ,*gK,*gV,*gO,*gE,*gEp,*gP;
    for(float** p:{&gG,&gB}){mem.allocate(*p,D*4);}
    for(float** p:{&gQ,&gK,&gV,&gO}){mem.allocate(*p,HD*D*4);}
    for(float** p:{&gE,&gEp}){mem.allocate(*p,256*D*4);CUDA_CHECK(cudaMemset(*p,0,256*D*4));}
    mem.allocate(gP,M*D*4);CUDA_CHECK(cudaMemset(gP,0,M*D*4));
    float *gDec,*gGat;mem.allocate(gDec,D*4);mem.allocate(gGat,D*4);
    mem_encode(dE,dEp,dPos,dDec,dGat,S,B,M,D);
    mem_forward(dX_,L,S,dG,dBt,dWq,dWk,dWv,dWo,Y,B,T,M,H,F,D);
    auto xout=read_gpu(dX_,N*D);auto enc=read_gpu(S.enc,B*M*D);
    CUDA_CHECK(cudaMemcpy(dX_,dY.data(),N*D*2,cudaMemcpyHostToDevice));  // dX <- dL/dx_out
    mem_backward(dX_,L,S,dG,dWq,dWk,dWv,dWo,gG,gB,gQ,gK,gV,gO,B,T,M,H,F,D);
    mem_encode_backward(S,dDec,dGat,gDec,gGat,gE,gEp,gP,B,M,D);
    // ---- FP64 reference: loss = sum(dY * x_out) ----
    auto f=[](const std::vector<bf16>& v){std::vector<double> o(v.size());for(size_t i=0;i<v.size();++i)o[i]=bf2f(v[i]);return o;};
    auto x=f(X),e=f(E),ep=f(Ep),pp=f(Pp),wq=f(Wq),wk=f(Wk),wv=f(Wv),wo=f(Wo),gy=f(dY);
    std::vector<double> m(B*M*D,0);
    for(int b=0;b<B;++b)for(int j=0;j<M;++j){int id=ids[b*M+j];if(id<0)continue;int pv=j?ids[b*M+j-1]:-1;
        for(int d=0;d<D;++d)m[(b*M+j)*D+d]=e[id*D+d]+pp[j*D+d]+(pv>=0?ep[pv*D+d]:0);}
    // Encoder recurrence: s_j = a s_(j-1) + (1-a) e_j, a = sigmoid(dec + gate e_j); m = e + s.
    std::vector<double> e0(m),st(B*M*D);
    for(int b=0;b<B;++b)for(int d=0;d<D;++d){double sv=0;for(int j=0;j<M;++j){int i=(b*M+j)*D+d;double xv=e0[i];
        double a=1/(1+exp(-edec[d]-egate[d]*xv));sv=a*sv+(1-a)*xv;st[i]=sv;m[i]=xv+sv;}}
    std::vector<double> hn(N*D),mu(N),rs(N),q(N*HD,0),k(B*M*HD,0),v(B*M*HD,0),o(N*HD,0),P(B*H*T*M,0),y(N*D,0);
    for(int n=0;n<N;++n){double s=0,s2=0;for(int d=0;d<D;++d){s+=x[n*D+d];s2+=x[n*D+d]*x[n*D+d];}
        mu[n]=s/D;rs[n]=1/sqrt(s2/D-mu[n]*mu[n]+1e-5);for(int d=0;d<D;++d)hn[n*D+d]=(x[n*D+d]-mu[n])*rs[n]*gamma[d]+beta[d];}
    for(int n=0;n<N;++n)for(int a=0;a<HD;++a)for(int d=0;d<D;++d)q[n*HD+a]+=hn[n*D+d]*wq[a*D+d];
    for(int r=0;r<B*M;++r)for(int a=0;a<HD;++a)for(int d=0;d<D;++d){k[r*HD+a]+=m[r*D+d]*wk[a*D+d];v[r*HD+a]+=m[r*D+d]*wv[a*D+d];}
    double sc=1/sqrt((double)F);
    for(int b=0;b<B;++b)for(int h=0;h<H;++h)for(int t=0;t<T;++t){double mx=-1e300,sum=0;int n=b*T+t;double* p=&P[((b*H+h)*T+t)*M];
        for(int j=0;j<M;++j){if(ids[b*M+j]<0)continue;double s=0;for(int d=0;d<F;++d)s+=q[n*HD+h*F+d]*k[(b*M+j)*HD+h*F+d];p[j]=s*sc;mx=std::max(mx,p[j]);}
        for(int j=0;j<M;++j){if(ids[b*M+j]<0){p[j]=0;continue;}p[j]=exp(p[j]-mx);sum+=p[j];}
        for(int j=0;j<M;++j){p[j]=sum>0?p[j]/sum:0;for(int d=0;d<F;++d)o[n*HD+h*F+d]+=p[j]*v[(b*M+j)*HD+h*F+d];}}
    for(int n=0;n<N;++n)for(int d=0;d<D;++d){for(int a=0;a<HD;++a)y[n*D+d]+=o[n*HD+a]*wo[d*HD+a];}
    std::vector<double> rgO(D*HD,0),dO(N*HD,0),dP(B*H*T*M,0),dS(B*H*T*M,0),dq(N*HD,0),dk(B*M*HD,0),dv(B*M*HD,0);
    for(int n=0;n<N;++n)for(int d=0;d<D;++d)for(int a=0;a<HD;++a){rgO[d*HD+a]+=gy[n*D+d]*o[n*HD+a];dO[n*HD+a]+=gy[n*D+d]*wo[d*HD+a];}
    for(int b=0;b<B;++b)for(int h=0;h<H;++h)for(int t=0;t<T;++t){int n=b*T+t;int base=((b*H+h)*T+t)*M;double dot=0;
        for(int j=0;j<M;++j){double s=0;for(int d=0;d<F;++d)s+=dO[n*HD+h*F+d]*v[(b*M+j)*HD+h*F+d];dP[base+j]=s;dot+=P[base+j]*s;}
        for(int j=0;j<M;++j){dS[base+j]=P[base+j]*(dP[base+j]-dot);
            for(int d=0;d<F;++d){dq[n*HD+h*F+d]+=sc*dS[base+j]*k[(b*M+j)*HD+h*F+d];dk[(b*M+j)*HD+h*F+d]+=sc*dS[base+j]*q[n*HD+h*F+d];
                dv[(b*M+j)*HD+h*F+d]+=P[base+j]*dO[n*HD+h*F+d];}}}
    std::vector<double> rgQ(HD*D,0),rgK(HD*D,0),rgV(HD*D,0),dhn(N*D,0),dm(B*M*D,0);
    for(int n=0;n<N;++n)for(int a=0;a<HD;++a)for(int d=0;d<D;++d){rgQ[a*D+d]+=dq[n*HD+a]*hn[n*D+d];dhn[n*D+d]+=dq[n*HD+a]*wq[a*D+d];}
    for(int r=0;r<B*M;++r)for(int a=0;a<HD;++a)for(int d=0;d<D;++d){rgK[a*D+d]+=dk[r*HD+a]*m[r*D+d];rgV[a*D+d]+=dv[r*HD+a]*m[r*D+d];
        dm[r*D+d]+=dk[r*HD+a]*wk[a*D+d]+dv[r*HD+a]*wv[a*D+d];}
    std::vector<double> rgG(D,0),rgB(D,0),dx(N*D,0);
    for(int n=0;n<N;++n){double s1=0,s2=0;for(int d=0;d<D;++d){double xh=(x[n*D+d]-mu[n])*rs[n];rgG[d]+=dhn[n*D+d]*xh;rgB[d]+=dhn[n*D+d];
            double g=dhn[n*D+d]*gamma[d];s1+=g;s2+=g*xh;}
        for(int d=0;d<D;++d){double xh=(x[n*D+d]-mu[n])*rs[n];dx[n*D+d]=gy[n*D+d]+rs[n]*(dhn[n*D+d]*gamma[d]-s1/D-xh*s2/D);}}
    std::vector<double> rgDec(D,0),rgGat(D,0),de(dm);
    for(int b=0;b<B;++b)for(int d=0;d<D;++d){double fut=0;for(int j=M-1;j>=0;--j){int i=(b*M+j)*D+d;double xv=e0[i];
        double a=1/(1+exp(-edec[d]-egate[d]*xv));double tot=dm[i]+fut;double prev=j?st[i-D]:0;double loc=(prev-xv)*a*(1-a);
        rgDec[d]+=tot*loc;rgGat[d]+=tot*loc*xv;de[i]+=tot*(1-a)+tot*loc*egate[d];fut=tot*a;}}
    std::vector<double> rgE(256*D,0),rgEp(256*D,0),rgP(M*D,0);
    for(int b=0;b<B;++b)for(int j=0;j<M;++j){int id=ids[b*M+j];if(id<0)continue;int pv=j?ids[b*M+j-1]:-1;
        for(int d=0;d<D;++d){double g=de[(b*M+j)*D+d];rgE[id*D+d]+=g;rgP[j*D+d]+=g;if(pv>=0)rgEp[pv*D+d]+=g;}}
    auto check=[&](const std::vector<float>& got,const std::vector<double>& ref,const char* what){double e2=0,n2=0;
        for(size_t i=0;i<ref.size();++i){e2+=pow(got[i]-ref[i],2);n2+=ref[i]*ref[i];}
        require(n2>0,"memory reference gradient is zero");near(sqrt(e2),0,1e-4+.03*sqrt(n2),0,what);};
    auto tof=[](const std::vector<bf16>& v){std::vector<float> o(v.size());for(size_t i=0;i<v.size();++i)o[i]=bf2f(v[i]);return o;};
    std::vector<double> xo(N*D);for(int i=0;i<N*D;++i)xo[i]=x[i]+y[i];
    check(tof(xout),xo,"memory forward");check(tof(enc),m,"memory encoding");
    for(int n=T;n<2*T;++n)for(int d=0;d<D;++d)near(bf2f(xout[n*D+d]),x[n*D+d],0,0,"stream without memory must be unchanged");
    check(tof(read_gpu(dX_,N*D)),dx,"memory input gradient");
    check(read_gpu(gO,D*HD),rgO,"memory Wo gradient");check(read_gpu(gQ,HD*D),rgQ,"memory Wq gradient");
    check(read_gpu(gK,HD*D),rgK,"memory Wk gradient");check(read_gpu(gV,HD*D),rgV,"memory Wv gradient");
    check(read_gpu(gG,D),rgG,"memory LN gamma gradient");check(read_gpu(gB,D),rgB,"memory LN beta gradient");
    check(read_gpu(gE,256*D),rgE,"memory embedding gradient");check(read_gpu(gEp,256*D),rgEp,"memory previous-byte gradient");
    check(read_gpu(gP,M*D),rgP,"memory position gradient");
    check(read_gpu(gDec,D),rgDec,"memory encoder decay gradient");check(read_gpu(gGat,D),rgGat,"memory encoder gate gradient");
    std::puts("PASS fact memory: untrained memory is an exact no-op; forward and all gradients match FP64 reference");
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
    if(a.c.traces){long bd=(long)a.c.batch*a.c.dim;compare(sa.temb,sb.temb,256*bd);
        for(int l=0;l<a.c.layers;++l){compare(sa.tdec[l],sb.tdec[l],bd);compare(sa.tgate[l],sb.tgate[l],bd);}}
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
test_cell();test_router();test_moe_backward();test_memory();test_model(1);test_model(3);test_traces();test_traces(70);test_docsep();test_trace_decay();test_mla_chunks();test_mla_reference();
    CUDA_CHECK(cudaDeviceSynchronize());std::puts("ALL ARCHITECTURE TESTS PASSED");return 0;
}catch(const std::exception& e){std::fprintf(stderr,"FAIL: %s\n",e.what());return 1;}}
