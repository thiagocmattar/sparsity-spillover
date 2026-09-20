// K019: fused partial RoPE, exact symmetric Q/K/V gates and head-major layout.
// Preserve each BF16 product and addition of the canonical eager RoPE.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

__device__ float bfround(float x){return __bfloat162float(__float2bfloat16_rn(x));}
__global__ void rope_gate(const __nv_bfloat16* x,const __nv_bfloat16* c,
 const __nv_bfloat16* s,__nv_bfloat16* q,__nv_bfloat16* k,__nv_bfloat16* v,
 int total,int T,int H,int D,int R,float tq,float tk,float tv){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=total)return;
    int d=i%D,t=(i/D)%T,h=(i/(D*T))%H,b=i/(D*T*H);
    int input=((b*T+t)*H+h)*3*D+d;
    float qv=__bfloat162float(x[input]),kv=__bfloat162float(x[input+D]);
    float vv=__bfloat162float(x[input+2*D]);
    if(d<R){
        int other=d<R/2?d+R/2:d-R/2;
        float sign=d<R/2?-1.f:1.f;
        float qr=sign*__bfloat162float(x[input-d+other]);
        float kr=sign*__bfloat162float(x[input-d+other+D]);
        float cv=__bfloat162float(c[(b*T+t)*R+d]);
        float sv=__bfloat162float(s[(b*T+t)*R+d]);
        qv=bfround(__fadd_rn(bfround(__fmul_rn(qv,cv)),bfround(__fmul_rn(qr,sv))));
        kv=bfround(__fadd_rn(bfround(__fmul_rn(kv,cv)),bfround(__fmul_rn(kr,sv))));
    }
    q[i]=__float2bfloat16_rn(fabsf(qv)<tq?0.f:qv);
    k[i]=__float2bfloat16_rn(fabsf(kv)<tk?0.f:kv);
    v[i]=__float2bfloat16_rn(fabsf(vv)<tv?0.f:vv);
}

void forward(torch::Tensor x,torch::Tensor c,torch::Tensor s,torch::Tensor q,
 torch::Tensor k,torch::Tensor v,int H,int D,double tq,double tk,double tv){
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()==3,"CUDA BF16 [B,T,3HD] required");
    for(const auto& t:{x,c,s,q,k,v})
        TORCH_CHECK(t.device()==x.device() && t.scalar_type()==x.scalar_type() && t.is_contiguous(),"operand type/layout mismatch");
    TORCH_CHECK(H>0 && D>0 && x.size(2)==3*H*D && c.dim()==3 && s.sizes()==c.sizes(),"input shape mismatch");
    int B=x.size(0),T=x.size(1),R=c.size(2);
    TORCH_CHECK(B>0 && T>0 && c.size(0)==B && c.size(1)==T && R>0 && R<=D && R%2==0,"invalid RoPE shape");
    TORCH_CHECK(q.dim()==4 && q.size(0)==B && q.size(1)==H && q.size(2)==T && q.size(3)==D && k.sizes()==q.sizes() && v.sizes()==q.sizes(),"output shape mismatch");
    TORCH_CHECK(q.numel()<=2147483647 && tq>=0 && tk>=0 && tv>=0 && std::isfinite(tq) && std::isfinite(tk) && std::isfinite(tv),"invalid size/threshold");
    c10::cuda::CUDAGuard guard(x.device());
    int total=q.numel();
    #define CP(t) reinterpret_cast<const __nv_bfloat16*>(t.data_ptr())
    #define MP(t) reinterpret_cast<__nv_bfloat16*>(t.data_ptr())
    rope_gate<<<(total+255)/256,256,0,at::cuda::getCurrentCUDAStream()>>>(CP(x),CP(c),CP(s),MP(q),MP(k),MP(v),total,T,H,D,R,tq,tk,tv);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
