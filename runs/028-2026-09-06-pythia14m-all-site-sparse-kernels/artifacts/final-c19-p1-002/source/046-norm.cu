// Welford operation ordering follows PyTorch's vectorized CUDA LayerNorm at
// 70d99e998b4955e0049d13a98d77ae1b14db1f45. See LICENSE-PYTORCH.
// Specialization: one warp per width128 row, four rows/CTA, two affine outputs.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cmath>
using bf16=__nv_bfloat16;
struct Moments {float mean,m2,n;};
__device__ __forceinline__ Moments add(Moments state,float x){
    float delta=x-state.mean,n=state.n+1.f;
    float mean=state.mean+delta*(1.f/n);
    return {mean,state.m2+delta*(x-mean),n};
}
__device__ __forceinline__ Moments merge(Moments b,Moments a){
    float delta=b.mean-a.mean,n=a.n+b.n;
    float inv=1.f/n,na=a.n*inv,nb=b.n*inv;
    return {na*a.mean+nb*b.mean,a.m2+b.m2+delta*delta*a.n*nb,n};
}
struct alignas(8) Vec4 {bf16 v[4];};
__global__ void norm_pair(const bf16* x,const bf16* wa,const bf16* ba,const bf16* wm,const bf16* bm,
    bf16* a,bf16* m,int rows,float eps,bool ga,bool gm,float ta,float tm){
    int lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32;
    if(row>=rows)return; // whole-warp boundary
    Vec4 xv=reinterpret_cast<const Vec4*>(x+row*128)[lane];
    Moments state{0.f,0.f,0.f};
    #pragma unroll
    for(int i=0;i<4;++i)state=add(state,__bfloat162float(xv.v[i]));
    for(int offset=16;offset>0;offset>>=1){
        Moments other{__shfl_down_sync(0xffffffff,state.mean,offset),
            __shfl_down_sync(0xffffffff,state.m2,offset),__shfl_down_sync(0xffffffff,state.n,offset)};
        state=merge(state,other);
    }
    float mean=__shfl_sync(0xffffffff,state.mean,0);
    float variance=__shfl_sync(0xffffffff,state.m2,0)/128.f;
    float inv=rsqrtf(variance+eps);
    Vec4 av,mv,wav=reinterpret_cast<const Vec4*>(wa)[lane],bav=reinterpret_cast<const Vec4*>(ba)[lane];
    Vec4 wmv=reinterpret_cast<const Vec4*>(wm)[lane],bmv=reinterpret_cast<const Vec4*>(bm)[lane];
    #pragma unroll
    for(int i=0;i<4;++i){
        float normalized=inv*(__bfloat162float(xv.v[i])-mean);
        bf16 ay=__float2bfloat16_rn(__bfloat162float(wav.v[i])*normalized+__bfloat162float(bav.v[i]));
        bf16 my=__float2bfloat16_rn(__bfloat162float(wmv.v[i])*normalized+__bfloat162float(bmv.v[i]));
        av.v[i]=ga && __bfloat162float(ay)<ta?__float2bfloat16_rn(0.f):ay;
        mv.v[i]=gm && __bfloat162float(my)<tm?__float2bfloat16_rn(0.f):my;
    }
    reinterpret_cast<Vec4*>(a+row*128)[lane]=av;
    reinterpret_cast<Vec4*>(m+row*128)[lane]=mv;
}
void forward(torch::Tensor x,torch::Tensor wa,torch::Tensor ba,torch::Tensor wm,torch::Tensor bm,
    torch::Tensor a,torch::Tensor m,double eps,bool ga,bool gm,double ta,double tm){
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()>=2 && x.size(-1)==128 && x.numel()>0,"CUDA BF16 width128 required");
    for(const auto& t:{x,wa,ba,wm,bm,a,m}){
        TORCH_CHECK(t.device()==x.device() && t.scalar_type()==x.scalar_type() && t.is_contiguous(),"Type/layout mismatch");
        TORCH_CHECK(reinterpret_cast<uintptr_t>(t.data_ptr())%8==0,"Eight-byte alignment required");
    }
    for(const auto& t:{wa,ba,wm,bm})TORCH_CHECK(t.dim()==1 && t.numel()==128,"Affine shape");
    TORCH_CHECK(a.sizes()==x.sizes() && m.sizes()==x.sizes(),"Output shape");
    TORCH_CHECK(std::isfinite(eps) && eps>0 && std::isfinite(ta) && std::isfinite(tm) && ta>=0 && tm>=0,"Epsilon/gates");
    c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
    int rows=x.numel()/128;
    #define P(t) reinterpret_cast<bf16*>(t.data_ptr())
    norm_pair<<<(rows+3)/4,128,0,stream>>>(P(x),P(wa),P(ba),P(wm),P(bm),P(a),P(m),rows,eps,ga,gm,ta,tm);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
