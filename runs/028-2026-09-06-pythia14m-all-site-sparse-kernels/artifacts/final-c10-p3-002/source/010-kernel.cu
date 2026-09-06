// Run027 compatibility port of frozen Run026 K018; no retuning.
// Gated/ungated inputs and same-kernel zero-skipping-disabled control.
// Warp compaction derives from Run025 K001 / Run026 K017 (Sakana lineage).
// Preserve BF16 rounding at BOTH linear outputs and BOTH residual additions.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>

template<int E, bool Gate, bool Skip>
__device__ void accumulate(const __nv_bfloat16* x,const __nv_bfloat16* w,
 int row,int k,int n,int col,int lane,float threshold,float (&acc)[E]) {
    for(int base=0;base<k;base+=32){
        float v=base+lane<k?__bfloat162float(x[int64_t(row)*k+base+lane]):0.f;
        if constexpr (Gate) if(v<threshold) v=0.f;
        unsigned mask=__ballot_sync(0xffffffffu,base+lane<k && (!Skip || v!=0.f));
        while(mask){
            int j=__ffs(mask)-1;
            float value=__shfl_sync(0xffffffffu,v,j);
            #pragma unroll
            for(int e=0;e<E;++e) if(col+e<n)
                acc[e]=fmaf(value,__bfloat162float(w[int64_t(base+j)*n+col+e]),acc[e]);
            mask&=mask-1;
        }
    }
}

template<bool GH, bool GZ, bool Skip>
__global__ void joint(const __nv_bfloat16* h,const __nv_bfloat16* z,
 const __nv_bfloat16* wh,const __nv_bfloat16* wz,const __nv_bfloat16* bh,
 const __nv_bfloat16* bz,const __nv_bfloat16* residual,__nv_bfloat16* out,
 int m,int kh,int kz,int n,float th,float tz){
    int lane=threadIdx.x&31,row=blockIdx.x*4+(threadIdx.x>>5);
    if(row>=m)return;
    int col=blockIdx.y*128+lane*4;
    float ah[4]={},az[4]={};
    accumulate<4,GH,Skip>(h,wh,row,kh,n,col,lane,th,ah);
    accumulate<4,GZ,Skip>(z,wz,row,kz,n,col,lane,tz,az);
    #pragma unroll
    for(int e=0;e<4;++e)if(col+e<n){
        float hv=__bfloat162float(__float2bfloat16_rn(ah[e]+__bfloat162float(bh[col+e])));
        float zv=__bfloat162float(__float2bfloat16_rn(az[e]+__bfloat162float(bz[col+e])));
        float sum=__bfloat162float(__float2bfloat16_rn(hv+zv));
        out[int64_t(row)*n+col+e]=__float2bfloat16_rn(sum+__bfloat162float(residual[int64_t(row)*n+col+e]));
    }
}

void forward(torch::Tensor h,torch::Tensor z,torch::Tensor wh,torch::Tensor wz,
 torch::Tensor bh,torch::Tensor bz,torch::Tensor residual,torch::Tensor out,double th,double tz,bool gh,bool gz,bool skip){
    TORCH_CHECK(h.is_cuda() && h.scalar_type()==at::kBFloat16 && h.dim()==2,"CUDA BF16 h required");
    for(const auto& t : {h,z,wh,wz,bh,bz,residual,out})
        TORCH_CHECK(t.device()==h.device() && t.scalar_type()==h.scalar_type() && t.is_contiguous(),"operand type/layout mismatch");
    TORCH_CHECK(z.dim()==2 && wh.dim()==2 && wz.dim()==2 && residual.dim()==2 && out.dim()==2,"matrix dimensions required");
    int m=h.size(0),kh=h.size(1),kz=z.size(1),n=wh.size(1);
    TORCH_CHECK(m>0 && kh>0 && kz>0 && n>0 && kh<=65535 && kz<=65535 && n<=65535,"invalid shape");
    TORCH_CHECK(z.size(0)==m && wh.size(0)==kh && wz.size(0)==kz && wz.size(1)==n,"projection shape mismatch");
    TORCH_CHECK(bh.dim()==1 && bz.dim()==1 && bh.numel()==n && bz.numel()==n,"bias shape mismatch");
    TORCH_CHECK(residual.size(0)==m && residual.size(1)==n && out.sizes()==residual.sizes(),"residual/output mismatch");
    TORCH_CHECK(std::isfinite(th) && std::isfinite(tz) && th>=0 && tz>=0,"invalid threshold");
    c10::cuda::CUDAGuard guard(h.device());
    #define PTR(t) reinterpret_cast<const __nv_bfloat16*>(t.data_ptr())
    #define LAUNCH(GH,GZ,SKIP) joint<GH,GZ,SKIP><<<dim3((m+3)/4,(n+127)/128),128,0,at::cuda::getCurrentCUDAStream()>>>( \
        PTR(h),PTR(z),PTR(wh),PTR(wz),PTR(bh),PTR(bz),PTR(residual), \
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),m,kh,kz,n,th,tz)
    if(gh && gz){ if(skip){LAUNCH(true,true,true);}else{LAUNCH(true,true,false);} }
    else if(gh){ if(skip){LAUNCH(true,false,true);}else{LAUNCH(true,false,false);} }
    else if(gz){ if(skip){LAUNCH(false,true,true);}else{LAUNCH(false,true,false);} }
    else{ if(skip){LAUNCH(false,false,true);}else{LAUNCH(false,false,false);} }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
