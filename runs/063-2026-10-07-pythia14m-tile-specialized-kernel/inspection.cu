// Run061: K049 arithmetic and scheduling; exact scalar/corner gates before support and weight loads.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using bf16=__nv_bfloat16;

struct TinyRow { int count=3,i0=0,i1=0; float v0=0.f,v1=0.f; unsigned tiles=0; };
template<int K>
__device__ __forceinline__ bool rejected_corner(const bf16* x,int index,float threshold){
    int row=index/K,feature=index%K;
    int corner=(row/8*8)*K+(feature/16*16);
    return !(__bfloat162float(x[corner])>threshold);
}
template<int K>
__device__ __forceinline__ float gated(const bf16* x,int index,float threshold,int gate){
    if(gate==2 && rejected_corner<K>(x,index,threshold))return 0.f;
    float value=__bfloat162float(x[index]);
    return gate==1 && value<threshold?0.f:value;
}
template<int K>
__device__ __forceinline__ TinyRow inspect(const bf16* x,int row,int lane,float threshold,int gate){
    TinyRow out;out.count=0;
    #pragma unroll
    for(int base=0;base<K;base+=32){
        float value=gated<K>(x,row*K+base+lane,threshold,gate);
        unsigned mask=__ballot_sync(0xffffffffu,value!=0.f);
        if(mask&0xffffu)out.tiles|=1u<<(base/16);
        if(mask&0xffff0000u)out.tiles|=1u<<(base/16+1);
        if(out.count+__popc(mask)>2)out.count=3;
        else while(mask){
            int selected=__ffs(mask)-1;float next=__shfl_sync(0xffffffffu,value,selected);
            if(!(fabsf(next)>=0x1p-50f && fabsf(next)<=0x1p50f)){out.count=3;break;}
            if(out.count==0){out.i0=base+selected;out.v0=next;}
            else{out.i1=base+selected;out.v1=next;}
            ++out.count;mask&=mask-1;
        }
    }
    return out;
}

// Standalone hot-resident support microbenchmark; not an end-to-end timer.
template<int K,bool Generic>
__global__ void probe(const bf16* x,unsigned* masks,float threshold){
    int lane=threadIdx.x&31,warp=threadIdx.x/32,row=blockIdx.x*8;
    if constexpr(Generic){
        auto s=inspect<K>(x,row+warp,lane,threshold,2);
        if(lane==0)masks[blockIdx.x*8+warp]=s.tiles;
    }else{
        bool live=lane<K/16 && __bfloat162float(x[row*K+lane*16])>threshold;
        unsigned bits=__ballot_sync(0xffffffffu,live);
        if(lane==0)masks[blockIdx.x*8+warp]=bits;
    }
}
void inspect_only(torch::Tensor x,torch::Tensor masks,double threshold,bool generic){
    TORCH_CHECK(x.is_cuda() && x.dim()==2 && x.scalar_type()==at::kBFloat16 && x.is_contiguous(),"CUDA BF16 matrix");
    int m=x.size(0),k=x.size(1);
    TORCH_CHECK(m>0 && m%8==0 && (k==128 || k==512),"M8 K128/K512");
    TORCH_CHECK(masks.is_cuda() && masks.device()==x.device() && masks.scalar_type()==at::kInt && masks.is_contiguous() && masks.numel()==m,"Mask layout");
    c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
    auto input=reinterpret_cast<bf16*>(x.data_ptr());auto out=reinterpret_cast<unsigned*>(masks.data_ptr());
    #define INSPECT(K,G) probe<K,G><<<m/8,256,0,stream>>>(input,out,threshold)
    if(k==512){if(generic){INSPECT(512,true);}else{INSPECT(512,false);}}
    else{if(generic){INSPECT(128,true);}else{INSPECT(128,false);}}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("inspect_only",&inspect_only);}
