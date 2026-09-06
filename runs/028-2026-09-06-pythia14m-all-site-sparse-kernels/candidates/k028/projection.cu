// K028: sparse K021 traversal plus FP64 repair near BF16 rounding boundaries.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using bf16=__nv_bfloat16;

template<bool Skip,bool Count>
__global__ void project(const bf16* x,const bf16* w,const bf16* b,bf16* y,int m,int n,unsigned long long* stats){
    int lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32,col=blockIdx.y*128+lane*4;
    if(row>=m)return;
    float acc[4]={};int executed=0;
    for(int base=0;base<128;base+=32){
        float v=__bfloat162float(x[row*128+base+lane]);
        unsigned active=__ballot_sync(0xffffffffu,!Skip || v!=0.f);
        while(active){
            int j=__ffs(active)-1;
            float value=__shfl_sync(0xffffffffu,v,j);
            #pragma unroll
            for(int e=0;e<4;++e)acc[e]=fmaf(value,__bfloat162float(w[(base+j)*n+col+e]),acc[e]);
            if constexpr(Count)executed+=4;
            active&=active-1;
        }
    }
    #pragma unroll
    for(int e=0;e<4;++e){
        float value=acc[e]+__bfloat162float(b[col+e]);
        unsigned low=__float_as_uint(value)&0xffffu;
        if(low>=0x7ff8u && low<=0x8008u){
            double precise=0.;int repair_executed=0;
            for(int j=0;j<128;++j){
                float xj=__bfloat162float(x[row*128+j]);
                if(!Skip || xj!=0.f){
                    precise=fma(double(xj),double(__bfloat162float(w[j*n+col+e])),precise);
                    if constexpr(Count)++repair_executed;
                }
            }
            precise+=double(__bfloat162float(b[col+e]));
            y[row*n+col+e]=__double2bfloat16(precise);
            if constexpr(Count){atomicAdd(stats,1ull);atomicAdd(stats+1,(unsigned long long)repair_executed);}
        }else y[row*n+col+e]=__float2bfloat16_rn(value);
    }
    if constexpr(Count){atomicAdd(stats+2,(unsigned long long)executed);atomicAdd(stats+3,(unsigned long long)(512-executed));}
}

void forward(torch::Tensor x,torch::Tensor w,torch::Tensor b,torch::Tensor y,torch::Tensor stats,bool skip,bool count){
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()==2 && x.size(1)==128,"CUDA BF16 Mx128 required");
    for(const auto& v:{x,w,b,y})TORCH_CHECK(v.device()==x.device() && v.scalar_type()==x.scalar_type() && v.is_contiguous(),"Type/layout mismatch");
    TORCH_CHECK(stats.device()==x.device() && stats.scalar_type()==at::kLong && stats.is_contiguous() && stats.numel()==4,"Counter mismatch");
    int m=x.size(0),n=w.size(1);
    TORCH_CHECK(m>0 && w.dim()==2 && w.size(0)==128 && (n==384 || n==512),"Pythia14M QKV/up shapes only");
    TORCH_CHECK(b.dim()==1 && b.numel()==n && y.dim()==2 && y.size(0)==m && y.size(1)==n,"Output/bias mismatch");
    c10::cuda::CUDAGuard guard(x.device());
    #define P(v) reinterpret_cast<bf16*>(v.data_ptr())
    auto stream=at::cuda::getCurrentCUDAStream();
    #define LAUNCH(S,C) project<S,C><<<dim3((m+3)/4,n/128),128,0,stream>>>(P(x),P(w),P(b),P(y),m,n,reinterpret_cast<unsigned long long*>(stats.data_ptr()))
    if(skip){if(count){LAUNCH(true,true);}else{LAUNCH(true,false);}}
    else{if(count){LAUNCH(false,true);}else{LAUNCH(false,false);}}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
