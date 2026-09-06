// K021: exact nonzero traversal for a->QKV and m->W1, fixed Pythia14M shapes.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using bf16=__nv_bfloat16;

template<bool Skip>
__global__ void project(const bf16* x,const bf16* w,const bf16* b,bf16* y,int m,int n){
    int lane=threadIdx.x&31,row=blockIdx.x*4+threadIdx.x/32,col=blockIdx.y*128+lane*4;
    if(row>=m)return;
    float acc[4]={};
    for(int base=0;base<128;base+=32){
        float v=__bfloat162float(x[row*128+base+lane]);
        unsigned active=__ballot_sync(0xffffffffu,!Skip || v!=0.f);
        while(active){
            int j=__ffs(active)-1;
            float value=__shfl_sync(0xffffffffu,v,j);
            #pragma unroll
            for(int e=0;e<4;++e)acc[e]=fmaf(value,__bfloat162float(w[(base+j)*n+col+e]),acc[e]);
            active&=active-1;
        }
    }
    #pragma unroll
    for(int e=0;e<4;++e)y[row*n+col+e]=__float2bfloat16_rn(acc[e]+__bfloat162float(b[col+e]));
}

void forward(torch::Tensor x,torch::Tensor w,torch::Tensor b,torch::Tensor y,bool skip){
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()==2 && x.size(1)==128,"CUDA BF16 Mx128 required");
    for(const auto& v:{x,w,b,y})TORCH_CHECK(v.device()==x.device() && v.scalar_type()==x.scalar_type() && v.is_contiguous(),"Type/layout mismatch");
    int m=x.size(0),n=w.size(1);
    TORCH_CHECK(m>0 && w.dim()==2 && w.size(0)==128 && (n==384 || n==512),"Pythia14M QKV/up shapes only");
    TORCH_CHECK(b.dim()==1 && b.numel()==n && y.dim()==2 && y.size(0)==m && y.size(1)==n,"Output/bias mismatch");
    c10::cuda::CUDAGuard guard(x.device());
    #define P(v) reinterpret_cast<bf16*>(v.data_ptr())
    auto stream=at::cuda::getCurrentCUDAStream();
    if(skip)project<true><<<dim3((m+3)/4,n/128),128,0,stream>>>(P(x),P(w),P(b),P(y),m,n);
    else project<false><<<dim3((m+3)/4,n/128),128,0,stream>>>(P(x),P(w),P(b),P(y),m,n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
