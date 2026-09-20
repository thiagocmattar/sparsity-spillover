// BF16 tensor-core a/m projections; fixed K128, no TF32 or fast math.
// Register mapping: NVIDIA PTX ISA, mma.m16n8k16 with floating-point type.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using bf16=__nv_bfloat16;

__device__ __forceinline__ void mma(float (&c)[4],unsigned a0,unsigned a1,unsigned a2,unsigned a3,unsigned b0,unsigned b1){
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
        : "+f"(c[0]),"+f"(c[1]),"+f"(c[2]),"+f"(c[3])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
}

template<bool Skip>
__global__ void project(const bf16* x,const bf16* w,const bf16* b,bf16* y,int m,int n){
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    int row=blockIdx.x*64+warp*16+lane/4,col=blockIdx.y*8;
    float acc[4]={};
    #pragma unroll
    for(int base=0;base<128;base+=16){
        int k=base+(lane%4)*2;
        unsigned a0=*reinterpret_cast<const unsigned*>(x+row*128+k);
        unsigned a1=*reinterpret_cast<const unsigned*>(x+(row+8)*128+k);
        unsigned a2=*reinterpret_cast<const unsigned*>(x+row*128+k+8);
        unsigned a3=*reinterpret_cast<const unsigned*>(x+(row+8)*128+k+8);
        bool zero=Skip && __all_sync(0xffffffffu,((a0|a1|a2|a3)&0x7fff7fffu)==0);
        if(!zero){
            unsigned b0=*reinterpret_cast<const unsigned*>(w+(col+lane/4)*128+k);
            unsigned b1=*reinterpret_cast<const unsigned*>(w+(col+lane/4)*128+k+8);
            mma(acc,a0,a1,a2,a3,b0,b1);
        }
    }
    int outcol=col+(lane%4)*2;
    y[row*n+outcol]=__float2bfloat16_rn(acc[0]+__bfloat162float(b[outcol]));
    y[row*n+outcol+1]=__float2bfloat16_rn(acc[1]+__bfloat162float(b[outcol+1]));
    y[(row+8)*n+outcol]=__float2bfloat16_rn(acc[2]+__bfloat162float(b[outcol]));
    y[(row+8)*n+outcol+1]=__float2bfloat16_rn(acc[3]+__bfloat162float(b[outcol+1]));
}

void forward(torch::Tensor x,torch::Tensor w,torch::Tensor b,torch::Tensor y,bool skip){
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()==2 && x.size(1)==128,"CUDA BF16 Mx128 required");
    for(const auto& t:{x,w,b,y})TORCH_CHECK(t.device()==x.device() && t.scalar_type()==x.scalar_type() && t.is_contiguous(),"Type/layout mismatch");
    int m=x.size(0),n=w.size(0);
    TORCH_CHECK(m>0 && m%64==0 && w.dim()==2 && w.size(1)==128 && (n==384 || n==512),"Fixed Pythia14M shapes, rows multiple of64");
    TORCH_CHECK(b.dim()==1 && b.numel()==n && y.dim()==2 && y.size(0)==m && y.size(1)==n,"Output/bias mismatch");
    c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
    #define P(t) reinterpret_cast<bf16*>(t.data_ptr())
    if(skip)project<true><<<dim3(m/64,n/8),128,0,stream>>>(P(x),P(w),P(b),P(y),m,n);
    else project<false><<<dim3(m/64,n/8),128,0,stream>>>(P(x),P(w),P(b),P(y),m,n);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
