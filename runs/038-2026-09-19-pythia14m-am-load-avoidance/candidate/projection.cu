// K049-derived M8/K16 hybrid for standalone a/m projections, K128 N384/N512.
// Eight real rows use a padded M16 MMA; the dense control keeps this layout.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using bf16=__nv_bfloat16;
struct TinyRow { int count=0,i0=0,i1=0; float v0=0.f,v1=0.f; unsigned tiles=0; };

__device__ __forceinline__ TinyRow inspect(const bf16* x,int row,int lane){
    TinyRow out;
    #pragma unroll
    for(int base=0;base<128;base+=32){
        float value=__bfloat162float(x[row*128+base+lane]);
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
template<int N>
__device__ __forceinline__ bf16 short_linear(TinyRow a,const bf16* wt,const bf16* bias,int col){
    float value=0.f;
    if(a.count>0)value=__fmul_rn(a.v0,__bfloat162float(wt[a.i0*N+col]));
    if(a.count==2)value=__fmaf_rn(a.v1,__bfloat162float(wt[a.i1*N+col]),value);
    return __float2bfloat16_rn(__fadd_rn(value,__bfloat162float(bias[col])));
}
template<int N,bool Skip,bool Count>
__global__ void project(const bf16* x,const bf16* w,const bf16* wt,const bf16* bias,
                       bf16* out,long long* stats,bool fast_weights){
    __shared__ bool complex[8];
    __shared__ unsigned masks[8];
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    int first=blockIdx.x*8,first_col=blockIdx.y*128;
    bool hybrid=Skip && fast_weights;
    unsigned active=0xffu;
    int issued=0,bypassed=0,scalar=0;
    if(hybrid){
        auto a=inspect(x,first+warp,lane);
        bool is_complex=a.count>2;
        if(lane==0){complex[warp]=is_complex;masks[warp]=is_complex?a.tiles:0;}
        if(!is_complex){
            scalar=a.count*128;
            #pragma unroll
            for(int e=0;e<4;++e){int col=first_col+lane*4+e;
                out[(first+warp)*N+col]=short_linear<N>(a,wt,bias,col);
            }
        }
        __syncthreads();
        active=0;for(int i=0;i<8;++i)active|=masks[i];
    }
    if(warp<4){
        int local=lane/4,row=first+local,col=first_col+warp*32;
        float acc[4][4]={};
        unsigned pending=Skip?active:0xffu;
        if constexpr(Count)bypassed=(8-__popc(pending))*4;
        while(pending){
            int base=(__ffs(pending)-1)*16;pending&=pending-1;
            int k=base+(lane%4)*2;
            bool use=!hybrid || complex[local];
            unsigned a0=use?*reinterpret_cast<const unsigned*>(x+row*128+k):0;
            unsigned a2=use?*reinterpret_cast<const unsigned*>(x+row*128+k+8):0;
            if constexpr(Skip){
                if(!hybrid && __all_sync(0xffffffffu,((a0|a2)&0x7fff7fffu)==0)){
                    if constexpr(Count)bypassed+=4;
                    continue;
                }
            }
            #pragma unroll
            for(int atom=0;atom<4;++atom){
                unsigned b0=*reinterpret_cast<const unsigned*>(w+(col+atom*8+lane/4)*128+k);
                unsigned b1=*reinterpret_cast<const unsigned*>(w+(col+atom*8+lane/4)*128+k+8);
                unsigned zero=0;
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[atom][0]),"+f"(acc[atom][1]),"+f"(acc[atom][2]),"+f"(acc[atom][3])
                    : "r"(a0),"r"(zero),"r"(a2),"r"(zero),"r"(b0),"r"(b1));
            }
            if constexpr(Count)issued+=4;
        }
        if(!hybrid || complex[local]){
            #pragma unroll
            for(int atom=0;atom<4;++atom){
                #pragma unroll
                for(int e=0;e<2;++e){int c=col+atom*8+(lane%4)*2+e;
                    out[row*N+c]=__float2bfloat16_rn(acc[atom][e]+__bfloat162float(bias[c]));
                }
            }
        }
    }
    if constexpr(Count){if(lane==0){
        int slot=((blockIdx.y*gridDim.x+blockIdx.x)*8+warp)*3;
        stats[slot]=issued;stats[slot+1]=bypassed;stats[slot+2]=scalar;
    }}
}
void forward(torch::Tensor x,torch::Tensor w,torch::Tensor wt,torch::Tensor b,
             torch::Tensor out,torch::Tensor stats,bool skip,bool fast_weights,bool count){
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()==2 && x.size(1)==128,"CUDA BF16 Mx128");
    for(const auto& t:{x,w,wt,b,out})TORCH_CHECK(t.device()==x.device() && t.scalar_type()==x.scalar_type() && t.is_contiguous(),"Type/layout mismatch");
    int m=x.size(0),n=w.size(0);
    TORCH_CHECK(m>0 && m%8==0 && w.sizes()==at::IntArrayRef({n,128}) && (n==384 || n==512),"M8 N384/N512");
    TORCH_CHECK(wt.sizes()==at::IntArrayRef({128,n}) && b.numel()==n && out.sizes()==at::IntArrayRef({m,n}),"Output/weight shape");
    TORCH_CHECK(stats.device()==x.device() && stats.is_contiguous() && stats.scalar_type()==at::kLong && stats.numel()==(m/8)*(n/128)*8*3,"Counter shape");
    c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
    #define P(t) reinterpret_cast<bf16*>(t.data_ptr())
    #define LAUNCH(N,S,C) project<N,S,C><<<dim3(m/8,n/128),256,0,stream>>>(P(x),P(w),P(wt),P(b),P(out),reinterpret_cast<long long*>(stats.data_ptr()),fast_weights)
    #define SELECT(N) if(skip){if(count){LAUNCH(N,true,true);}else{LAUNCH(N,true,false);}}else{if(count){LAUNCH(N,false,true);}else{LAUNCH(N,false,false);}}
    if(n==384){SELECT(384);}else{SELECT(512);}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
