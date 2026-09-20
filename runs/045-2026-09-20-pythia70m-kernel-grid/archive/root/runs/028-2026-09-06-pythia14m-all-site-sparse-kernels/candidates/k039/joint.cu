// K039: M32N32 W2/Wo; 4 N atoms reuse A, with unchanged K16 order/rounding.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using bf16=__nv_bfloat16;

__device__ __forceinline__ unsigned gate_pair(unsigned v,float threshold,bool gate){
    if(!gate)return v;
    bf16 a=__ushort_as_bfloat16(v&65535),b=__ushort_as_bfloat16(v>>16);
    if(__bfloat162float(a)<threshold)a=__float2bfloat16_rn(0.f);
    if(__bfloat162float(b)<threshold)b=__float2bfloat16_rn(0.f);
    return unsigned(__bfloat16_as_ushort(a)) | (unsigned(__bfloat16_as_ushort(b))<<16);
}

template<int K,bool Skip,int NAtoms>
__device__ void accumulate(const bf16* x,const bf16* w,int row,int col,int lane,float threshold,bool gate,float (&c)[NAtoms][4]){
    #pragma unroll
    for(int base=0;base<K;base+=16){
        int k=base+(lane%4)*2;
        unsigned a0=gate_pair(*reinterpret_cast<const unsigned*>(x+row*K+k),threshold,gate);
        unsigned a1=gate_pair(*reinterpret_cast<const unsigned*>(x+(row+8)*K+k),threshold,gate);
        unsigned a2=gate_pair(*reinterpret_cast<const unsigned*>(x+row*K+k+8),threshold,gate);
        unsigned a3=gate_pair(*reinterpret_cast<const unsigned*>(x+(row+8)*K+k+8),threshold,gate);
        bool zero=Skip && __all_sync(0xffffffffu,((a0|a1|a2|a3)&0x7fff7fffu)==0);
        if(!zero){
            #pragma unroll
            for(int atom=0;atom<NAtoms;++atom){
            unsigned b0=*reinterpret_cast<const unsigned*>(w+(col+atom*8+lane/4)*K+k);
            unsigned b1=*reinterpret_cast<const unsigned*>(w+(col+atom*8+lane/4)*K+k+8);
            asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                : "+f"(c[atom][0]),"+f"(c[atom][1]),"+f"(c[atom][2]),"+f"(c[atom][3])
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
        }
    }
}

template<bool Skip>
__global__ void joint(const bf16* h,const bf16* z,const bf16* wh,const bf16* wz,
 const bf16* bh,const bf16* bz,const bf16* residual,bf16* out,float th,float tz,bool gh,bool gz){
    int lane=threadIdx.x&31,warp=threadIdx.x/32;
    int row=blockIdx.x*32+warp*16+lane/4,col=blockIdx.y*32;
    float ah[4][4]={},az[4][4]={};
    accumulate<512,Skip,4>(h,wh,row,col,lane,th,gh,ah);
    accumulate<128,Skip,4>(z,wz,row,col,lane,tz,gz,az);
    #pragma unroll
    for(int atom=0;atom<4;++atom){
    #pragma unroll
    for(int e=0;e<4;++e){
        int r=row+(e/2)*8,c=col+atom*8+(lane%4)*2+e%2;
        float hv=__bfloat162float(__float2bfloat16_rn(ah[atom][e]+__bfloat162float(bh[c])));
        float zv=__bfloat162float(__float2bfloat16_rn(az[atom][e]+__bfloat162float(bz[c])));
        float sum=__bfloat162float(__float2bfloat16_rn(hv+zv));
        out[r*128+c]=__float2bfloat16_rn(sum+__bfloat162float(residual[r*128+c]));
    }
    }
}

void forward(torch::Tensor h,torch::Tensor z,torch::Tensor wh,torch::Tensor wz,
 torch::Tensor bh,torch::Tensor bz,torch::Tensor residual,torch::Tensor out,double th,double tz,bool gh,bool gz,bool skip){
    TORCH_CHECK(h.is_cuda() && h.scalar_type()==at::kBFloat16 && h.dim()==2,"CUDA BF16 required");
    for(const auto& t:{h,z,wh,wz,bh,bz,residual,out})TORCH_CHECK(t.device()==h.device() && t.scalar_type()==h.scalar_type() && t.is_contiguous(),"Type/layout mismatch");
    int m=h.size(0);
    TORCH_CHECK(m>0 && m%32==0 && h.size(1)==512 && z.sizes()==at::IntArrayRef({m,128}),"M multiple32, H512 Z128 required");
    TORCH_CHECK(wh.sizes()==at::IntArrayRef({128,512}) && wz.sizes()==at::IntArrayRef({128,128}),"Weight shape mismatch");
    TORCH_CHECK(bh.numel()==128 && bz.numel()==128 && residual.sizes()==z.sizes() && out.sizes()==z.sizes(),"Output shape mismatch");
    TORCH_CHECK(std::isfinite(th) && std::isfinite(tz) && th>=0 && tz>=0,"Invalid gate");
    c10::cuda::CUDAGuard guard(h.device());auto stream=at::cuda::getCurrentCUDAStream();
    #define P(t) reinterpret_cast<bf16*>(t.data_ptr())
    if(skip)joint<true><<<dim3(m/32,4),64,0,stream>>>(P(h),P(z),P(wh),P(wz),P(bh),P(bz),P(residual),P(out),th,tz,gh,gz);
    else joint<false><<<dim3(m/32,4),64,0,stream>>>(P(h),P(z),P(wh),P(wz),P(bh),P(bz),P(residual),P(out),th,tz,gh,gz);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
