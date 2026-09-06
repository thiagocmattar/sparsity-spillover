// Exact short-row SIMT plus native-order MMA for the remaining h/z rows.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
using bf16=__nv_bfloat16;

struct TinyRow { int count=3,i0=0,i1=0; float v0=0.f,v1=0.f; };
__device__ __forceinline__ float gated(const bf16* x,int index,float threshold,bool gate){
    float value=__bfloat162float(x[index]);return gate && value<threshold?0.f:value;
}
template<int K>
__device__ __forceinline__ TinyRow inspect(const bf16* x,int row,int lane,float threshold,bool gate){
    TinyRow out;out.count=0;
    #pragma unroll
    for(int base=0;base<K;base+=32){
        float value=gated(x,row*K+base+lane,threshold,gate);
        unsigned mask=__ballot_sync(0xffffffffu,value!=0.f);
        if(out.count+__popc(mask)>2)return TinyRow{};
        while(mask){
            int selected=__ffs(mask)-1;
            float next=__shfl_sync(0xffffffffu,value,selected);
            if(!(fabsf(next)>=0x1p-50f && fabsf(next)<=0x1p50f))return TinyRow{};
            if(out.count==0){out.i0=base+selected;out.v0=next;}
            else{out.i1=base+selected;out.v1=next;}
            ++out.count;mask&=mask-1;
        }
    }
    return out;
}
__device__ __forceinline__ bf16 short_linear(TinyRow a,const bf16* wt,const bf16* bias,int col){
    float value=0.f;
    if(a.count>0)value=__fmul_rn(a.v0,__bfloat162float(wt[a.i0*128+col]));
    if(a.count==2)value=__fmaf_rn(a.v1,__bfloat162float(wt[a.i1*128+col]),value);
    return __float2bfloat16_rn(__fadd_rn(value,__bfloat162float(bias[col])));
}
__device__ __forceinline__ unsigned gate_pair(unsigned value,float threshold,bool gate){
    if(!gate)return value;
    bf16 a=__ushort_as_bfloat16(value&65535),b=__ushort_as_bfloat16(value>>16);
    if(__bfloat162float(a)<threshold)a=__float2bfloat16_rn(0.f);
    if(__bfloat162float(b)<threshold)b=__float2bfloat16_rn(0.f);
    return unsigned(__bfloat16_as_ushort(a))|(unsigned(__bfloat16_as_ushort(b))<<16);
}
template<int K,bool Skip,bool Count>
__device__ __forceinline__ void accumulate(const bf16* x,const bf16* w,int row,int col,int lane,
    float threshold,bool gate,bool hybrid,const bool* complex,int local_row,float (&acc)[4][4],int& issued,int& bypassed){
    #pragma unroll
    for(int base=0;base<K;base+=16){
        int k=base+(lane%4)*2;
        bool low=!hybrid || complex[local_row],high=false;
        unsigned a0=low?gate_pair(*reinterpret_cast<const unsigned*>(x+row*K+k),threshold,gate):0;
        unsigned a1=high?gate_pair(*reinterpret_cast<const unsigned*>(x+(row+8)*K+k),threshold,gate):0;
        unsigned a2=low?gate_pair(*reinterpret_cast<const unsigned*>(x+row*K+k+8),threshold,gate):0;
        unsigned a3=high?gate_pair(*reinterpret_cast<const unsigned*>(x+(row+8)*K+k+8),threshold,gate):0;
        bool zero=Skip && __all_sync(0xffffffffu,((a0|a1|a2|a3)&0x7fff7fffu)==0);
        if(!zero){
            #pragma unroll
            for(int atom=0;atom<4;++atom){
                unsigned b0=*reinterpret_cast<const unsigned*>(w+(col+atom*8+lane/4)*K+k);
                unsigned b1=*reinterpret_cast<const unsigned*>(w+(col+atom*8+lane/4)*K+k+8);
                asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(acc[atom][0]),"+f"(acc[atom][1]),"+f"(acc[atom][2]),"+f"(acc[atom][3])
                    : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
            }
            if constexpr(Count)issued+=4;
        }else if constexpr(Count)bypassed+=4;
    }
}
template<bool Count>
__device__ __forceinline__ void counters(long long* stats,int warp,int lane,int hi,int hb,int zi,int zb,int hs,int zs){
    if constexpr(Count){if(lane==0){int slot=(blockIdx.x*8+warp)*6;
        stats[slot]=hi;stats[slot+1]=hb;stats[slot+2]=zi;stats[slot+3]=zb;stats[slot+4]=hs;stats[slot+5]=zs;}}
}
template<bool Skip,bool Count>
__global__ void joint(const bf16* h,const bf16* z,const bf16* wh,const bf16* wz,const bf16* wht,const bf16* wzt,
    const bf16* bh,const bf16* bz,const bf16* residual,bf16* out,long long* stats,float th,float tz,bool gh,bool gz,bool fast_weights){
    __shared__ bool hc[8],zc[8];
    __shared__ bf16 hs[8*128],zs[8*128];
    __shared__ unsigned complex_masks[8];
    int lane=threadIdx.x&31,warp=threadIdx.x/32,first_row=blockIdx.x*8;
    int h_scalar=0,z_scalar=0;bool hybrid=Skip && fast_weights;
    if(hybrid){
        unsigned complex_mask=0;
        for(int local=warp;local<8;local+=8){
            int row=first_row+local;auto ha=inspect<512>(h,row,lane,th,gh);auto za=inspect<128>(z,row,lane,tz,gz);
            bool h_complex=ha.count>2,z_complex=za.count>2;
            if(lane==0){hc[local]=h_complex;zc[local]=z_complex;}
            if(h_complex || z_complex)complex_mask|=1u<<local;
            if constexpr(Count){if(!h_complex)h_scalar+=ha.count*128;if(!z_complex)z_scalar+=za.count*128;}
            #pragma unroll
            for(int e=0;e<4;++e){
                int col=lane*4+e,slot=local*128+lane*4+e;bf16 hv,zv;
                if(!h_complex){hv=short_linear(ha,wht,bh,col);hs[slot]=hv;}
                if(!z_complex){zv=short_linear(za,wzt,bz,col);zs[slot]=zv;}
                if(!h_complex && !z_complex){
                    float sum=__bfloat162float(__float2bfloat16_rn(__bfloat162float(hv)+__bfloat162float(zv)));
                    out[row*128+col]=__float2bfloat16_rn(sum+__bfloat162float(residual[row*128+col]));
                }
            }
        }
        if(lane==0)complex_masks[warp]=complex_mask;
        __syncthreads();
        unsigned mask=0;for(int i=0;i<8;++i)mask|=complex_masks[i];
        if(mask==0){counters<Count>(stats,warp,lane,0,warp<4?128:0,0,warp<4?32:0,h_scalar,z_scalar);return;}
    }
    if(warp>=4){counters<Count>(stats,warp,lane,0,0,0,0,h_scalar,z_scalar);return;}
    int local_row=lane/4,row=first_row+local_row,col=warp*32;
    float ah[4][4]={},az[4][4]={};int hi=0,hb=0,zi=0,zb=0;
    accumulate<512,Skip,Count>(h,wh,row,col,lane,th,gh,hybrid,hc,local_row,ah,hi,hb);
    accumulate<128,Skip,Count>(z,wz,row,col,lane,tz,gz,hybrid,zc,local_row,az,zi,zb);
    #pragma unroll
    for(int atom=0;atom<4;++atom){
        #pragma unroll
        for(int e=0;e<2;++e){
            int r=row+(e/2)*8,lr=local_row+(e/2)*8,c=col+atom*8+(lane%4)*2+e%2,slot=lr*128+c;
            if(!hybrid || hc[lr] || zc[lr]){
                float hv=hybrid && !hc[lr]?__bfloat162float(hs[slot]):__bfloat162float(__float2bfloat16_rn(ah[atom][e]+__bfloat162float(bh[c])));
                float zv=hybrid && !zc[lr]?__bfloat162float(zs[slot]):__bfloat162float(__float2bfloat16_rn(az[atom][e]+__bfloat162float(bz[c])));
                float sum=__bfloat162float(__float2bfloat16_rn(hv+zv));
                out[r*128+c]=__float2bfloat16_rn(sum+__bfloat162float(residual[r*128+c]));
            }
        }
    }
    counters<Count>(stats,warp,lane,hi,hb,zi,zb,h_scalar,z_scalar);
}
void forward(torch::Tensor h,torch::Tensor z,torch::Tensor wh,torch::Tensor wz,torch::Tensor wht,torch::Tensor wzt,
    torch::Tensor bh,torch::Tensor bz,torch::Tensor residual,torch::Tensor out,torch::Tensor stats,
    double th,double tz,bool gh,bool gz,bool skip,bool fast_weights,bool count){
    TORCH_CHECK(h.is_cuda() && h.scalar_type()==at::kBFloat16 && h.dim()==2,"CUDA BF16 required");
    for(const auto& tensor:{h,z,wh,wz,wht,wzt,bh,bz,residual,out})TORCH_CHECK(tensor.device()==h.device() && tensor.scalar_type()==h.scalar_type() && tensor.is_contiguous(),"Type/layout mismatch");
    int m=h.size(0);
    TORCH_CHECK(m>0 && m%8==0 && h.size(1)==512 && z.sizes()==at::IntArrayRef({m,128}),"M8 H512 Z128 required");
    TORCH_CHECK(wh.sizes()==at::IntArrayRef({128,512}) && wz.sizes()==at::IntArrayRef({128,128}),"Weight shape");
    TORCH_CHECK(wht.sizes()==at::IntArrayRef({512,128}) && wzt.sizes()==at::IntArrayRef({128,128}),"Transposed weight shape");
    TORCH_CHECK(bh.numel()==128 && bz.numel()==128 && residual.sizes()==z.sizes() && out.sizes()==z.sizes(),"Output shape");
    TORCH_CHECK(stats.device()==h.device() && stats.scalar_type()==at::kLong && stats.is_contiguous() && stats.numel()==(m/8)*8*6,"Counter shape");
    TORCH_CHECK(std::isfinite(th) && std::isfinite(tz) && th>=0 && tz>=0,"Gate threshold");
    c10::cuda::CUDAGuard guard(h.device());auto stream=at::cuda::getCurrentCUDAStream();
    #define P(t) reinterpret_cast<bf16*>(t.data_ptr())
    #define LAUNCH(S,C) joint<S,C><<<dim3(m/8),256,0,stream>>>(P(h),P(z),P(wh),P(wz),P(wht),P(wzt),P(bh),P(bz),P(residual),P(out),reinterpret_cast<long long*>(stats.data_ptr()),th,tz,gh,gz,fast_weights)
    if(skip){if(count){LAUNCH(true,true);}else{LAUNCH(true,false);}}
    else{if(count){LAUNCH(false,true);}else{LAUNCH(false,false);}}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
