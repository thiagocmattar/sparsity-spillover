// Run063: exact Run060 corner gate, K049 M8/K16 arithmetic, corner-only support.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <cmath>
using bf16=__nv_bfloat16;

__device__ __forceinline__ unsigned long long tick(){
    unsigned long long value; asm volatile("mov.u64 %0, %%clock64;" : "=l"(value) :: "memory"); return value;
}
__device__ __forceinline__ void save_ticks(long long* ticks,int warp,int lane,unsigned long long a,unsigned long long b,unsigned long long c,unsigned long long d,unsigned long long e){
    if(lane==0 && warp<4){int slot=(blockIdx.x*4+warp)*5;ticks[slot]=a;ticks[slot+1]=b;ticks[slot+2]=c;ticks[slot+3]=d;ticks[slot+4]=e;}
}


template<int K>
__device__ __forceinline__ unsigned corners(const bf16* x,int first_row,int lane,float threshold){
    bool admitted=false;
    if(lane<K/16)admitted=__bfloat162float(x[first_row*K+lane*16])>threshold;
    return __ballot_sync(0xffffffffu,admitted);
}

template<int K,bool Skip,bool Count>
__device__ __forceinline__ void accumulate(const bf16* x,const bf16* w,int row,int col,int lane,
    unsigned live,float (&acc)[4][4],int& issued,int& bypassed){
    unsigned pending=Skip?live:(K==512?0xffffffffu:0xffu);
    if constexpr(Count)bypassed+=(K/16-__popc(pending))*4;
    while(pending){
        int tile=__ffs(pending)-1;pending&=pending-1;
        int k=tile*16+(lane%4)*2;
        bool admitted=bool(live&(1u<<tile));
        unsigned a0=admitted?*reinterpret_cast<const unsigned*>(x+row*K+k):0u;
        unsigned a2=admitted?*reinterpret_cast<const unsigned*>(x+row*K+k+8):0u;
        // M8 operands occupy the lower half of an M16 MMA, exactly as Run061.
        unsigned a1=0u,a3=0u;
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
    }
}

template<bool Skip,bool Count>
__global__ void joint(const bf16* h,const bf16* z,const bf16* wh,const bf16* wz,
    const bf16* bh,const bf16* bz,const bf16* residual,bf16* out,long long* stats,float th,float tz,long long* ticks){
    int lane=threadIdx.x&31,warp=threadIdx.x/32,first_row=blockIdx.x*8;
    unsigned long long t0=tick();
    unsigned hm=corners<512>(h,first_row,lane,th),zm=corners<128>(z,first_row,lane,tz);
    int row=first_row+lane/4,col=warp*32;
    float ah[4][4]={},az[4][4]={};int hi=0,hb=0,zi=0,zb=0;
    unsigned long long t1=tick();
    accumulate<512,Skip,Count>(h,wh,row,col,lane,hm,ah,hi,hb);
    unsigned long long t2=tick();
    accumulate<128,Skip,Count>(z,wz,row,col,lane,zm,az,zi,zb);
    unsigned long long t3=tick();
    #pragma unroll
    for(int atom=0;atom<4;++atom){
        #pragma unroll
        for(int e=0;e<2;++e){
            int c=col+atom*8+(lane%4)*2+e;
            // Even an all-dead block contributes both biases, with the original
            // ordered BF16 branch, branch-sum, and residual rounding.
            float hv=__bfloat162float(__float2bfloat16_rn(ah[atom][e]+__bfloat162float(bh[c])));
            float zv=__bfloat162float(__float2bfloat16_rn(az[atom][e]+__bfloat162float(bz[c])));
            float sum=__bfloat162float(__float2bfloat16_rn(hv+zv));
            out[row*128+c]=__float2bfloat16_rn(sum+__bfloat162float(residual[row*128+c]));
        }
    }
    save_ticks(ticks,warp,lane,t0,t1,t2,t3,tick());
    if constexpr(Count){if(lane==0){
        int slot=(blockIdx.x*8+warp)*6;
        stats[slot]=hi;stats[slot+1]=hb;stats[slot+2]=zi;stats[slot+3]=zb;stats[slot+4]=0;stats[slot+5]=0;
        #pragma unroll
        for(int j=0;j<6;++j)stats[slot+4*6+j]=0;
    }}
}

void forward(torch::Tensor h,torch::Tensor z,torch::Tensor wh,torch::Tensor wz,torch::Tensor wht,torch::Tensor wzt,
    torch::Tensor bh,torch::Tensor bz,torch::Tensor residual,torch::Tensor out,torch::Tensor stats,
    double th,double tz,int gh,int gz,bool skip,bool fast_weights,bool count,torch::Tensor ticks){
    TORCH_CHECK(h.is_cuda() && h.scalar_type()==at::kBFloat16 && h.dim()==2,"CUDA BF16 required");
    for(const auto& t:{h,z,wh,wz,wht,wzt,bh,bz,residual,out})TORCH_CHECK(t.device()==h.device() && t.scalar_type()==h.scalar_type() && t.is_contiguous(),"Type/layout mismatch");
    int m=h.size(0);
    TORCH_CHECK(m>0 && m%8==0 && h.size(1)==512 && z.sizes()==at::IntArrayRef({m,128}),"M8 H512 Z128 required");
    TORCH_CHECK(wh.sizes()==at::IntArrayRef({128,512}) && wz.sizes()==at::IntArrayRef({128,128}),"Weight shape");
    TORCH_CHECK(wht.sizes()==at::IntArrayRef({512,128}) && wzt.sizes()==at::IntArrayRef({128,128}),"Transposed weight shape");
    TORCH_CHECK(bh.numel()==128 && bz.numel()==128 && residual.sizes()==z.sizes() && out.sizes()==z.sizes(),"Output shape");
    TORCH_CHECK(stats.device()==h.device() && stats.scalar_type()==at::kLong && stats.is_contiguous() && stats.numel()==(m/8)*8*6,"Counter shape");
    TORCH_CHECK(gh==2 && gz==2,"Only the unchanged corner8x16 gate is supported");
    TORCH_CHECK(std::isfinite(th) && std::isfinite(tz) && th>=0 && tz>=0,"Gate threshold");
    c10::cuda::CUDAGuard guard(h.device());auto stream=at::cuda::getCurrentCUDAStream();
    #define P(t) reinterpret_cast<bf16*>(t.data_ptr())
    #define LAUNCH(S,C) joint<S,C><<<dim3(m/8),128,0,stream>>>(P(h),P(z),P(wh),P(wz),P(bh),P(bz),P(residual),P(out),reinterpret_cast<long long*>(stats.data_ptr()),th,tz,reinterpret_cast<long long*>(ticks.data_ptr()))
    if(skip){if(count){LAUNCH(true,true);}else{LAUNCH(true,false);}}
    else{if(count){LAUNCH(false,true);}else{LAUNCH(false,false);}}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
