// K030: match the measured B1 H4 T2048 D32 native two-split KV schedule.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#define FLASH_NAMESPACE run028_k030_flash
#define UNFUSE_FMA
#include "flash.h"
#include "sparse_gemm.h"
#include "flash_fwd_kernel.h"

namespace run028_k030_flash {
struct Params:Flash_fwd_params{int64_t* run028_stats;int64_t* prefix_stats;};
template<bool Skip,bool Count> struct Traits:Flash_fwd_kernel_traits<32,64,256,4,false,false,cutlass::bfloat16_t>{
    static constexpr bool Run028Skip=Skip,Run028Count=Count;
};
// Exact-grid, CTA-local prefix reduction. No preparation kernel or global prefix.
__device__ bool zero_query_partial(const Params& p){
    int tid=threadIdx.x,lane=tid&31,warp=tid/32;
    int head=blockIdx.z,split=blockIdx.y,block=blockIdx.x;
    auto qb=reinterpret_cast<const unsigned short*>(p.q_ptr);
    bool eligible=block*64>=split*1024;
    for(int i=tid;i<64*32;i+=128)eligible&=(qb[(head*2048+block*64)*32+i]&0x7fffu)==0;
    if(!__syncthreads_and(eligible))return false;
    auto vb=reinterpret_cast<const __nv_bfloat16*>(p.v_ptr);
    auto kb=reinterpret_cast<const unsigned short*>(p.k_ptr);
    auto oa=reinterpret_cast<float*>(p.oaccum_ptr);
    auto la=reinterpret_cast<float*>(p.softmax_lseaccum_ptr);
    bool safe=true;
    #pragma unroll 1
    for(int group=0;group<8;++group){
        int col=group*4+warp;float values[32];float total=0.;
        #pragma unroll
        for(int i=0;i<32;++i){
            int index=(head*2048+split*1024+lane*32+i)*32+col;
            float value=__bfloat162float(vb[index]),mag=fabsf(value);
            safe&=(mag==0.f || (mag>=.5f && mag<=64.f));
            safe&=(kb[index]&0x7f80u)!=0x7f80u;
            total+=value;values[i]=total;
        }
        float inclusive=total;
        #pragma unroll
        for(int delta=1;delta<32;delta*=2){
            float other=__shfl_up_sync(0xffffffffu,inclusive,delta);
            if(lane>=delta)inclusive+=other;
        }
        float previous=__shfl_up_sync(0xffffffffu,inclusive,1);
        float offset=lane==0?0.f:previous;
        int target=block*64-split*1024;
        if(target>=1024){
            float sum=__shfl_sync(0xffffffffu,inclusive,31);
            for(int r=lane;r<64;r+=32){
                int row=block*64+r;
                oa[((split*4+head)*2048+row)*32+col]=sum*__fdiv_rn(1.f,1024.f);
                if(col==0)la[(split*4+head)*2048+row]=__logf(1024.f);
            }
        }else{
            #pragma unroll
            for(int i=0;i<32;++i){
                int at=lane*32+i;
                if(at>=target && at<target+64){
                    int row=split*1024+at,n=at+1;
                    oa[((split*4+head)*2048+row)*32+col]=(values[i]+offset)*__fdiv_rn(1.f,float(n));
                    if(col==0)la[(split*4+head)*2048+row]=__logf(float(n));
                }
            }
        }
    }
    // Unsafe sums are never returned: the normal kernel overwrites every output.
    return __syncthreads_and(safe);
}
template<bool Skip,bool Count,bool Prefix>
__global__ void kernel(const __grid_constant__ Params p){
    int tid=threadIdx.x,head=blockIdx.z,split=blockIdx.y,block=blockIdx.x;
    int atom_offset=(((head*32+block)*2+split)*4+tid/32)*4;
    int prefix_offset=(((head*32+block)*2+split)*4+tid/32)*3;
    if constexpr(Count) if((tid&31)==0)for(int j=0;j<3;++j)p.prefix_stats[prefix_offset+j]=0;
    if constexpr(Skip && Prefix)if(zero_query_partial(p)){
        if constexpr(Count)if((tid&31)==0){
            for(int j=0;j<4;++j)p.run028_stats[atom_offset+j]=0;
            int nblocks=(min((block+1)*64-split*1024,1024)+255)/256;
            p.prefix_stats[prefix_offset]=64*nblocks;
            p.prefix_stats[prefix_offset+1]=64*nblocks;
            p.prefix_stats[prefix_offset+2]=16;
        }
        return;
    }
    compute_attn_splitkv<Traits<Skip,Count>,true,false,false,true,true,false,true,false>(p);
}
__global__ void combine(const __grid_constant__ Params p){
    combine_attn_seqk_parallel<Traits<false,false>,16,1,true>(p);
}
}

void run028_forward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor out,
                    torch::Tensor lse,torch::Tensor lse_accum,torch::Tensor o_accum,
                    torch::Tensor stats,torch::Tensor prefix_stats,double scale,bool skip,bool count,bool shortcut){
    TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.dim()==4,"CUDA BF16 BH TD");
    TORCH_CHECK(q.size(0)==1 && q.size(1)==4 && q.size(2)==2048 && q.size(3)==32,"B1 H4 T2048 D32 only");
    TORCH_CHECK(q.sizes()==k.sizes() && q.sizes()==v.sizes() && q.sizes()==out.sizes(),"QKV/out shape mismatch");
    for(const auto& x:{q,k,v,out})TORCH_CHECK(x.device()==q.device() && x.scalar_type()==q.scalar_type() && x.is_contiguous(),"BF16 layout/device mismatch");
    for(const auto& x:{lse,lse_accum,o_accum})TORCH_CHECK(x.is_contiguous() && x.device()==q.device() && x.scalar_type()==at::kFloat,"FP32 workspace mismatch");
    TORCH_CHECK(lse.numel()==4*2048 && lse_accum.numel()==2*4*2048 && o_accum.numel()==2*4*2048*32,"Workspace size mismatch");
    TORCH_CHECK(stats.is_contiguous() && stats.device()==q.device() && stats.scalar_type()==at::kLong && stats.numel()==4*32*2*4*4,"Counter capacity mismatch");
    TORCH_CHECK(prefix_stats.device()==q.device() && prefix_stats.scalar_type()==at::kLong && prefix_stats.is_contiguous() && prefix_stats.numel()==4*32*2*4*3,"Prefix counters");
    TORCH_CHECK(std::isfinite(scale) && scale>0,"Scale invalid");
    c10::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream();
    run028_k030_flash::Params p{};
    p.q_ptr=q.data_ptr();p.k_ptr=k.data_ptr();p.v_ptr=v.data_ptr();p.o_ptr=out.data_ptr();
    p.q_batch_stride=p.k_batch_stride=p.v_batch_stride=p.o_batch_stride=4*2048*32;
    p.q_row_stride=p.k_row_stride=p.v_row_stride=p.o_row_stride=32;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=2048*32;
    p.h=p.h_k=4;p.h_h_k_ratio=1;p.b=1;p.seqlen_q=p.seqlen_k=2048;p.d=p.d_rounded=32;
    p.seqlen_q_rounded=p.seqlen_k_rounded=p.total_q=2048;
    p.scale_softmax=float(scale);p.scale_softmax_log2=p.scale_softmax*M_LOG2E;
    p.p_dropout=p.rp_dropout=1.f;p.p_dropout_in_uint8_t=255;p.scale_softmax_rp_dropout=p.scale_softmax;
    p.is_bf16=p.is_causal=p.is_seqlens_k_cumulative=true;
    p.window_size_left=2048;p.window_size_right=0;p.num_splits=2;
    p.softmax_lse_ptr=lse.data_ptr();p.softmax_lseaccum_ptr=lse_accum.data_ptr();p.oaccum_ptr=o_accum.data_ptr();
    p.run028_stats=stats.data_ptr<int64_t>();p.prefix_stats=prefix_stats.data_ptr<int64_t>();
    constexpr int smem=run028_k030_flash::Traits<false,false>::kSmemSize;
    #define LAUNCH(S,C,P) run028_k030_flash::kernel<S,C,P><<<dim3(32,2,4),128,smem,stream>>>(p)
    if(skip && shortcut){if(count){LAUNCH(true,true,true);}else{LAUNCH(true,false,true);}}
    else if(skip){if(count){LAUNCH(true,true,false);}else{LAUNCH(true,false,false);}}
    else{if(count){LAUNCH(false,true,false);}else{LAUNCH(false,false,false);}}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    run028_k030_flash::combine<<<512,128,0,stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&run028_forward);}
