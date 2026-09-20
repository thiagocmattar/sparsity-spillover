// K027: match the measured B1 H4 T2048 D32 native two-split KV schedule.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#define FLASH_NAMESPACE run028_k027_flash
#define UNFUSE_FMA
#include "flash.h"
#include "sparse_gemm.h"
#include "flash_fwd_kernel.h"

namespace run028_k027_flash {
struct Params:Flash_fwd_params{int64_t* run028_stats;};
template<bool Skip,bool Count> struct Traits:Flash_fwd_kernel_traits<32,64,256,4,false,false,cutlass::bfloat16_t>{
    static constexpr bool Run028Skip=Skip,Run028Count=Count;
};
template<bool Skip,bool Count>
__global__ void kernel(const __grid_constant__ Params p){
    compute_attn_splitkv<Traits<Skip,Count>,true,false,false,true,true,false,true,false>(p);
}
__global__ void combine(const __grid_constant__ Params p){
    combine_attn_seqk_parallel<Traits<false,false>,16,1,true>(p);
}
}

void run028_forward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor out,
                    torch::Tensor lse,torch::Tensor lse_accum,torch::Tensor o_accum,
                    torch::Tensor stats,double scale,bool skip,bool count){
    TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.dim()==4,"CUDA BF16 BH TD");
    TORCH_CHECK(q.size(0)==1 && q.size(1)==4 && q.size(2)==2048 && q.size(3)==32,"B1 H4 T2048 D32 only");
    TORCH_CHECK(q.sizes()==k.sizes() && q.sizes()==v.sizes() && q.sizes()==out.sizes(),"QKV/out shape mismatch");
    for(const auto& x:{q,k,v,out})TORCH_CHECK(x.device()==q.device() && x.scalar_type()==q.scalar_type() && x.is_contiguous(),"BF16 layout/device mismatch");
    for(const auto& x:{lse,lse_accum,o_accum})TORCH_CHECK(x.is_contiguous() && x.device()==q.device() && x.scalar_type()==at::kFloat,"FP32 workspace mismatch");
    TORCH_CHECK(lse.numel()==4*2048 && lse_accum.numel()==2*4*2048 && o_accum.numel()==2*4*2048*32,"Workspace size mismatch");
    TORCH_CHECK(stats.is_contiguous() && stats.device()==q.device() && stats.scalar_type()==at::kLong && stats.numel()==4*32*2*4*4,"Counter capacity mismatch");
    TORCH_CHECK(std::isfinite(scale) && scale>0,"Scale invalid");
    c10::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream();
    run028_k027_flash::Params p{};
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
    p.run028_stats=stats.data_ptr<int64_t>();
    constexpr int smem=run028_k027_flash::Traits<false,false>::kSmemSize;
    #define LAUNCH(S,C) run028_k027_flash::kernel<S,C><<<dim3(32,2,4),128,smem,stream>>>(p)
    if(skip){if(count){LAUNCH(true,true);}else{LAUNCH(true,false);}}
    else{if(count){LAUNCH(false,true);}else{LAUNCH(false,false);}}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    run028_k027_flash::combine<<<512,128,0,stream>>>(p);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&run028_forward);}
