// K026: fixed D32 Flash Attention with optional zero-MMA-atom bypass.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#define FLASH_NAMESPACE run028_k026_flash
#define UNFUSE_FMA
#include "flash.h"
#include "sparse_gemm.h"
#include "flash_fwd_kernel.h"

namespace run028_k026_flash {
struct Params:Flash_fwd_params{int64_t* run028_stats;};
template<bool Skip,bool Count> struct Traits:Flash_fwd_kernel_traits<32,128,128,4,false,false,cutlass::bfloat16_t>{
    static constexpr bool Run028Skip=Skip,Run028Count=Count;
};
template<bool Skip,bool Count,bool Even>
__global__ void kernel(const __grid_constant__ Params p){
    compute_attn<Traits<Skip,Count>,false,true,false,false,Even,true,false,false>(p);
}
}

void run028_forward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor out,
             torch::Tensor lse,torch::Tensor stats,double scale,bool skip,bool count){
    TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.dim()==4,"CUDA BF16 BH TD");
    TORCH_CHECK(q.size(0)==1 && q.size(3)==32 && q.size(2)>0 && q.size(2)<=2048,"B1 D32 T<=2048");
    TORCH_CHECK(q.sizes()==k.sizes() && q.sizes()==v.sizes() && q.sizes()==out.sizes(),"QKV/out shape mismatch");
    for(const auto& x:{q,k,v,out})TORCH_CHECK(x.device()==q.device() && x.scalar_type()==q.scalar_type() && x.is_contiguous(),"BF16 layout/device mismatch");
    int t=q.size(2),h=q.size(1),tiles=(t+127)/128;
    TORCH_CHECK(lse.is_contiguous() && lse.device()==q.device() && lse.scalar_type()==at::kFloat && lse.numel()==h*t,"LSE mismatch");
    TORCH_CHECK(stats.is_contiguous() && stats.device()==q.device() && stats.scalar_type()==at::kLong && stats.numel()==h*tiles*4*4,"counter capacity mismatch");
    TORCH_CHECK(std::isfinite(scale) && scale>0,"Scale invalid");
    c10::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream();
    run028_k026_flash::Params p{};
    p.q_ptr=q.data_ptr();p.k_ptr=k.data_ptr();p.v_ptr=v.data_ptr();p.o_ptr=out.data_ptr();
    p.q_batch_stride=p.k_batch_stride=p.v_batch_stride=p.o_batch_stride=h*t*32;
    p.q_row_stride=p.k_row_stride=p.v_row_stride=p.o_row_stride=32;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=t*32;
    p.h=p.h_k=h;p.h_h_k_ratio=1;p.b=1;p.seqlen_q=p.seqlen_k=t;p.d=p.d_rounded=32;
    p.seqlen_q_rounded=p.seqlen_k_rounded=tiles*128;p.total_q=t;
    p.scale_softmax=float(scale);p.scale_softmax_log2=p.scale_softmax*M_LOG2E;
    p.p_dropout=p.rp_dropout=1.f;p.p_dropout_in_uint8_t=255;p.scale_softmax_rp_dropout=p.scale_softmax;
    p.is_bf16=p.is_causal=p.is_seqlens_k_cumulative=true;
    p.window_size_left=-1;p.window_size_right=0;p.num_splits=1;
    p.softmax_lse_ptr=lse.data_ptr();p.run028_stats=stats.data_ptr<int64_t>();
    constexpr int smem=run028_k026_flash::Traits<false,false>::kSmemSize;
    #define LAUNCH(S,C,E) run028_k026_flash::kernel<S,C,E><<<dim3(tiles,1,h),128,smem,stream>>>(p)
    if(t%128==0){
        if(skip){if(count){LAUNCH(true,true,true);}else{LAUNCH(true,false,true);}}
        else{if(count){LAUNCH(false,true,true);}else{LAUNCH(false,false,true);}}
    }else{
        if(skip){if(count){LAUNCH(true,true,false);}else{LAUNCH(true,false,false);}}
        else{if(count){LAUNCH(false,true,false);}else{LAUNCH(false,false,false);}}
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&run028_forward);}
