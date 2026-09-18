// K035-derived B1 H8 T2048 D64 port. Native128x128 causal schedule, no KV splitting; exact-zero MMA bypass.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#define FLASH_NAMESPACE run035_flash
#define UNFUSE_FMA
#include "flash.h"
#include "sparse_gemm.h"
#include "flash_fwd_kernel.h"

namespace run035_flash {
struct Params:Flash_fwd_params{int64_t* run028_stats;};
template<bool Skip,bool Count> struct Traits:Flash_fwd_kernel_traits<64,128,128,4,false,false,cutlass::bfloat16_t>{
    static constexpr bool Run028Skip=Skip,Run028Count=Count;
};
template<bool Skip,bool Count>
__global__ void kernel(const __grid_constant__ Params p){
    compute_attn<Traits<Skip,Count>,false,true,false,false,true,true,false,false>(p);
}
template<bool Skip,bool Count> void launch(Params p,cudaStream_t stream){
    constexpr int smem=Traits<Skip,Count>::kSmemSize;
    if constexpr(smem>=48*1024){
        C10_CUDA_CHECK(cudaFuncSetAttribute(kernel<Skip,Count>,cudaFuncAttributeMaxDynamicSharedMemorySize,smem));
    }
    kernel<Skip,Count><<<dim3(16,1,8),128,smem,stream>>>(p);
}
}

void run035_forward(torch::Tensor q,torch::Tensor k,torch::Tensor v,torch::Tensor out,
    torch::Tensor lse,torch::Tensor lse_accum,torch::Tensor o_accum,
    torch::Tensor stats,torch::Tensor prefix,torch::Tensor safe,torch::Tensor prefix_stats,
    double scale,bool skip,bool count,bool shortcut){
    TORCH_CHECK(!shortcut,"Frozen 70M no-prefix policy required");
    TORCH_CHECK(q.is_cuda() && q.scalar_type()==at::kBFloat16 && q.sizes()==at::IntArrayRef({1,8,2048,64}),"B1 H8 T2048 D64 BF16 required");
    TORCH_CHECK(q.sizes()==k.sizes() && q.sizes()==v.sizes() && q.sizes()==out.sizes(),"QKV/output shape");
    for(const auto& x:{q,k,v,out})TORCH_CHECK(x.device()==q.device() && x.scalar_type()==q.scalar_type() && x.is_contiguous(),"BF16 device/layout");
    for(const auto& x:{lse,lse_accum,o_accum})TORCH_CHECK(x.is_contiguous() && x.device()==q.device() && x.scalar_type()==at::kFloat,"FP32 workspace");
    TORCH_CHECK(lse.numel()==8*2048 && lse_accum.numel()==2*8*2048 && o_accum.numel()==2*8*2048*64,"Workspace shape");
    TORCH_CHECK(stats.is_contiguous() && stats.device()==q.device() && stats.scalar_type()==at::kLong && stats.numel()==8*16*1*4*4,"Counter shape");
    TORCH_CHECK(prefix_stats.is_contiguous() && prefix_stats.device()==q.device() && prefix_stats.scalar_type()==at::kLong && prefix_stats.numel()==8*16*1*4*3,"Prefix counter shape");
    TORCH_CHECK(std::isfinite(scale) && scale>0,"Positive finite scale required");
    c10::cuda::CUDAGuard guard(q.device());auto stream=at::cuda::getCurrentCUDAStream();
    run035_flash::Params p{};
    p.q_ptr=q.data_ptr();p.k_ptr=k.data_ptr();p.v_ptr=v.data_ptr();p.o_ptr=out.data_ptr();
    p.q_batch_stride=p.k_batch_stride=p.v_batch_stride=p.o_batch_stride=8*2048*64;
    p.q_row_stride=p.k_row_stride=p.v_row_stride=p.o_row_stride=64;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=p.o_head_stride=2048*64;
    p.h=p.h_k=8;p.h_h_k_ratio=1;p.b=1;p.seqlen_q=p.seqlen_k=2048;p.d=p.d_rounded=64;
    p.seqlen_q_rounded=p.seqlen_k_rounded=p.total_q=2048;
    p.scale_softmax=float(scale);p.scale_softmax_log2=p.scale_softmax*M_LOG2E;
    p.p_dropout=p.rp_dropout=1.f;p.p_dropout_in_uint8_t=255;p.scale_softmax_rp_dropout=p.scale_softmax;
    p.is_bf16=p.is_causal=p.is_seqlens_k_cumulative=true;
    p.window_size_left=2048;p.window_size_right=0;p.num_splits=1;
    p.softmax_lse_ptr=lse.data_ptr();p.softmax_lseaccum_ptr=lse_accum.data_ptr();p.oaccum_ptr=o_accum.data_ptr();p.run028_stats=stats.data_ptr<int64_t>();
    if(count)C10_CUDA_CHECK(cudaMemsetAsync(prefix_stats.data_ptr(),0,prefix_stats.numel()*sizeof(int64_t),stream));
    if(skip){if(count)run035_flash::launch<true,true>(p,stream);else run035_flash::launch<true,false>(p,stream);}
    else{if(count)run035_flash::launch<false,true>(p,stream);else run035_flash::launch<false,false>(p,stream);}
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&run035_forward);}
