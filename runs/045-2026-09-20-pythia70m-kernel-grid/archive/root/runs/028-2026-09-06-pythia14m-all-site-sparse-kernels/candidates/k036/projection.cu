// K036: pinned CUTLASS pipeline with exact-zero MMA-atom bypass.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "cutlass/cutlass.h"
#include "cutlass/arch/mma_sm80.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

namespace cutlass { namespace arch {
struct Run028SparseMultiplyAdd {};
template<>
struct Mma<gemm::GemmShape<16,8,16>,32,bfloat16_t,layout::RowMajor,
           bfloat16_t,layout::ColumnMajor,float,layout::RowMajor,Run028SparseMultiplyAdd>
 : Mma<gemm::GemmShape<16,8,16>,32,bfloat16_t,layout::RowMajor,
       bfloat16_t,layout::ColumnMajor,float,layout::RowMajor,OpMultiplyAdd>{
    using Base=Mma<gemm::GemmShape<16,8,16>,32,bfloat16_t,layout::RowMajor,
                   bfloat16_t,layout::ColumnMajor,float,layout::RowMajor,OpMultiplyAdd>;
    CUTLASS_DEVICE void operator()(FragmentC &d,FragmentA const &a,FragmentB const &b,FragmentC const &c) const {
        const unsigned* words=reinterpret_cast<const unsigned*>(&a);
        bool zero=((words[0]|words[1]|words[2]|words[3])&0x7fff7fffu)==0;
        if(__all_sync(0xffffffffu,zero))d=c;
        else Base::operator()(d,a,b,c);
    }
};
}}

template<bool Skip>
void launch(torch::Tensor x,torch::Tensor w,torch::Tensor b,torch::Tensor out,cudaStream_t stream){
    using Element=cutlass::bfloat16_t;
    using Op=typename std::conditional<Skip,cutlass::arch::Run028SparseMultiplyAdd,cutlass::arch::OpMultiplyAdd>::type;
    using Gemm=cutlass::gemm::device::Gemm<Element,cutlass::layout::RowMajor,
        Element,cutlass::layout::ColumnMajor,Element,cutlass::layout::RowMajor,float,
        cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<32,32,64>,cutlass::gemm::GemmShape<16,16,64>,cutlass::gemm::GemmShape<16,8,16>,
        cutlass::epilogue::thread::LinearCombination<Element,4,float,float>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,2,8,8,false,Op>;
    int m=x.size(0),n=w.size(0),k=x.size(1);
    // Row-major C stride0 broadcasts the bias, with one final BF16 rounding.
    typename Gemm::Arguments args({m,n,k},
        {reinterpret_cast<Element*>(x.data_ptr()),k},{reinterpret_cast<Element*>(w.data_ptr()),k},
        {reinterpret_cast<Element*>(b.data_ptr()),0},{reinterpret_cast<Element*>(out.data_ptr()),n},{1.f,1.f});
    Gemm op;
    TORCH_CHECK(op.can_implement(args)==cutlass::Status::kSuccess,"CUTLASS unsupported shape");
    TORCH_CHECK(op(args,nullptr,stream)==cutlass::Status::kSuccess,"CUTLASS projection failed");
}

void forward(torch::Tensor x,torch::Tensor w,torch::Tensor b,torch::Tensor out,bool skip){
    TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16 && x.dim()==2 && x.size(1)==128,"CUDA BF16 Mx128");
    for(const auto& t:{x,w,b,out})TORCH_CHECK(t.device()==x.device() && t.scalar_type()==x.scalar_type() && t.is_contiguous(),"Type/layout");
    int m=x.size(0),n=w.size(0);
    TORCH_CHECK(m>0 && m%32==0 && w.dim()==2 && w.size(1)==128 && (n==384 || n==512),"Pythia a/m shapes");
    TORCH_CHECK(b.numel()==n && out.sizes()==at::IntArrayRef({m,n}),"Output/bias shape");
    c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
    if(skip)launch<true>(x,w,b,out,stream);else launch<false>(x,w,b,out,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
