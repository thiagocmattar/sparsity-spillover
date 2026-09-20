
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

template<int M,int N,int K,int WM,int WN,int Stages>
void launch(torch::Tensor x,torch::Tensor w,torch::Tensor out,cudaStream_t stream){
 using E=cutlass::bfloat16_t;
 using Gemm=cutlass::gemm::device::Gemm<E,cutlass::layout::RowMajor,
  E,cutlass::layout::ColumnMajor,E,cutlass::layout::RowMajor,float,
  cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<M,N,K>,cutlass::gemm::GemmShape<WM,WN,K>,cutlass::gemm::GemmShape<16,8,16>,
  cutlass::epilogue::thread::LinearCombination<E,8,float,float>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,Stages,8,8,false,cutlass::arch::OpMultiplyAdd>;
 typename Gemm::Arguments args({2048,50304,512},
  {reinterpret_cast<E*>(x.data_ptr()),512},{reinterpret_cast<E*>(w.data_ptr()),512},
  {reinterpret_cast<E*>(out.data_ptr()),50304},{reinterpret_cast<E*>(out.data_ptr()),50304},{1.f,0.f});
 Gemm op;
 TORCH_CHECK(op.can_implement(args)==cutlass::Status::kSuccess,"Head shape unsupported");
 TORCH_CHECK(op(args,nullptr,stream)==cutlass::Status::kSuccess,"Head GEMM failed");
}
void forward(torch::Tensor x,torch::Tensor w,torch::Tensor out,int tile){
 TORCH_CHECK(x.is_cuda() && x.scalar_type()==at::kBFloat16,"CUDA BF16 required");
 for(const auto& v:{x,w,out})TORCH_CHECK(v.device()==x.device() && v.scalar_type()==x.scalar_type() && v.is_contiguous(),"Head type/layout");
 TORCH_CHECK(x.sizes()==at::IntArrayRef({2048,512}) && w.sizes()==at::IntArrayRef({50304,512}) && out.sizes()==at::IntArrayRef({2048,50304}),"Full head shape");
 c10::cuda::CUDAGuard guard(x.device());auto stream=at::cuda::getCurrentCUDAStream();
 switch(tile){
  case 0:launch<64,128,32,32,64,3>(x,w,out,stream);break;
  case 1:launch<64,128,32,32,64,2>(x,w,out,stream);break;
  case 2:launch<128,64,32,64,32,3>(x,w,out,stream);break;
  case 3:launch<64,64,32,32,32,3>(x,w,out,stream);break;
  case 4:launch<64,256,32,32,64,3>(x,w,out,stream);break;
  case 5:launch<128,128,32,32,64,3>(x,w,out,stream);break;
  case 6:launch<128,128,32,64,64,2>(x,w,out,stream);break;
  case 7:launch<128,128,64,64,64,2>(x,w,out,stream);break;
  default:TORCH_CHECK(false,"Unknown head tile");
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
