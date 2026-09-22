// Run050 exact two-of-four decomposition and FP32 sparse-MMA output.
// CUTLASS is the existing pinned BSD-3-Clause dependency. Metadata addressing
// follows cutlass/util/host_reorder.h; no values are pruned on overflow.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_sparse.h"
#include "cutlass/epilogue/thread/linear_combination.h"
using BF=cutlass::bfloat16_t;
using Gemm=cutlass::gemm::device::SparseGemm<BF,cutlass::layout::RowMajor,
 BF,cutlass::layout::ColumnMajor,float,cutlass::layout::RowMajor,float,
 cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
 cutlass::gemm::GemmShape<128,64,64>,cutlass::gemm::GemmShape<64,32,64>,
 cutlass::gemm::GemmShape<16,8,32>,
 cutlass::epilogue::thread::LinearCombination<float,4,float,float>,
 cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3>;

__global__ void encode(const __nv_bfloat16* x,__nv_bfloat16* a,__nv_bfloat16* overflow,
 unsigned short* meta,int m,int k,float threshold,Gemm::LayoutE layout){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=m*(k/16))return;
 int row=i/(k/16),col=i%(k/16);unsigned short bits=0;
 for(int group=0;group<4;group++){
  int base=col*16+group*4;__nv_bfloat16 values[4];int pos[2]={0,1},count=0;
  #pragma unroll
  for(int j=0;j<4;j++){
   auto v=x[(long long)row*k+base+j];values[j]=__bfloat162float(v)>=threshold?v:__float2bfloat16(0.f);
   if(__bfloat162float(values[j])!=0.f){if(count<2)pos[count]=j;count++;}
  }
  if(count==0){pos[0]=0;pos[1]=1;}
  if(count==1){int p=pos[0];pos[0]=p==3?0:p;pos[1]=p==3?3:p+1;}
  a[(long long)row*(k/2)+base/2]=values[pos[0]];
  a[(long long)row*(k/2)+base/2+1]=values[pos[1]];
  bits|=(pos[0]|(pos[1]<<2))<<(group*4);
  #pragma unroll
  for(int j=0;j<4;j++)overflow[(long long)row*k+base+j]=(j==pos[0]||j==pos[1])?__float2bfloat16(0.f):values[j];
 }
 int dr=row/32*32+(row%8)*4+(row%32)/8,dc=col;
 if(dr%2==0&&dc%2==1){dr++;dc--;}
 else if(dr%2==1&&dc%2==0){dr--;dc++;}
 meta[layout(cutlass::MatrixCoord(dr,dc))]=bits;
}

void run(torch::Tensor x,torch::Tensor weights,torch::Tensor a,torch::Tensor overflow,
 torch::Tensor meta,torch::Tensor y,float threshold){
 TORCH_CHECK(Gemm::kSparse==2&&Gemm::kElementsPerElementE==8&&sizeof(Gemm::ElementE)==2,"unexpected sparse metadata format");
 int m=x.size(0),k=x.size(1),n=weights.size(0);
 TORCH_CHECK(m%32==0&&k%64==0&&n%64==0,"Sparse shape alignment");
 TORCH_CHECK(x.scalar_type()==at::kBFloat16&&weights.scalar_type()==at::kBFloat16&&y.scalar_type()==at::kFloat,"Sparse dtype contract");
 auto stream=at::cuda::getCurrentCUDAStream();auto layout=Gemm::LayoutE::packed({m,k/16});
 TORCH_CHECK(meta.numel()>=layout.capacity({m,k/16}),"Metadata capacity");
 encode<<<(m*(k/16)+127)/128,128,0,stream>>>(reinterpret_cast<__nv_bfloat16*>(x.data_ptr()),reinterpret_cast<__nv_bfloat16*>(a.data_ptr()),reinterpret_cast<__nv_bfloat16*>(overflow.data_ptr()),reinterpret_cast<unsigned short*>(meta.data_ptr()),m,k,threshold,layout);
 Gemm::Arguments args({m,n,k},{reinterpret_cast<BF*>(a.data_ptr()),k/2},
  {reinterpret_cast<BF*>(weights.data_ptr()),k},{y.data_ptr<float>(),n},{y.data_ptr<float>(),n},
  {reinterpret_cast<Gemm::ElementE*>(meta.data_ptr()),layout},{1.f,0.f},1);
 Gemm op;
 TORCH_CHECK(op.can_implement(args)==cutlass::Status::kSuccess,"Unsupported sparse arguments");
 TORCH_CHECK(Gemm::get_workspace_size(args)==0,"Unexpected sparse workspace");
 TORCH_CHECK(op(args,nullptr,stream)==cutlass::Status::kSuccess,"Sparse GEMM failed");
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("run",&run);}
