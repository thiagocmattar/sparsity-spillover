"""Dense vocabulary-projection schedules; full BF16 logits and FP32 accumulation."""
import subprocess
import sys
from io_utils import RUN,write,record

CUDA=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/epilogue/thread/linear_combination.h"

template<int M,int N,int K,int WM,int WN>
void launch(torch::Tensor x,torch::Tensor w,torch::Tensor out,cudaStream_t stream){
 using E=cutlass::bfloat16_t;
 using Gemm=cutlass::gemm::device::Gemm<E,cutlass::layout::RowMajor,
  E,cutlass::layout::ColumnMajor,E,cutlass::layout::RowMajor,float,
  cutlass::arch::OpClassTensorOp,cutlass::arch::Sm80,
  cutlass::gemm::GemmShape<M,N,K>,cutlass::gemm::GemmShape<WM,WN,K>,cutlass::gemm::GemmShape<16,8,16>,
  cutlass::epilogue::thread::LinearCombination<E,8,float,float>,
  cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,3,8,8,false,cutlass::arch::OpMultiplyAdd>;
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
  case 0:launch<128,128,32,64,64>(x,w,out,stream);break;
  case 1:launch<128,256,32,64,64>(x,w,out,stream);break;
  case 2:launch<256,128,64,64,64>(x,w,out,stream);break;
  case 3:launch<128,128,64,64,64>(x,w,out,stream);break;
  default:TORCH_CHECK(false,"Unknown head tile");
 }
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("forward",&forward);}
'''

PYTHON='''from functools import lru_cache
from pathlib import Path
import os
import torch
from common import RUN,sha256

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('head.cu')
    major,minor=torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    return load(name='run042_head_'+sha256(source)[:12],sources=[str(source)],
        extra_include_paths=[str(RUN/'runtime/vendor/cutlass/include')],
        extra_cuda_cflags=['-O3','-lineinfo','--expt-relaxed-constexpr','--ptxas-options=-v'],verbose=True)

class Head:
    def __init__(self,linear,tile):
        assert linear.bias is None
        self.weight=linear.weight;self.tile=tile;self.out=None
    def __call__(self,x):
        assert not torch.is_grad_enabled() and x.dtype==torch.bfloat16
        shape=x.shape[:-1]
        x=x.reshape(2048,512).contiguous()
        if self.out is None:self.out=torch.empty((2048,50304),device=x.device,dtype=x.dtype)
        with torch.profiler.record_function('run042_dense_head'):
            extension().forward(x,self.weight,self.out,self.tile)
        return self.out.view(*shape,50304)
'''

def main():
    for index,tile in enumerate([(128,128,32),(128,256,32),(256,128,64),(128,128,64)]):
        identifier=f'opt{25+index:03d}';folder=RUN/'candidates'/identifier
        folder.mkdir(exist_ok=False)
        (folder/'head.cu').write_text(CUDA,newline='\n')
        (folder/'head.py').write_text(PYTHON,newline='\n')
        source=f'''"""Full vocabulary GEMM tile {tile}; sparse h/z unchanged."""
from pathlib import Path
from types import MethodType
from io_utils import RUN,module
HERE=Path(__file__).resolve().parent
def install(model):
    base=module('run042_{identifier}_base',RUN/'base70/candidate.py')
    metadata=base.install(model)
    head=module('run042_{identifier}_head',HERE/'head.py')
    linear=model.embed_out
    linear._run042_head=head.Head(linear,{index})
    linear.forward=MethodType(lambda obj,x:obj._run042_head(x),linear)
    return {{**metadata,'identity':'{identifier}','head_tile':{tile},
             'head_precision':'BF16 operands/output, FP32 accumulation, full50304 vocabulary',
             'dense_head_change':True,'sparse_sites':['h','z']}}
'''
        (folder/'candidate.py').write_text(source,newline='\n')
        write(folder/'spec.json',{'kind':'dense-head','tile':tile,'index':index,
            'sparse_joint':'Run040 M8 N256, unchanged','sparse_attention':False,
            'hypothesis':'The full vocabulary projection is about0.5ms; test dense scheduling separately from sparse gains.',
            'constraints':'No omitted logits, changed weights, quantization, split-K or reduced-precision accumulator.'})
        subprocess.run([sys.executable,str(RUN/'11_candidate.py'),'register',identifier],check=True)

if __name__=='__main__':main()
