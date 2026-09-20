from functools import lru_cache
from pathlib import Path
import os
import torch
from common import RUN,sha256

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    from io_utils import read,verify
    source=verify(read(Path(__file__).with_name('spec.json'))['shared_head_cuda'])
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
