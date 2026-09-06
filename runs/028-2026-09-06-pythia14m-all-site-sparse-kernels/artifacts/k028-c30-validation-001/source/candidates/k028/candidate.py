"""K027 attention plus sparse a/m projections with BF16-boundary repair."""
from functools import lru_cache
from pathlib import Path
from types import MethodType
import os
import torch
from common import RUN,module,sha256

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('projection.cu')
    major,minor=torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}';os.environ.setdefault('MAX_JOBS','2')
    return load(name='run028_k028_'+sha256(source)[:12],sources=[str(source)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)

class Projection:
    def __init__(self,linear,skip=True):
        self.linear,self.skip=linear,skip;self.weight=self.output=self.stats=None
    def __call__(self,x,*,count=False):
        if torch.is_grad_enabled() or x.dtype!=torch.bfloat16:raise ValueError('BF16 inference only')
        shape=x.shape[:-1];x=x.reshape(-1,128).contiguous()
        if self.weight is None:self.weight=self.linear.weight.T.contiguous()
        n=self.weight.shape[1]
        if self.output is None or self.output.shape!=(x.shape[0],n):
            self.output=torch.empty((x.shape[0],n),device=x.device,dtype=x.dtype)
            self.stats=torch.empty(4,device=x.device,dtype=torch.int64)
        if count:self.stats.zero_()
        extension().forward(x,self.weight,self.linear.bias,self.output,self.stats,self.skip,count)
        return self.output.view(*shape,n)

def install(model,shortcut=True,round_p=False,skip=True,projection_skip=True):
    base=module('run028_k028_base',RUN/'candidates/k027/candidate.py')
    coverage=base.install(model,skip=skip,projection_skip=projection_skip)
    for layer in model.gpt_neox.layers:
        for linear in [layer.attention.query_key_value,layer.mlp.dense_h_to_4h]:
            linear._run028_projection=Projection(linear,skip=projection_skip)
            linear.forward=MethodType(lambda obj,x:obj._run028_projection(x),linear)
    coverage.update(input_projections='k028 FP64 repair within 8 FP32 ULPs of a BF16 midpoint',
                    repair_counter_unit=['repaired_outputs','repair_fp64_products','main_fp32_products','main_skipped_products'])
    return coverage
