"""Pinned CUTLASS sparse a/m, with K033 h/z and K035 segmented attention."""
from functools import lru_cache
from pathlib import Path
from types import MethodType
import os
import torch
from common import RUN,module,sha256,read_json,verify_record


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('projection.cu')
    for row in read_json(RUN/'runtime/vendor/inventory.json')['files']:verify_record(row)
    major,minor=torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='run028_k042_'+sha256(source)[:12],sources=[str(source)],extra_include_paths=[str(RUN/'runtime/vendor/cutlass/include')],extra_cuda_cflags=['-O3','-lineinfo','--expt-relaxed-constexpr'],verbose=True)


class Projection:
    def __init__(self,linear,skip=True):
        self.linear,self.skip=linear,skip;self.output=None
    def __call__(self,x):
        if torch.is_grad_enabled() or x.dtype!=torch.bfloat16:raise ValueError('BF16 inference only')
        shape=x.shape[:-1];x=x.reshape(-1,128).contiguous();n=self.linear.weight.shape[0]
        if self.output is None or self.output.shape!=(x.shape[0],n):
            self.output=torch.empty((x.shape[0],n),device=x.device,dtype=x.dtype)
        extension().forward(x,self.linear.weight,self.linear.bias,self.output,self.skip)
        return self.output.view(*shape,n)


def install(model,shortcut=True,round_p=False,skip=True,projection_skip=True):
    base=module('run028_k042_base',RUN/'candidates/k035/candidate.py')
    metadata=base.install(model,shortcut=shortcut,skip=skip,projection_skip=projection_skip)
    for layer in model.gpt_neox.layers:
        for linear in [layer.attention.query_key_value,layer.mlp.dense_h_to_4h]:
            linear._run028_projection=Projection(linear,skip=projection_skip)
            linear.forward=MethodType(lambda obj,x:obj._run028_projection(x),linear)
    metadata['input_projections']='k042 pinned CUTLASS32x64x64 BF16 MMA sparse atom pipeline'
    metadata['input_projection_skip_granularity']='wholly zero 16x16 activation fragment; no weight-fragment test'
    return metadata
