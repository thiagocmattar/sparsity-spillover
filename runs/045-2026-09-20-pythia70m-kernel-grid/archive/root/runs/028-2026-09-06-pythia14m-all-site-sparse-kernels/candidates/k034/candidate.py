"""Shared-memory tiled tensor projections with K031 sparse attention."""
from functools import lru_cache
from pathlib import Path
from types import MethodType
import os
import torch
from common import RUN,module,sha256


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('kernel.cu')
    major,minor=torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='run028_k034_'+sha256(source)[:12],sources=[str(source)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)


class Projection:
    def __init__(self,linear,skip=True):self.linear,self.skip=linear,skip;self.output=None
    def __call__(self,x):
        if torch.is_grad_enabled() or x.dtype!=torch.bfloat16:raise ValueError('BF16 inference only')
        shape=x.shape[:-1];x=x.reshape(-1,128).contiguous();n=self.linear.weight.shape[0]
        if self.output is None or self.output.shape!=(x.shape[0],n):self.output=torch.empty((x.shape[0],n),device=x.device,dtype=x.dtype)
        extension().projection(x,self.linear.weight,self.linear.bias,self.output,self.skip)
        return self.output.view(*shape,n)


class Joint:
    def __init__(self,previous):
        self.w2,self.wo=previous.w2,previous.wo
        self.gh,self.gz,self.th,self.tz=previous.gh,previous.gz,previous.th,previous.tz
        self.skip=previous.skip;self.out=None
    def __call__(self,h,z,residual):
        if torch.is_grad_enabled() or h.dtype!=torch.bfloat16:raise ValueError('BF16 inference only')
        h=h.reshape(-1,512).contiguous();z=z.reshape(-1,128).contiguous();r=residual.reshape(-1,128).contiguous()
        if self.out is None or self.out.shape!=r.shape:self.out=torch.empty_like(r)
        extension().output(h,z,self.w2.weight,self.wo.weight,self.w2.bias,self.wo.bias,r,self.out,self.th,self.tz,self.gh,self.gz,self.skip)
        return self.out.view_as(residual)


def install(model,shortcut=True,round_p=False,skip=True,projection_skip=True):
    base=module('run028_k034_base',RUN/'candidates/k033/candidate.py')
    metadata=base.install(model,shortcut=shortcut,skip=skip,projection_skip=projection_skip)
    for layer in model.gpt_neox.layers:
        for linear in [layer.attention.query_key_value,layer.mlp.dense_h_to_4h]:
            linear._run028_projection=Projection(linear,skip=projection_skip)
            linear.forward=MethodType(lambda obj,x:obj._run028_projection(x),linear)
        layer._run026_joint=Joint(layer._run026_joint)
    metadata['input_projections']='k034 shared32x64x64, forward K16 BF16 MMA'
    metadata['output_projections']='k034 shared32x64x64, forward K16 BF16 MMA, rounded joint epilogue'
    return metadata
