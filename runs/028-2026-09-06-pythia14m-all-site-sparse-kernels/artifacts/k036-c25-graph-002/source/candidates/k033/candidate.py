"""All four projections use tensor cores; K031 native-order sparse attention."""
from functools import lru_cache
from pathlib import Path
import os
import torch
from common import RUN,module,sha256


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('joint.cu')
    major,minor=torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='run028_k033_'+sha256(source)[:12],sources=[str(source)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)


class Joint:
    def __init__(self,previous):
        self.w2,self.wo=previous.w2,previous.wo
        self.gh,self.gz,self.th,self.tz=previous.gh,previous.gz,previous.th,previous.tz
        self.skip=previous.skip;self.out=None
    def __call__(self,h,z,residual):
        if torch.is_grad_enabled() or h.dtype!=torch.bfloat16:raise ValueError('BF16 inference only')
        h=h.reshape(-1,512).contiguous();z=z.reshape(-1,128).contiguous();r=residual.reshape(-1,128).contiguous()
        if self.out is None or self.out.shape!=r.shape:self.out=torch.empty_like(r)
        extension().forward(h,z,self.w2.weight,self.wo.weight,self.w2.bias,self.wo.bias,r,self.out,self.th,self.tz,self.gh,self.gz,self.skip)
        return self.out.view_as(residual)


def install(model,shortcut=True,round_p=False,skip=True,projection_skip=True):
    base=module('run028_k033_base',RUN/'candidates/k032/candidate.py')
    metadata=base.install(model,shortcut=shortcut,skip=skip,projection_skip=projection_skip)
    for layer in model.gpt_neox.layers:layer._run026_joint=Joint(layer._run026_joint)
    metadata['output_projections']='k033 BF16 MMA, K16 forward order, BF16 linear and residual rounding'
    metadata['output_projection_skip_granularity']='wholly zero gated 16x16 activation fragment'
    return metadata
