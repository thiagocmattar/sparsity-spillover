"""K045 geometry with warp-uniform ballot classification and early third-nonzero return."""
from functools import lru_cache
from pathlib import Path
import os
import torch
from common import RUN,module,sha256


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source=Path(__file__).with_name('joint.cu');major,minor=torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST']=f'{major}.{minor}';os.environ.setdefault('MAX_JOBS','2')
    return load(name='run028_k048_'+sha256(source)[:12],sources=[str(source)],extra_cuda_cflags=['-O3','-lineinfo'],verbose=True)


class Joint:
    def __init__(self,previous):
        self.w2,self.wo=previous.w2,previous.wo
        self.gh,self.gz,self.th,self.tz=previous.gh,previous.gz,previous.th,previous.tz
        self.skip=previous.skip;self.count=False;self.out=None
        # Same weight values in a coalesced SIMT layout; preparation is outside timing.
        self.wht=self.w2.weight.t().contiguous();self.wzt=self.wo.weight.t().contiguous()
        self.fast_weights=all(bool((torch.isfinite(value)&((value==0)|((value.abs()>=2.**-50)&(value.abs()<=2.**50)))).all())
            for value in [self.w2.weight,self.wo.weight,self.w2.bias,self.wo.bias])
    def __call__(self,h,z,residual):
        if torch.is_grad_enabled() or h.dtype!=torch.bfloat16:raise ValueError('BF16 inference only')
        h=h.reshape(-1,512).contiguous();z=z.reshape(-1,128).contiguous();r=residual.reshape(-1,128).contiguous()
        if self.out is None or self.out.shape!=r.shape:
            self.out=torch.empty_like(r);self.stats=torch.empty((r.shape[0]//8,8,6),device=r.device,dtype=torch.int64)
        extension().forward(h,z,self.w2.weight,self.wo.weight,self.wht,self.wzt,self.w2.bias,self.wo.bias,r,self.out,self.stats,
            self.th,self.tz,self.gh,self.gz,self.skip,self.fast_weights,self.count)
        return self.out.view_as(residual)


def install(model,shortcut=True,round_p=False,skip=True,projection_skip=True):
    base=module('run028_k048_base',RUN/'candidates/k042/candidate.py')
    metadata=base.install(model,shortcut=shortcut,skip=skip,projection_skip=projection_skip)
    for layer in model.gpt_neox.layers:layer._run026_joint=Joint(layer._run026_joint)
    metadata['output_projections']='k048 M8N128 padded-M16 fallback: ballot-classified <=2-nonzero normal-range rows use SIMT; others retain K16 MMA order'
    metadata['output_projection_counter_limits']='MMA bypass includes replacement by SIMT; report executed SIMT products separately'
    return metadata
