"""Two compile-time controls; unchanged K050 interfaces and other operations."""
from functools import lru_cache
import hashlib
import os
import torch
from io_utils import RUN, module, sha
from frozen_replay import R28

base_joint = module('run056_frozen_joint_class', R28 / 'candidates/k049/candidate.py')


@lru_cache(None)
def extension(tile, short):
    from torch.utils.cpp_extension import load
    source = RUN / 'candidate/joint.cu'
    defines = [f'-DRUN056_TILE={int(tile)}', f'-DRUN056_SHORT={int(short)}']
    identity = hashlib.sha256((sha(source)+str(defines)).encode()).hexdigest()[:12]
    major, minor = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name=f'run056_joint_{identity}', sources=[str(source)],
                extra_cuda_cflags=['-O3', '-lineinfo', '--ptxas-options=-v']+defines,
                verbose=True)


class Joint(base_joint.Joint):
    def __init__(self, previous, tile, short):
        super().__init__(previous)
        self.tile, self.short = tile, short

    def __call__(self, h, z, residual):
        if torch.is_grad_enabled() or h.dtype != torch.bfloat16:
            raise ValueError('BF16 inference only')
        h = h.reshape(-1, 512).contiguous()
        z = z.reshape(-1, 128).contiguous()
        r = residual.reshape(-1, 128).contiguous()
        if self.out is None or self.out.shape != r.shape:
            self.out = torch.empty_like(r)
            self.stats = torch.empty((r.shape[0]//8, 8, 6), device=r.device, dtype=torch.int64)
        extension(self.tile, self.short).forward(
            h, z, self.w2.weight, self.wo.weight, self.wht, self.wzt,
            self.w2.bias, self.wo.bias, r, self.out, self.stats,
            self.th, self.tz, self.gh, self.gz, self.skip, self.fast_weights, self.count)
        return self.out.view_as(residual)
