"""Exact input certificates; only the downstream projection implementation varies."""
from functools import lru_cache
import os
from common import RUN, record
import head as original


@lru_cache(None)
def extension():
    import torch
    from torch.utils.cpp_extension import load
    source = RUN/'variants.cu'
    major, minor = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS','2')
    return load(name='run069_'+record(source)['sha256'][:12], sources=[str(source)],
                extra_cuda_cflags=['-O3','-lineinfo','--ptxas-options=-v'], verbose=True)


class TunedHead(original.Head):
    def __init__(self, linear, spec, variant, *, skip):
        super().__init__(linear, spec, skip=skip, tile_vocabulary=variant['n'])
        self.variant = variant

    def project(self, *, count=False):
        v = self.variant
        extension().project(self.x_half, self.support, self.mma_weight, self.out,
                            self.stats, self.skip, count, v['n'], v['groups'],
                            v['family']=='direct', v['family']=='prefetch')
        return self.out


def make(linear, spec, variant, skip):
    if variant['family']=='legacy':
        return original.Head(linear, spec, skip=skip, tile_vocabulary=variant['n'])
    return TunedHead(linear, spec, variant, skip=skip)


def dense(linear):
    return original.DenseHead(linear)
