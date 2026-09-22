"""The opt063 joint projection with a declared compile-time scalar capacity."""
from functools import lru_cache
from pathlib import Path
import os
import torch
from io_utils import sha


@lru_cache(None)
def extension(limit):
    if limit not in (8, 16, 32, 64):
        raise ValueError('Declared short-row capacity required')
    from torch.utils.cpp_extension import load
    source = Path(__file__).with_suffix('.cu')
    major, minor = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name=f'run049_short{limit}_{sha(source)[:12]}', sources=[str(source)],
                extra_cuda_cflags=['-O3', '-lineinfo', '--ptxas-options=-v', f'-DSHORT_LIMIT={limit}'], verbose=True)


class Joint:
    def __init__(self, previous, limit):
        if limit not in (8, 16, 32, 64):
            raise ValueError(limit)
        for name in ('w2', 'wo', 'gh', 'gz', 'th', 'tz', 'skip'):
            setattr(self, name, getattr(previous, name))
        self.short_limit = limit
        self.parallel_prepass_duplicates = True
        self.count, self.out = False, None
        self.wht = self.w2.weight.t().contiguous()
        self.wzt = self.wo.weight.t().contiguous()
        self.fast_weights = all(bool((torch.isfinite(v) & ((v == 0) | ((v.abs() >= 2.**-50) & (v.abs() <= 2.**50)))).all())
                                for v in (self.w2.weight, self.wo.weight, self.w2.bias, self.wo.bias))

    def __call__(self, h, z, residual):
        if torch.is_grad_enabled() or h.dtype != torch.bfloat16:
            raise ValueError('BF16 inference only')
        h = h.reshape(-1, 2048).contiguous()
        z = z.reshape(-1, 512).contiguous()
        residual_flat = residual.reshape(-1, 512).contiguous()
        if self.out is None or self.out.shape != residual_flat.shape:
            self.row_counts = torch.empty((len(h), 2), device=h.device, dtype=torch.int32)
            self.out = torch.empty_like(residual_flat)
            self.stats = torch.empty((len(h) // 8, 2, 8, 6), device=h.device, dtype=torch.int64)
        extension(self.short_limit).forward(h, z, self.w2.weight, self.wo.weight, self.wht, self.wzt,
            self.w2.bias, self.wo.bias, residual_flat, self.out, self.stats,
            self.th, self.tz, self.gh, self.gz, self.skip, self.fast_weights, self.count, self.row_counts)
        return self.out.view_as(residual)
