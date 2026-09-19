"""K049's sparse-row strategy adapted to standalone N384/N512 projections."""
from functools import lru_cache
import os
import torch
from io_utils import RUN, sha


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    source = RUN / 'candidate/projection.cu'
    major, minor = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS', '2')
    return load(name='run038_projection_' + sha(source)[:12], sources=[str(source)],
                extra_cuda_cflags=['-O3', '-lineinfo'], verbose=True)


class Projection:
    def __init__(self, linear, skip=True):
        self.linear, self.skip = linear, skip
        self.output = None
        self.count = False
        self.weight_t = linear.weight.t().contiguous()
        self.fast_weights = all(bool((torch.isfinite(v) & ((v == 0) |
            ((v.abs() >= 2.**-50) & (v.abs() <= 2.**50)))).all())
            for v in [linear.weight, linear.bias])

    def __call__(self, x):
        if torch.is_grad_enabled() or x.dtype != torch.bfloat16:
            raise ValueError('BF16 inference only')
        shape = x.shape[:-1]
        x = x.reshape(-1, 128).contiguous()
        n = self.linear.weight.shape[0]
        if self.output is None or self.output.shape != (x.shape[0], n):
            self.output = torch.empty((x.shape[0], n), device=x.device, dtype=x.dtype)
            self.stats = torch.empty((x.shape[0] // 8 * (n // 128), 8, 3), device=x.device, dtype=torch.int64)
        extension().forward(x, self.linear.weight, self.weight_t, self.linear.bias,
                            self.output, self.stats, self.skip, self.fast_weights, self.count)
        return self.output.view(*shape, n)


def expected_counts(x, n, fast_weights=True, skip=True):
    x = x.reshape(-1, 128)
    potential = x.shape[0] // 8 * 8 * (n // 8)
    if not skip:
        return [potential, 0, 0]
    live = x != 0
    nnz = live.sum(-1)
    safe = ((x == 0) | ((x.abs() >= 2.**-50) & (x.abs() <= 2.**50))).all(-1)
    simple = (nnz <= 2) & safe if fast_weights else torch.zeros_like(safe)
    active = (live & ~simple[:, None]).reshape(-1, 8, 8, 16).any(1).any(-1)
    issued = int(active.sum()) * (n // 8)
    return [issued, potential - issued, int(nnz[simple].sum()) * n]
