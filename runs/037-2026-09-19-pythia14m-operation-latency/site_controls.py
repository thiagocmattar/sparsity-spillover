"""Independent h/z and QK/PV controls with K050's unchanged tensor interfaces."""
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import torch
from io_utils import RUN, module, sha
from frozen_replay import R28

base_joint = module('run037_frozen_joint_class', R28 / 'candidates/k049/candidate.py')


@lru_cache(None)
def extension(kind, first, second):
    from torch.utils.cpp_extension import load
    folder = RUN / 'candidate'
    names = ['joint.cu'] if kind == 'joint' else ['kernel.cu', 'sparse_gemm.h', 'flash_fwd_kernel.h']
    switches = ('H', 'Z') if kind == 'joint' else ('QK', 'PV')
    defines = [f'-DRUN037_SKIP_{key}={int(flag)}' for key, flag in zip(switches, (first, second))]
    identity = ''.join(sha(folder / name) for name in names) + str(defines)
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    major, minor = torch.cuda.get_device_capability()
    os.environ['TORCH_CUDA_ARCH_LIST'] = f'{major}.{minor}'
    os.environ.setdefault('MAX_JOBS', '2')
    includes = [] if kind == 'joint' else [str(folder),
        str(R28 / 'runtime/vendor/flash/csrc/flash_attn/src'), str(R28 / 'runtime/vendor/cutlass/include')]
    return load(name=f'run037_{kind}_{digest}', sources=[str(folder / names[0])],
                extra_include_paths=includes,
                extra_cuda_cflags=['-O3', '-lineinfo', '--expt-relaxed-constexpr',
                                  '--expt-extended-lambda'] + defines, verbose=True)


class Joint(base_joint.Joint):
    def __init__(self, previous, skip_h, skip_z):
        super().__init__(previous)
        self.skip_h, self.skip_z = skip_h, skip_z

    def __call__(self, h, z, residual):
        if torch.is_grad_enabled() or h.dtype != torch.bfloat16:
            raise ValueError('BF16 inference only')
        h = h.reshape(-1, 512).contiguous()
        z = z.reshape(-1, 128).contiguous()
        r = residual.reshape(-1, 128).contiguous()
        if self.out is None or self.out.shape != r.shape:
            self.out = torch.empty_like(r)
            self.stats = torch.empty((r.shape[0] // 8, 8, 6), device=r.device, dtype=torch.int64)
        extension('joint', self.skip_h, self.skip_z).forward(
            h, z, self.w2.weight, self.wo.weight, self.wht, self.wzt,
            self.w2.bias, self.wo.bias, r, self.out, self.stats,
            self.th, self.tz, self.gh, self.gz, self.skip, self.fast_weights, self.count)
        return self.out.view_as(residual)


class Attention:
    def __init__(self, skip_qk, skip_pv):
        self.skip_qk, self.skip_pv = skip_qk, skip_pv
        self.shape = None

    def __call__(self, q, k, v, scale, *, count=False):
        if torch.is_grad_enabled() or q.dtype != torch.bfloat16:
            raise ValueError('BF16 inference only')
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        if tuple(q.shape) != (1, 4, 2048, 32):
            raise ValueError('B1 H4 T2048 D32 only')
        if self.shape != q.shape:
            self.out = torch.empty_like(q)
            self.lse = torch.empty((4, 2048), device=q.device, dtype=torch.float32)
            self.lse_accum = torch.empty((2, 4, 2048), device=q.device, dtype=torch.float32)
            self.o_accum = torch.empty((2, 4, 2048, 32), device=q.device, dtype=torch.float32)
            self.stats = torch.empty((4, 32, 2, 4, 4), device=q.device, dtype=torch.int64)
            self.prefix = torch.empty((2, 4, 1024, 32), device=q.device, dtype=torch.float32)
            self.safe = torch.empty((2, 4, 16, 32), device=q.device, dtype=torch.int32)
            self.prefix_stats = torch.empty((4, 32, 2, 4, 3), device=q.device, dtype=torch.int64)
            self.shape = q.shape
        extension('attention', self.skip_qk, self.skip_pv).forward(
            q, k, v, self.out, self.lse, self.lse_accum, self.o_accum,
            self.stats, self.prefix, self.safe, self.prefix_stats, scale, True, count, False)
        return self.out
