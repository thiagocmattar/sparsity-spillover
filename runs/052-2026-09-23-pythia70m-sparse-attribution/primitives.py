"""Two initial no-global-packing variants, plus immutable Run050 controls."""
from functools import lru_cache
import torch
import legacy_ops
from support import RUN, sha

combine = legacy_ops.combine


def candidates(include_ablations=False):
    controls = [s for s in legacy_ops.candidates() if s['id'] in {
        'dense_native', 'dense_fused', 'dense_dot',
        'c_m4_k32_n128', 'c_m8_k32_n128', 'c_m16_k32_n128'}]
    new = [{'id': f'e_m{gm}_n{bn}', 'family': 'e', 'gm': gm, 'bn': bn}
           for gm in (4, 8, 16) for bn in (64, 128)]
    new += [{'id': f'f_r4_v{v}', 'family': 'f', 'rows': 4, 'v': v} for v in (4, 8)]
    result = controls + new
    if include_ablations:
        result += [{**s, 'id': s['id']+'_noskip', 'no_skip': True}
                   for s in controls+new if s['family'] in ('c', 'e', 'f')]
    return result


@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    return load(name='r052_sparse_'+sha(RUN/'fused_sparse.cu')[:12],
                sources=[str(RUN/'fused_sparse.cu')],
                extra_cuda_cflags=['-O3', '-lineinfo', '--ptxas-options=-v'],
                verbose=True)


class Linear:
    def __new__(cls, linear, threshold, spec, m=2048):
        if spec['family'] not in ('e', 'f'):
            return legacy_ops.Linear(linear, threshold, spec, m)
        return super().__new__(cls)

    def __init__(self, linear, threshold, spec, m=2048):
        self.spec = spec
        self.w = linear.weight.t().contiguous()
        self.bias = linear.bias
        self.k, self.n = self.w.shape
        self.m = m
        self.t = float(torch.tensor(threshold, dtype=torch.bfloat16)) if threshold is not None else -float('inf')
        self.out = torch.empty((m, self.n), device=self.w.device, dtype=torch.bfloat16)
        groups = (m + spec['gm']-1)//spec['gm'] if spec['family']=='e' else m
        width = spec['bn'] if spec['family']=='e' else 32*spec['v']
        self.count = torch.empty((groups, (self.n+width-1)//width), device=self.w.device, dtype=torch.int32)
        self.ext = extension()
        self.compiled = []  # CUDA build/PTX evidence is collected separately.

    def __call__(self, x):
        s = self.spec
        x = x.reshape(self.m, self.k)
        if not x.is_contiguous():
            raise ValueError('The current consumer requires contiguous operands')
        if s['family']=='e':
            self.ext.gather(x, self.w, self.bias, self.out, self.count, self.t,
                            s['gm'], s['bn'], s.get('no_skip', False))
        else:
            self.ext.ballot(x, self.w, self.bias, self.out, self.count, self.t,
                            s['v'], s.get('no_skip', False))
        return self.out
