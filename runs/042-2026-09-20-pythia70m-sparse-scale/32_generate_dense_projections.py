"""Six dense a/m schedules motivated by the training-only component profile."""
from io_utils import RUN, write, record

PROJECTION = '''import torch
import triton
import triton.language as tl

@triton.jit
def project(X, W, B, Y, N: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm, pn = tl.program_id(0), tl.program_id(1)
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(512 // BK):
        kk = block * BK + k
        x = tl.load(X + m[:, None] * 512 + kk[None, :])
        w = tl.load(W + n[None, :] * 512 + kk[:, None])
        acc = tl.dot(x, w, acc, out_dtype=tl.float32)
    bias = tl.load(B + n)
    acc = acc + bias[None, :].to(tl.float32)
    tl.store(Y + m[:, None] * N + n[None, :], acc.to(tl.bfloat16))

class Projection:
    def __init__(self, linear):
        assert linear.bias is not None and linear.in_features == 512
        self.weight, self.bias = linear.weight, linear.bias
        self.n = linear.out_features
        assert self.n in (1536, 2048)
        self.out = None
    def __call__(self, x):
        assert not torch.is_grad_enabled() and x.dtype == torch.bfloat16
        shape = x.shape[:-1]
        x = x.reshape(2048, 512).contiguous()
        if self.out is None:
            self.out = torch.empty((2048, self.n), device=x.device, dtype=x.dtype)
        bm, bn, bk, warps, stages = CONFIG
        project[(2048 // bm, self.n // bn)](x, self.weight, self.bias, self.out,
            self.n, bm, bn, bk, num_warps=warps, num_stages=stages)
        return self.out.view(*shape, self.n)
'''


def main():
    configs = [(32,64,32,4,3), (32,128,32,4,3), (64,64,32,4,3),
               (64,128,32,4,3), (64,128,64,4,3), (128,128,32,4,3)]
    for number, config in enumerate(configs, 44):
        identifier = f'opt{number:03d}'
        folder = RUN/'candidates'/identifier
        folder.mkdir(exist_ok=False)
        (folder/'projection.py').write_text(PROJECTION.replace('CONFIG', repr(config)), newline='\n')
        code = f'''"""Dense a/m scheduling; all sparse, attention and head components retained."""
from types import MethodType
from pathlib import Path
from io_utils import RUN,module,read,verify
HERE=Path(__file__).resolve().parent
def install(model):
    spec=read(HERE/'spec.json')
    for row in read(verify(spec['parent_manifest']))['files']:verify(row)
    base=module('run042_{identifier}_parent',RUN/'candidates/opt032/candidate.py')
    metadata=base.install(model)
    projection=module('run042_{identifier}_projection',HERE/'projection.py')
    for layer in model.gpt_neox.layers:
        for linear in (layer.attention.query_key_value,layer.mlp.dense_h_to_4h):
            linear._run042_dense_projection=projection.Projection(linear)
            linear.forward=MethodType(lambda obj,x:obj._run042_dense_projection(x),linear)
    return {{**metadata,'identity':'{identifier}','dense_am_change':True,
            'projection_schedule':{config},'input_projection_skipping':False,
            'input_projections':'Dense Triton BF16 products, FP32 accumulation and bias epilogue; original gates.'}}
'''
        (folder/'candidate.py').write_text(code, newline='\n')
        write(folder/'spec.json', {'kind':'dense-am-triton','schedule':config,
              'parent_manifest':record(RUN/'candidates/opt032/manifest.json'),
              'hypothesis':'Native a/m projections account for 0.264ms in the development profile; test dense schedules without claiming a sparse gain.',
              'constraints':'Same weights, biases, all products, FP32 accumulator, BF16 output, and gates; final eager-anchored bounds unchanged.'})
        write(folder/'manifest.json', {'candidate':identifier,
              'files':[record(p) for p in sorted(folder.glob('*'))],
              'selection_data':'fixed first16 training-development blocks only'})


if __name__ == '__main__':
    main()
