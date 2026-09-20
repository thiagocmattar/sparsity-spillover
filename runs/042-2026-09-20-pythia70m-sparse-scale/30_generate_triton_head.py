"""Eight development-only dense schedules after the author's continuation request."""
from io_utils import RUN, write, record

HEAD = '''import torch
import triton
import triton.language as tl

@triton.jit
def matmul(X, W, Y, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pid = tl.program_id(0)
    nm: tl.constexpr = 2048 // BM
    nn: tl.constexpr = triton.cdiv(50304, BN)
    group = pid // (8 * nn)
    first = group * 8
    size = tl.minimum(nm - first, 8)
    pm = first + (pid % (8 * nn)) % size
    pn = (pid % (8 * nn)) // size
    m = pm * BM + tl.arange(0, BM)
    n = pn * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for block in range(512 // BK):
        kk = block * BK + k
        x = tl.load(X + m[:, None] * 512 + kk[None, :])
        w = tl.load(W + n[None, :] * 512 + kk[:, None], n[None, :] < 50304, 0)
        acc = tl.dot(x, w, acc, out_dtype=tl.float32)
    tl.store(Y + m[:, None] * 50304 + n[None, :], acc.to(tl.bfloat16), n[None, :] < 50304)

class Head:
    def __init__(self, linear, tile):
        assert linear.bias is None and tuple(linear.weight.shape) == (50304, 512)
        self.weight = linear.weight
        self.out = None
    def __call__(self, x):
        assert not torch.is_grad_enabled() and x.dtype == torch.bfloat16
        shape = x.shape[:-1]
        x = x.reshape(2048, 512).contiguous()
        if self.out is None:
            self.out = torch.empty((2048, 50304), device=x.device, dtype=x.dtype)
        bm, bn, bk, warps, stages = CONFIG
        with torch.profiler.record_function('run042_dense_head'):
            matmul[((2048 // bm) * triton.cdiv(50304, bn),)](
                x, self.weight, self.out, bm, bn, bk, num_warps=warps, num_stages=stages)
        return self.out.view(*shape, 50304)
'''


def main():
    configs = [(32,128,64,4,3), (64,128,64,4,3), (64,256,64,8,3),
               (128,128,64,8,3), (128,256,64,8,3), (64,128,32,4,4),
               (128,128,32,4,4), (128,256,32,8,4)]
    parent = record(RUN/'candidates/opt032/manifest.json')
    for number, config in enumerate(configs, 33):
        identifier = f'opt{number:03d}'
        folder = RUN/'candidates'/identifier
        folder.mkdir(exist_ok=False)
        (folder/'head.py').write_text(HEAD.replace('CONFIG', repr(config)), newline='\n')
        code = f'''"""Dense full-logit scheduling only; exact parent sparse components."""
from types import MethodType
from pathlib import Path
from io_utils import RUN, module, read, verify
HERE = Path(__file__).resolve().parent
def install(model):
    spec = read(HERE/'spec.json')
    parent = read(verify(spec['parent_manifest']))
    for row in parent['files']: verify(row)
    base = module('run042_{identifier}_parent', RUN/'candidates/opt032/candidate.py')
    metadata = base.install(model)
    head = module('run042_{identifier}_head', HERE/'head.py')
    linear = model.embed_out
    linear._run042_head = head.Head(linear, 0)
    linear.forward = MethodType(lambda obj, x: obj._run042_head(x), linear)
    return {{**metadata, 'identity': '{identifier}', 'triton_version': head.triton.__version__,
            'head_schedule': {config}, 'dense_head_change': True}}
'''
        (folder/'candidate.py').write_text(code, newline='\n')
        write(folder/'spec.json', {'kind': 'dense-head-triton', 'index': 0,
              'schedule': config, 'parent_manifest': parent,
              'hypothesis': 'Development profile assigns 0.441 ms to the full vocabulary head; test grouped dense schedules with unchanged sparse components.',
              'constraints': 'Full 50304 logits; BF16 operands/output; FP32 accumulation; no split-K, quantization, caching, or omitted products.',
              'authorization': 'Author requested continued optimization until goal or original deadline after the initial 32-candidate sweep.'})
        files = sorted(folder.glob('*'))
        write(folder/'manifest.json', {'candidate': identifier, 'files': [record(p) for p in files],
              'selection_data': 'fixed first16 training-development blocks only'})


if __name__ == '__main__':
    main()
