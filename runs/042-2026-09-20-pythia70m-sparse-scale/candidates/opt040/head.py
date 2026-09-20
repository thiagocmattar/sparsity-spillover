import torch
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
        bm, bn, bk, warps, stages = (128, 256, 32, 8, 4)
        with torch.profiler.record_function('run042_dense_head'):
            matmul[((2048 // bm) * triton.cdiv(50304, bn),)](
                x, self.weight, self.out, bm, bn, bk, num_warps=warps, num_stages=stages)
        return self.out.view(*shape, 50304)
