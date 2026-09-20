import torch
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
        bm, bn, bk, warps, stages = (64, 64, 32, 4, 3)
        project[(2048 // bm, self.n // bn)](x, self.weight, self.bias, self.out,
            self.n, bm, bn, bk, num_warps=warps, num_stages=stages)
        return self.out.view(*shape, self.n)
