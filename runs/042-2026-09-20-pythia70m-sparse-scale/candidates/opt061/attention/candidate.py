import torch
from torch.nn.attention import SDPBackend,sdpa_kernel

class Attention:
    native=True
    def __init__(self,skip=False,shortcut=False):
        assert not skip and not shortcut
    def __call__(self,q,k,v,scale):
        assert not torch.is_grad_enabled() and q.dtype==torch.bfloat16
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return torch.nn.functional.scaled_dot_product_attention(q,k,v,is_causal=True,scale=scale)
