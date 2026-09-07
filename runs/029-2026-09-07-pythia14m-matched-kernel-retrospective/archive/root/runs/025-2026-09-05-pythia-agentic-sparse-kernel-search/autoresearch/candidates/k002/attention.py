"""K002 host bridge: gate/RoPE-preserving zero-query causal attention."""
from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent


def validate_shape(query, key, value, block_m):
    if query.ndim != 4 or query.shape != key.shape or query.shape != value.shape:
        raise ValueError("K002 requires matching [B,H,T,D] full-sequence operands")
    if query.shape[-1] not in (32, 64) or not 1 <= query.shape[-2] <= 2048:
        raise ValueError("K002 supports D32/D64 and 1<=T<=2048")
    if block_m not in (16, 32):
        raise ValueError("K002 query tile must be 16 or 32")
    if any(t.dtype != torch.bfloat16 for t in (query, key, value)):
        raise ValueError("K002 requires BF16 operands")
    if any(t.device != query.device for t in (key, value)):
        raise ValueError("K002 operands must share a device")
    if any(any(stride <= 0 for stride in t.stride()) for t in (query, key, value)):
        raise ValueError("K002 requires positive strides")


class Workspace:
    """Shape-static scratch; all its values are overwritten every invocation."""
    def __init__(self, query, block_m=16):
        b, h, t, d = query.shape
        self.shape, self.device, self.block_m = tuple(query.shape), query.device, block_m
        self.chunk = 128
        self.nchunks = (t + self.chunk - 1) // self.chunk
        self.ntiles = (t + block_m - 1) // block_m
        self.prefix = torch.empty((b, h, t, d), dtype=torch.float32, device=query.device)
        self.totals = torch.empty((b, h, self.nchunks, d), dtype=torch.float32, device=query.device)
        self.output = torch.empty_like(query, memory_format=torch.contiguous_format)
        self.fast_tiles = torch.empty((b, h, self.ntiles), dtype=torch.uint8, device=query.device)

    @property
    def bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.prefix, self.totals, self.output, self.fast_tiles))


def _kernels():
    name = "run025_candidate_k002_kernels"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, HERE / "kernels.py")
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def attention(query, key, value, *, scale=None, block_m=16, workspace=None):
    """Exact mathematical causal attention; floating-point gates still required.

    Returned output aliases workspace.output. Consume/clone it before replaying
    this workspace. Prefix construction and branch detection always execute here.
    """
    validate_shape(query, key, value, block_m)
    if not query.is_cuda:
        raise ValueError("K002 execution requires CUDA; no CPU performance substitute")
    if torch.is_grad_enabled() and any(t.requires_grad for t in (query, key, value)):
        raise ValueError("K002 is inference-only")
    b, h, t, d = query.shape
    scale = d ** -.5 if scale is None else float(scale)
    if not math.isfinite(scale):
        raise ValueError("Attention scale must be finite")
    if workspace is None:
        workspace = Workspace(query, block_m)
    if (workspace.shape, workspace.device, workspace.block_m) != (tuple(query.shape), query.device, block_m):
        raise ValueError("K002 workspace shape/device/tile mismatch")
    kernels = _kernels()
    kernels.value_prefix_chunks[(workspace.nchunks, b * h)](
        value, workspace.prefix, workspace.totals, *value.stride(),
        h, t, d, workspace.chunk, workspace.nchunks, num_warps=4)
    kernels.zero_query_or_attention[(workspace.ntiles, b * h)](
        query, key, value, workspace.prefix, workspace.totals, workspace.output, workspace.fast_tiles,
        *query.stride(), *key.stride(), *value.stride(), h, t, d, scale, block_m, 64,
        workspace.chunk, workspace.nchunks, 1 << (workspace.nchunks - 1).bit_length(), workspace.ntiles,
        num_warps=4, num_stages=2)
    return workspace.output


def interface(module, query, key, value, attention_mask, dropout=0., scaling=None,
              is_causal=None, **kwargs):
    """Transformers interface called AFTER canonical post-RoPE Q/K/V gates.

    Unsupported masks/caches/dropout use correct dense SDPA, with no key removal.
    Integrating code must retain a dense-only sibling and record fallback scope.
    """
    if kwargs.get("output_attentions", False):
        raise ValueError("K002 does not materialize attention probabilities")
    causal = getattr(module, "is_causal", True) if is_causal is None else is_causal
    supported = (attention_mask is None and dropout == 0. and causal
                 and query.shape == key.shape == value.shape and query.ndim == 4
                 and query.shape[-1] in (32, 64) and 1 <= query.shape[-2] <= 2048
                 and query.dtype == key.dtype == value.dtype == torch.bfloat16 and query.is_cuda)
    if not supported:
        # Matches the installed Transformers SDPA causal policy for masks/caches.
        output = F.scaled_dot_product_attention(query, key, value,
            attn_mask=attention_mask, dropout_p=dropout, scale=scaling,
            is_causal=bool(causal and attention_mask is None and query.shape[-2] > 1))
    else:
        shape = (tuple(query.shape), query.device)
        if getattr(module, "_run025_k002_workspace_key", None) != shape:
            module._run025_k002_workspace = Workspace(query)
            module._run025_k002_workspace_key = shape
        output = attention(query, key, value, scale=scaling, workspace=module._run025_k002_workspace)
    return output.transpose(1, 2).contiguous(), None


def reference(query, key, value, scale=None):
    """CPU/diagnostic oracle only; it intentionally computes both branches."""
    q, k, v = query.double(), key.double(), value.double()
    scale = q.shape[-1] ** -.5 if scale is None else scale
    scores = (q @ k.transpose(-2, -1)) * scale
    mask = torch.ones(scores.shape[-2:], device=query.device, dtype=torch.bool).tril()
    dense = torch.softmax(scores.masked_fill(~mask, -float("inf")), dim=-1) @ v
    prefix = v.cumsum(dim=-2) / torch.arange(1, v.shape[-2] + 1, device=v.device).view(1, 1, -1, 1)
    return torch.where((q == 0).all(dim=-1, keepdim=True), prefix, dense)
