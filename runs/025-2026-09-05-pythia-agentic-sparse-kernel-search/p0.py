"""Exact, non-truncating Sakana TwELL descendant. Inference only; no attention rewrite."""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

TILE, WORDS = 256, 288


def pack_reference(x):
    """CPU oracle of the CUDA wire format; includes negative nonzeros and tile tails."""
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.ndim != 2 or not 0 < x.shape[1] <= 65535:
        raise ValueError("Expected CPU BF16 [M,K], 1 <= K <= 65535")
    m, k = x.shape
    result = torch.zeros((m, (k + TILE - 1) // TILE, WORDS), dtype=torch.int64)
    bits = x.contiguous().view(torch.int16).to(torch.int64) & 65535
    for row in range(m):
        for tile in range(result.shape[1]):
            begin = tile * TILE
            columns = torch.nonzero(x[row, begin:begin + TILE] != 0).flatten() + begin
            result[row, tile, 0] = len(columns)
            result[row, tile, 1:len(columns) + 1] = (bits[row, columns] << 16) | columns
    return result.to(torch.int32)


def unpack_reference(packed, k):
    result = torch.zeros((packed.shape[0], k), dtype=torch.int16)
    for row in range(packed.shape[0]):
        for tile in range(packed.shape[1]):
            count = int(packed[row, tile, 0])
            if not 0 <= count <= TILE:
                raise ValueError("Invalid tile count")
            words = packed[row, tile, 1:count + 1].to(torch.int64) & 0xffffffff
            result[row, words & 65535] = (words >> 16).to(torch.int16)
    return result.view(torch.bfloat16)


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    if not torch.cuda.is_available():
        raise RuntimeError("P0 needs CUDA; CPU tests are reference-only")
    source = Path(__file__).parent / "kernels/twell_pythia.cu"
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    os.environ.setdefault("MAX_JOBS", "2")
    identity = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return load(name=f"run025_p0_{identity}_sm{major}{minor}", sources=[str(source)],
                extra_cuda_cflags=["-O3", "-lineinfo"], verbose=True)


class ExactTwELLLinear(nn.Module):
    def __init__(self, original, *, backend=None):
        super().__init__()
        self.weight, self.bias = original.weight, original.bias
        self.in_features, self.out_features = original.in_features, original.out_features
        self.mode = "adapter_dense"
        self.backend = backend
        self._weight_t = None
        self._packed = self._out = None
        self._empty_bias = None

    def forward(self, x):
        if self.mode == "adapter_dense":
            return F.linear(x, self.weight, self.bias)
        if self.mode != "p0" or self.training or torch.is_grad_enabled():
            raise RuntimeError("P0 is inference-only and requires no_grad/inference_mode")
        if x.dtype != torch.bfloat16:
            raise ValueError("P0 requires BF16 operands")
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        if self._weight_t is None:
            self._weight_t = self.weight.detach().T.contiguous()
            self._empty_bias = torch.empty(0, dtype=x.dtype, device=x.device)
        wanted = (flat.shape[0], self.out_features)
        if self._out is None or self._out.shape != wanted:
            self._out = torch.empty(wanted, dtype=x.dtype, device=x.device)
            self._packed = torch.empty((flat.shape[0], (flat.shape[1] + TILE - 1) // TILE, WORDS), dtype=torch.int32, device=x.device)
        (self.backend or extension().linear)(flat, self._weight_t,
            self.bias if self.bias is not None else self._empty_bias, self._packed, self._out)
        return self._out.view(*shape[:-1], self.out_features)


class Adapter:
    """Swap original Linear objects back for the native reference; leave all gates intact."""
    def __init__(self, model, *, backend=None):
        self.entries = []
        for layer in model.gpt_neox.layers:
            for parent, name, site in ((layer.mlp, "dense_h_to_4h", "m"),
                                       (layer.mlp, "dense_4h_to_h", "h"),
                                       (layer.attention, "query_key_value", "a"),
                                       (layer.attention, "dense", "z")):
                original = getattr(parent, name)
                wrapped = ExactTwELLLinear(original, backend=backend).eval()
                self.entries.append((parent, name, original, wrapped, site))

    def set_mode(self, mode):
        if mode not in {"native", "adapter_dense", "p0"}:
            raise ValueError(mode)
        for parent, name, original, wrapped, _ in self.entries:
            wrapped.mode = mode
            setattr(parent, name, original if mode == "native" else wrapped)

    def coverage(self):
        return {"linear_sites": ["a", "m", "h", "z"], "linear_count": len(self.entries),
                "attention_qk_pv": "unchanged dense SDPA", "lm_head": "unchanged dense",
                "packing": "dynamic signed exact; all nonzeros retained; inside timer",
                "static_transform": "BF16 weight transpose outside steady-state timing; setup time reported"}
