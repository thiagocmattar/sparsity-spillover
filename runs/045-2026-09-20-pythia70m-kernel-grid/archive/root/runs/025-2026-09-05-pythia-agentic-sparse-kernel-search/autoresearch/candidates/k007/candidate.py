"""K007 launch-geometry search over K004's qualified tensor-core math."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F


CANDIDATE_ID = "k007"
ALLOWED_SITES = frozenset({"a", "m", "h", "z"})
CONFIGS = {
    "m16n128k32w4": (16, 128, 32, 4, 2),
    "m16n256k32w4": (16, 256, 32, 4, 2),
    "m16n256k32w8": (16, 256, 32, 8, 2),
    "m16n128k64w8": (16, 128, 64, 8, 2),
    "m16n256k64w8": (16, 256, 64, 8, 2),
    "m32n128k32w8": (32, 128, 32, 8, 2),
}


def _kernels():
    name = "run025_candidate_k007_kernels"
    if name not in sys.modules:
        source = Path(__file__).with_name("kernels.py")
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def geometry(config_id):
    if config_id not in CONFIGS:
        raise ValueError(f"Unknown K007 geometry: {config_id}")
    block_m, block_n, block_k, num_warps, num_stages = CONFIGS[config_id]
    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "num_warps": num_warps,
        "num_stages": num_stages,
    }


def tile_activity_reference(x, *, config_id):
    """CPU integer oracle for the exact K007 activity predicate."""
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("Expected CPU BF16 [M,K]")
    cfg = geometry(config_id)
    block_m, block_k = cfg["block_m"], cfg["block_k"]
    padded_m = ((x.shape[0] + block_m - 1) // block_m) * block_m
    padded_k = ((x.shape[1] + block_k - 1) // block_k) * block_k
    padded = torch.zeros(padded_m, padded_k, dtype=x.dtype)
    padded[: x.shape[0], : x.shape[1]] = x
    return padded.reshape(
        padded_m // block_m, block_m, padded_k // block_k, block_k
    ).ne(0).any(dim=3).any(dim=1)


class TileSkipLinear(nn.Module):
    def __init__(self, original, *, config_id, backend=None):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.config_id = config_id
        self.config = geometry(config_id)
        self.mode = "adapter_dense"
        self.backend = backend
        self._weight_t = None
        self._out = None
        self._empty_bias = None

    def forward(self, x):
        if self.mode == "adapter_dense":
            return F.linear(x, self.weight, self.bias)
        if self.mode != CANDIDATE_ID or self.training or torch.is_grad_enabled():
            raise RuntimeError("K007 is BF16 inference-only")
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            raise ValueError("K007 requires CUDA BF16 activations")
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        if self._weight_t is None:
            self._weight_t = self.weight.detach().T.contiguous()
            self._empty_bias = torch.empty(0, dtype=x.dtype, device=x.device)
        wanted = (flat.shape[0], self.out_features)
        if self._out is None or self._out.shape != wanted:
            self._out = torch.empty(wanted, dtype=x.dtype, device=x.device)
        (self.backend or tile_skip_linear)(
            flat,
            self._weight_t,
            self.bias if self.bias is not None else self._empty_bias,
            self._out,
            config=self.config,
        )
        return self._out.view(*shape[:-1], self.out_features)


def tile_skip_linear(x, weight_t, bias, output, *, config):
    if any(t.dtype != torch.bfloat16 or not t.is_cuda for t in (x, weight_t, output)):
        raise ValueError("K007 operands must be CUDA BF16")
    if x.ndim != 2 or weight_t.ndim != 2 or output.shape != (x.shape[0], weight_t.shape[1]):
        raise ValueError("K007 expects X[M,K], W[K,N], output[M,N]")
    if x.shape[1] != weight_t.shape[0]:
        raise ValueError("K007 reduction dimension mismatch")
    if not x.is_contiguous() or not weight_t.is_contiguous() or not output.is_contiguous():
        raise ValueError("K007 operands must be contiguous")
    if bias.numel() not in (0, weight_t.shape[1]):
        raise ValueError("K007 bias shape mismatch")
    m, k, n = x.shape[0], x.shape[1], weight_t.shape[1]
    block_m, block_n, block_k = (
        config["block_m"], config["block_n"], config["block_k"]
    )
    grid = ((m + block_m - 1) // block_m, (n + block_n - 1) // block_n)
    _kernels().inline_tile_skip_linear[grid](
        x,
        weight_t,
        bias,
        output,
        m,
        n,
        k,
        bias.numel() != 0,
        block_m,
        block_n,
        block_k,
        num_warps=config["num_warps"],
        num_stages=config["num_stages"],
    )


class Adapter:
    def __init__(self, model, *, config_id, backend=None):
        self.config_id = config_id
        geometry(config_id)
        self.entries = []
        for layer in model.gpt_neox.layers:
            for parent, name, site in (
                (layer.mlp, "dense_h_to_4h", "m"),
                (layer.mlp, "dense_4h_to_h", "h"),
                (layer.attention, "query_key_value", "a"),
                (layer.attention, "dense", "z"),
            ):
                original = getattr(parent, name)
                wrapped = TileSkipLinear(
                    original, config_id=config_id, backend=backend
                ).eval()
                self.entries.append((parent, name, original, wrapped, site))

    def set_mode(self, mode):
        if mode not in {"native", "adapter_dense", CANDIDATE_ID}:
            raise ValueError(mode)
        for parent, name, original, wrapped, _site in self.entries:
            wrapped.mode = mode
            setattr(parent, name, original if mode == "native" else wrapped)

    def coverage(self):
        return {
            "candidate_id": CANDIDATE_ID,
            "config_id": self.config_id,
            "geometry": geometry(self.config_id),
            "exact_zero_test": True,
            "linear_sites": sorted(ALLOWED_SITES),
            "attention_qk_pv": "unchanged dense SDPA",
            "lm_head": "unchanged dense",
            "static_transform": "BF16 weight transpose outside steady-state timing",
        }
