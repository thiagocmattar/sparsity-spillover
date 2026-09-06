"""K004 Pythia linear adapter with exact zero-tile tensor-core bypass."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

CANDIDATE_ID = "k004"
ALLOWED_SITES = frozenset({"a", "m", "h", "z"})
STRATEGIES = frozenset({"inline", "flagged", "hybrid"})
BLOCK_M = 16
BLOCK_N = 128
BLOCK_K = 32


def _kernels():
    name = "run025_candidate_k004_kernels"
    if name not in sys.modules:
        source = Path(__file__).with_name("kernels.py")
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


def strategy_for_site(strategy, site):
    if strategy not in STRATEGIES or site not in ALLOWED_SITES:
        raise ValueError("Unsupported K004 strategy or Pythia linear site")
    if strategy != "hybrid":
        return strategy
    # N=128 at h/z means inline detection is consumed by one output tile.
    # Wider a/m projections reuse a separately materialized activity flag.
    return "inline" if site in {"h", "z"} else "flagged"


def tile_activity_reference(x, block_m=BLOCK_M, block_k=BLOCK_K):
    """CPU integer oracle for the exact activity flags used by K004."""
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("Expected CPU BF16 [M,K]")
    m, k = x.shape
    padded_m = ((m + block_m - 1) // block_m) * block_m
    padded_k = ((k + block_k - 1) // block_k) * block_k
    padded = torch.zeros(padded_m, padded_k, dtype=x.dtype)
    padded[:m, :k] = x
    return padded.reshape(
        padded_m // block_m, block_m, padded_k // block_k, block_k
    ).ne(0).any(dim=3).any(dim=1)


class TileSkipLinear(nn.Module):
    def __init__(self, original, *, site, strategy="hybrid", backend=None):
        super().__init__()
        self.weight = original.weight
        self.bias = original.bias
        self.in_features = original.in_features
        self.out_features = original.out_features
        self.site = site
        self.strategy = strategy_for_site(strategy, site)
        self.mode = "adapter_dense"
        self.backend = backend
        self._weight_t = None
        self._out = None
        self._flags = None
        self._empty_bias = None

    def forward(self, x):
        if self.mode == "adapter_dense":
            return F.linear(x, self.weight, self.bias)
        if self.mode != CANDIDATE_ID or self.training or torch.is_grad_enabled():
            raise RuntimeError("K004 is BF16 inference-only")
        if x.dtype != torch.bfloat16 or not x.is_cuda:
            raise ValueError("K004 requires CUDA BF16 activations")
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        if self._weight_t is None:
            self._weight_t = self.weight.detach().T.contiguous()
            self._empty_bias = torch.empty(0, dtype=x.dtype, device=x.device)
        wanted = (flat.shape[0], self.out_features)
        if self._out is None or self._out.shape != wanted:
            self._out = torch.empty(wanted, dtype=x.dtype, device=x.device)
        flag_shape = (
            (flat.shape[0] + BLOCK_M - 1) // BLOCK_M,
            (flat.shape[1] + BLOCK_K - 1) // BLOCK_K,
        )
        if self.strategy == "flagged" and (
            self._flags is None or self._flags.shape != flag_shape
        ):
            self._flags = torch.empty(flag_shape, dtype=torch.uint8, device=x.device)
        function = self.backend or tile_skip_linear
        function(
            flat,
            self._weight_t,
            self.bias if self.bias is not None else self._empty_bias,
            self._out,
            self._flags,
            strategy=self.strategy,
        )
        return self._out.view(*shape[:-1], self.out_features)


def tile_skip_linear(x, weight_t, bias, output, flags, *, strategy):
    if strategy not in {"inline", "flagged"}:
        raise ValueError("K004 execution strategy must be resolved")
    if any(t.dtype != torch.bfloat16 or not t.is_cuda for t in (x, weight_t, output)):
        raise ValueError("K004 operands must be CUDA BF16")
    if x.ndim != 2 or weight_t.ndim != 2 or output.shape != (x.shape[0], weight_t.shape[1]):
        raise ValueError("K004 expects X[M,K], W[K,N], output[M,N]")
    if x.shape[1] != weight_t.shape[0]:
        raise ValueError("K004 reduction dimension mismatch")
    if not x.is_contiguous() or not weight_t.is_contiguous() or not output.is_contiguous():
        raise ValueError("K004 operands must be contiguous")
    if bias.numel() not in (0, weight_t.shape[1]):
        raise ValueError("K004 bias shape mismatch")
    m, k = x.shape
    n = weight_t.shape[1]
    kernels = _kernels()
    if strategy == "flagged":
        wanted = (
            (m + BLOCK_M - 1) // BLOCK_M,
            (k + BLOCK_K - 1) // BLOCK_K,
        )
        if flags is None or flags.shape != wanted or flags.dtype != torch.uint8:
            raise ValueError("K004 flagged strategy requires shape-static uint8 flags")
        kernels.detect_active_tiles[wanted](
            x, flags, m, k, BLOCK_M, BLOCK_K, num_warps=4
        )
        kernel = kernels.flagged_tile_skip_linear
    else:
        kernel = kernels.inline_tile_skip_linear
    grid = (
        (m + BLOCK_M - 1) // BLOCK_M,
        (n + BLOCK_N - 1) // BLOCK_N,
    )
    arguments = [x, weight_t, bias]
    if strategy == "flagged":
        arguments.append(flags)
    arguments += [output, m, n, k, bias.numel() != 0, BLOCK_M, BLOCK_N, BLOCK_K]
    kernel[grid](*arguments, num_warps=8, num_stages=2)


class Adapter:
    """Pythia site adapter; unselected sites remain native at dispatch time."""

    def __init__(self, model, *, strategy="hybrid", backend=None):
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown K004 strategy: {strategy}")
        self.strategy = strategy
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
                    original, site=site, strategy=strategy, backend=backend
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
            "strategy": self.strategy,
            "block_shape": [BLOCK_M, BLOCK_N, BLOCK_K],
            "exact_zero_test": True,
            "linear_sites": sorted(ALLOWED_SITES),
            "attention_qk_pv": "unchanged dense SDPA",
            "lm_head": "unchanged dense",
            "static_transform": "BF16 weight transpose outside steady-state timing",
        }
