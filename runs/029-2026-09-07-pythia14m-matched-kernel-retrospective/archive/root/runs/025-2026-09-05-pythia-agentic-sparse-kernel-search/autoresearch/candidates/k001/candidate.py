"""K001 fused-compaction Sakana descendant; exact BF16 inference only."""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

CANDIDATE_ID = "k001"
ALLOWED_SITES = frozenset({"a", "m", "h", "z"})


@lru_cache(maxsize=1)
def extension():
    from torch.utils.cpp_extension import load
    if not torch.cuda.is_available():
        raise RuntimeError("K001 needs CUDA; CPU tests are reference-only")
    source = Path(__file__).with_name("kernel.cu")
    major, minor = torch.cuda.get_device_capability()
    os.environ["TORCH_CUDA_ARCH_LIST"] = f"{major}.{minor}"
    os.environ.setdefault("MAX_JOBS", "2")
    identity = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    return load(name=f"run025_k001_{identity}_sm{major}{minor}",
                sources=[str(source)], extra_cuda_cflags=["-O3", "-lineinfo"],
                verbose=True)


def compaction_reference(x):
    """Exact register-compaction order; no padded format is materialized on GPU."""
    if x.device.type != "cpu" or x.dtype != torch.bfloat16 or x.ndim != 2:
        raise ValueError("Expected CPU BF16 [M,K]")
    rows = []
    for row in x:
        columns = []
        for begin in range(0, row.numel(), 32):
            columns.extend((torch.nonzero(row[begin:begin + 32] != 0)
                            .flatten() + begin).tolist())
        rows.append((columns, row[columns].clone()))
    return rows


class FusedExactLinear(nn.Module):
    def __init__(self, original, *, backend=None):
        super().__init__()
        self.weight, self.bias = original.weight, original.bias
        self.in_features, self.out_features = original.in_features, original.out_features
        self.mode = "adapter_dense"
        self.backend = backend
        self._weight_t = self._out = self._empty_bias = None

    def forward(self, x):
        if self.mode == "adapter_dense":
            return F.linear(x, self.weight, self.bias)
        if self.mode != CANDIDATE_ID or self.training or torch.is_grad_enabled():
            raise RuntimeError("K001 is inference-only and requires no_grad/inference_mode")
        if x.dtype != torch.bfloat16:
            raise ValueError("K001 requires BF16 operands")
        shape = x.shape
        flat = x.reshape(-1, shape[-1]).contiguous()
        if self._weight_t is None:
            self._weight_t = self.weight.detach().T.contiguous()
            self._empty_bias = torch.empty(0, dtype=x.dtype, device=x.device)
        wanted = (flat.shape[0], self.out_features)
        if self._out is None or self._out.shape != wanted:
            self._out = torch.empty(wanted, dtype=x.dtype, device=x.device)
        (self.backend or extension().linear)(flat, self._weight_t,
            self.bias if self.bias is not None else self._empty_bias, self._out)
        return self._out.view(*shape[:-1], self.out_features)


class Adapter:
    """Site-only dispatch; unselected Linear modules remain the native objects."""
    def __init__(self, model, *, sites=("a", "m", "h", "z"), backend=None):
        self.sites = frozenset(sites)
        if not self.sites <= ALLOWED_SITES:
            raise ValueError(f"Unsupported linear sites: {sorted(self.sites - ALLOWED_SITES)}")
        self.entries = []
        for layer in model.gpt_neox.layers:
            for parent, name, site in ((layer.mlp, "dense_h_to_4h", "m"),
                                       (layer.mlp, "dense_4h_to_h", "h"),
                                       (layer.attention, "query_key_value", "a"),
                                       (layer.attention, "dense", "z")):
                original = getattr(parent, name)
                wrapped = FusedExactLinear(original, backend=backend).eval()
                self.entries.append((parent, name, original, wrapped, site))

    def set_mode(self, mode):
        if mode not in {"native", "adapter_dense", CANDIDATE_ID}:
            raise ValueError(mode)
        for parent, name, original, wrapped, site in self.entries:
            wrapped.mode = mode
            use_native = mode == "native" or site not in self.sites
            setattr(parent, name, original if use_native else wrapped)

    def coverage(self):
        return {"candidate_id": CANDIDATE_ID,
                "linear_sites": sorted(self.sites),
                "sparse_linear_count": sum(site in self.sites for *_, site in self.entries),
                "dense_fallback_sites": sorted(ALLOWED_SITES - self.sites),
                "dense_fallback_reason": "fixed site selection, not checkpoint identity or runtime density",
                "dense_fallback_count": sum(site not in self.sites for *_, site in self.entries),
                "attention_qk_pv": "unchanged dense SDPA", "lm_head": "unchanged dense",
                "packing": "signed exact warp-register compaction inside fused linear; no persistent packing",
                "static_transform": "BF16 weight transpose outside steady-state timing; setup time reported"}
