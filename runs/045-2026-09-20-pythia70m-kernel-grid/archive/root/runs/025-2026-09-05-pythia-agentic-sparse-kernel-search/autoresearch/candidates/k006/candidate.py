"""K006 architecture-shape portfolio over qualified K001 and K005 kernels."""

from __future__ import annotations

import importlib.util
from pathlib import Path


CANDIDATE_ID = "k006"
HERE = Path(__file__).resolve().parent

# Chosen from K005's direct K001/K005 development primitive comparison on the
# RTX PRO 4500.  Every other projection stays on qualified K001.
K005_SHAPES = {
    (128, 128): "pythia14m_z",
    (2048, 512): "pythia70m_h",
}


def _load(identifier):
    path = HERE.parent / identifier / "candidate.py"
    spec = importlib.util.spec_from_file_location(
        f"run025_k006_inherited_{identifier}", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


K001 = _load("k001")
K005 = _load("k005")


def backend_for_shape(k, n):
    return "k005" if (k, n) in K005_SHAPES else "k001"


class Adapter:
    """Resolve the inherited CUDA kernel once from each projection shape."""

    def __init__(self, model):
        self.entries = []
        self.backends = []
        for layer in model.gpt_neox.layers:
            for parent, name, site in (
                (layer.mlp, "dense_h_to_4h", "m"),
                (layer.mlp, "dense_4h_to_h", "h"),
                (layer.attention, "query_key_value", "a"),
                (layer.attention, "dense", "z"),
            ):
                original = getattr(parent, name)
                backend = backend_for_shape(
                    original.in_features, original.out_features
                )
                cls = (
                    K005.WidthSpecializedLinear
                    if backend == "k005"
                    else K001.FusedExactLinear
                )
                wrapped = cls(original).eval()
                self.entries.append((parent, name, original, wrapped, site))
                self.backends.append(backend)

    def set_mode(self, mode):
        if mode not in {"native", "adapter_dense", CANDIDATE_ID}:
            raise ValueError(mode)
        for entry, backend in zip(self.entries, self.backends, strict=True):
            parent, name, original, wrapped, _site = entry
            wrapped.mode = (
                "adapter_dense" if mode == "adapter_dense" else backend
            )
            setattr(parent, name, original if mode == "native" else wrapped)

    def coverage(self):
        selected = {}
        for (*_prefix, site), backend in zip(
            self.entries, self.backends, strict=True
        ):
            selected.setdefault(site, set()).add(backend)
        return {
            "candidate_id": CANDIDATE_ID,
            "dispatch": {site: sorted(values) for site, values in selected.items()},
            "k005_shapes": {
                f"K{k}_N{n}": label for (k, n), label in K005_SHAPES.items()
            },
            "fallback": "K001 fused exact compaction",
            "attention_qk_pv": "unchanged dense SDPA",
            "lm_head": "unchanged dense",
            "runtime_density_dispatch": False,
            "additional_pruning": False,
        }
