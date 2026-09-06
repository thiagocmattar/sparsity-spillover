"""K009 frozen 70M policy: K001 on transformer layers 4 and 5."""

from __future__ import annotations

import importlib.util
from pathlib import Path


CANDIDATE_ID = "k009"
EXPECTED_LAYERS = 6
SELECTED_LAYERS = frozenset((4, 5))


def _k001():
    source = Path(__file__).parents[1] / "k001/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_k009_inherited_k001", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Adapter:
    def __init__(self, model):
        layers = model.gpt_neox.layers
        if len(layers) != EXPECTED_LAYERS:
            raise ValueError("K009 is the frozen 70M policy")
        k001 = _k001()
        self.entries = []
        self.layer_selected = []
        for layer_index, layer in enumerate(layers):
            for parent, name, site in (
                (layer.mlp, "dense_h_to_4h", "m"),
                (layer.mlp, "dense_4h_to_h", "h"),
                (layer.attention, "query_key_value", "a"),
                (layer.attention, "dense", "z"),
            ):
                original = getattr(parent, name)
                wrapped = k001.FusedExactLinear(original).eval()
                self.entries.append((parent, name, original, wrapped, site))
                self.layer_selected.append(layer_index in SELECTED_LAYERS)

    def set_mode(self, mode):
        if mode not in {"native", "adapter_dense", CANDIDATE_ID}:
            raise ValueError(mode)
        for entry, selected in zip(self.entries, self.layer_selected, strict=True):
            parent, name, original, wrapped, _site = entry
            if mode == "native" or not selected:
                setattr(parent, name, original)
            else:
                wrapped.mode = "adapter_dense" if mode == "adapter_dense" else "k001"
                setattr(parent, name, wrapped)

    def coverage(self):
        return {
            "candidate_id": CANDIDATE_ID,
            "selected_layer_indices": sorted(SELECTED_LAYERS),
            "total_layers": EXPECTED_LAYERS,
            "selection_provenance": "K008 bounded 12-mask development search",
            "kernel": "qualified K001 fused exact warp compaction",
            "runtime_density_dispatch": False,
            "additional_pruning": False,
            "attention_qk_pv": "unchanged dense SDPA",
            "lm_head": "unchanged dense",
        }
