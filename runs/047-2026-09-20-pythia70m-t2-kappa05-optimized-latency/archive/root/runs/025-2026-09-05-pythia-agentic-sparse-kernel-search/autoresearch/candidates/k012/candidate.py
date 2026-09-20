"""K012 frozen-candidate 14M policy: K001 on transformer layers 1 through 5."""

from __future__ import annotations

import importlib.util
from pathlib import Path


CANDIDATE_ID = "k012"
EXPECTED_LAYERS = 6
EXPECTED_HIDDEN = 128
SELECTED_LAYERS = frozenset(range(1, 6))


def _k001():
    source = Path(__file__).parents[1] / "k001/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_k012_inherited_k001", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Adapter:
    def __init__(self, model):
        layers = model.gpt_neox.layers
        if len(layers) != EXPECTED_LAYERS:
            raise ValueError("K012 requires the six-layer Pythia-14M architecture")
        if layers[0].attention.dense.in_features != EXPECTED_HIDDEN:
            raise ValueError("K012 is the frozen Pythia-14M policy")
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
            "architecture": "Pythia-14M",
            "selected_layer_indices": sorted(SELECTED_LAYERS),
            "total_layers": EXPECTED_LAYERS,
            "kernel": "qualified K001 fused signed-exact warp compaction",
            "selection_provenance": "K011 fixed 12-mask development search and top-two confirmation",
            "runtime_density_dispatch": False,
            "checkpoint_identity_dispatch": False,
            "additional_pruning": False,
            "attention_qk_pv": "unchanged dense SDPA",
            "lm_head": "unchanged dense",
        }
