"""K014 bounded 70M layer-mask search over the exact K001 kernel."""

from __future__ import annotations

import importlib.util
from pathlib import Path


CANDIDATE_ID = "k014"
EXPECTED_LAYERS = 6
EXPECTED_HIDDEN_SIZE = 512
MASKS = {
    "suffix2": (4, 5),
    "suffix3": (3, 4, 5),
    "suffix4": (2, 3, 4, 5),
    "suffix5": (1, 2, 3, 4, 5),
    "all6": (0, 1, 2, 3, 4, 5),
    "odd": (1, 3, 5),
}


def implementation_for(mask_id):
    if mask_id not in MASKS:
        raise ValueError(f"Unknown K014 layer mask: {mask_id}")
    return f"{CANDIDATE_ID}-{mask_id}"


def selected_layers(mask_id):
    implementation_for(mask_id)
    return frozenset(MASKS[mask_id])


def _k001():
    source = Path(__file__).parents[1] / "k001/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_k014_inherited_k001", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Adapter:
    def __init__(self, model, *, mask_id):
        layers = model.gpt_neox.layers
        if len(layers) != EXPECTED_LAYERS:
            raise ValueError("K014 requires the six-layer Pythia-70M architecture")
        if layers[0].attention.dense.in_features != EXPECTED_HIDDEN_SIZE:
            raise ValueError("K014 is a Pythia-70M-only development search")
        self.mask_id = mask_id
        self.implementation = implementation_for(mask_id)
        self.selected = selected_layers(mask_id)
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
                self.layer_selected.append(layer_index in self.selected)

    def set_mode(self, mode):
        if mode not in {"native", "adapter_dense", self.implementation}:
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
            "implementation": self.implementation,
            "architecture": "Pythia-70M",
            "mask_id": self.mask_id,
            "selected_layer_indices": sorted(self.selected),
            "total_layers": EXPECTED_LAYERS,
            "kernel": "qualified K001 fused signed-exact warp compaction",
            "runtime_density_dispatch": False,
            "checkpoint_identity_dispatch": False,
            "additional_pruning": False,
            "attention_qk_pv": "unchanged dense SDPA",
            "lm_head": "unchanged dense",
        }
