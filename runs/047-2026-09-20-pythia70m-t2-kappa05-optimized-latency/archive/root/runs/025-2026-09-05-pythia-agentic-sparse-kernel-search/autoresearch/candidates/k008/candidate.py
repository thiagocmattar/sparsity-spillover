"""K008 bounded 70M layer-mask search over the qualified K001 kernel."""

from __future__ import annotations

import importlib.util
from pathlib import Path


CANDIDATE_ID = "k008"
EXPECTED_LAYERS = 6
MASKS = {
    **{f"prefix{count}": tuple(range(count)) for count in range(1, EXPECTED_LAYERS)},
    **{
        f"suffix{count}": tuple(range(EXPECTED_LAYERS - count, EXPECTED_LAYERS))
        for count in range(1, EXPECTED_LAYERS)
    },
    "even": (0, 2, 4),
    "odd": (1, 3, 5),
}


def _k001():
    source = Path(__file__).parents[1] / "k001/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_k008_inherited_k001", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def selected_layers(mask_id):
    if mask_id not in MASKS:
        raise ValueError(f"Unknown K008 layer mask: {mask_id}")
    return frozenset(MASKS[mask_id])


class Adapter:
    def __init__(self, model, *, mask_id):
        layers = model.gpt_neox.layers
        if len(layers) != EXPECTED_LAYERS:
            raise ValueError("K008 is a 70M-only development search")
        self.mask_id = mask_id
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
            "mask_id": self.mask_id,
            "selected_layer_indices": sorted(self.selected),
            "total_layers": EXPECTED_LAYERS,
            "kernel": "qualified K001 fused exact warp compaction",
            "runtime_density_dispatch": False,
            "additional_pruning": False,
            "attention_qk_pv": "unchanged dense SDPA",
            "lm_head": "unchanged dense",
        }
