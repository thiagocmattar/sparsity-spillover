"""K015 bounded 70M A7 site/layer isolation over exact K001 compaction."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path


CANDIDATE_ID = "k015"
EXPECTED_LAYERS = 6
EXPECTED_HIDDEN_SIZE = 512
PLAN = Path(__file__).with_name("SEARCH.json")
VARIANTS = json.loads(PLAN.read_text(encoding="utf-8"))["variants"]


def implementation_for(variant_id):
    if variant_id not in VARIANTS:
        raise ValueError(f"Unknown K015 variant: {variant_id}")
    return f"{CANDIDATE_ID}-{variant_id}"


def variant(variant_id):
    implementation_for(variant_id)
    row = VARIANTS[variant_id]
    return frozenset(row["layers"]), frozenset(row["sites"])


def _k001():
    source = Path(__file__).parents[1] / "k001/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_k015_inherited_k001", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Adapter:
    def __init__(self, model, *, variant_id):
        layers = model.gpt_neox.layers
        if len(layers) != EXPECTED_LAYERS:
            raise ValueError("K015 requires the six-layer Pythia-70M architecture")
        if layers[0].attention.dense.in_features != EXPECTED_HIDDEN_SIZE:
            raise ValueError("K015 is a Pythia-70M-only development search")
        self.variant_id = variant_id
        self.implementation = implementation_for(variant_id)
        self.selected_layers, self.selected_sites = variant(variant_id)
        k001 = _k001()
        self.entries = []
        for layer_index, layer in enumerate(layers):
            for parent, name, site in (
                (layer.mlp, "dense_h_to_4h", "m"),
                (layer.mlp, "dense_4h_to_h", "h"),
                (layer.attention, "query_key_value", "a"),
                (layer.attention, "dense", "z"),
            ):
                if layer_index not in self.selected_layers or site not in self.selected_sites:
                    continue
                original = getattr(parent, name)
                wrapped = k001.FusedExactLinear(original).eval()
                self.entries.append((parent, name, original, wrapped, site))

    def set_mode(self, mode):
        if mode not in {"native", "adapter_dense", self.implementation}:
            raise ValueError(mode)
        for parent, name, original, wrapped, _site in self.entries:
            if mode == "native":
                setattr(parent, name, original)
            else:
                wrapped.mode = "adapter_dense" if mode == "adapter_dense" else "k001"
                setattr(parent, name, wrapped)

    def coverage(self):
        return {
            "candidate_id": CANDIDATE_ID,
            "implementation": self.implementation,
            "architecture": "Pythia-70M",
            "variant_id": self.variant_id,
            "selected_layer_indices": sorted(self.selected_layers),
            "selected_sites": sorted(self.selected_sites),
            "sparse_linear_count": len(self.entries),
            "kernel": "qualified K001 fused signed-exact warp compaction",
            "runtime_density_dispatch": False,
            "checkpoint_identity_dispatch": False,
            "additional_pruning": False,
            "attention_qk_pv": "unchanged dense SDPA",
            "lm_head": "unchanged dense",
        }
