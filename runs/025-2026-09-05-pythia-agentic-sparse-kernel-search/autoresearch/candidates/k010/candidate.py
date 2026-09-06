"""K010 frozen 410M policy: flagged K004 at z in layers 2 through 20."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


CANDIDATE_ID = "k010"
EXPECTED_LAYERS = 24
SELECTED_LAYERS = frozenset(range(2, 21))
SELECTED_SITE = "z"
STRATEGY = "flagged"


def _k004():
    name = "run025_k010_inherited_k004"
    if name not in sys.modules:
        source = Path(__file__).parents[1] / "k004/candidate.py"
        spec = importlib.util.spec_from_file_location(name, source)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return sys.modules[name]


class Adapter:
    def __init__(self, model):
        layers = model.gpt_neox.layers
        if len(layers) != EXPECTED_LAYERS:
            raise ValueError("K010 is the frozen Pythia-410M policy")
        k004 = _k004()
        self.entries = []
        for layer_index in sorted(SELECTED_LAYERS):
            parent = layers[layer_index].attention
            original = parent.dense
            wrapped = k004.TileSkipLinear(
                original, site=SELECTED_SITE, strategy=STRATEGY
            ).eval()
            self.entries.append((parent, "dense", original, wrapped, SELECTED_SITE))

    def set_mode(self, mode):
        if mode not in {"native", "adapter_dense", CANDIDATE_ID}:
            raise ValueError(mode)
        for parent, name, original, wrapped, _site in self.entries:
            if mode == "native":
                setattr(parent, name, original)
            else:
                wrapped.mode = "adapter_dense" if mode == "adapter_dense" else "k004"
                setattr(parent, name, wrapped)

    def coverage(self):
        return {
            "candidate_id": CANDIDATE_ID,
            "selected_site": SELECTED_SITE,
            "selected_layer_indices": sorted(SELECTED_LAYERS),
            "selected_linear_count": len(self.entries),
            "total_layers": EXPECTED_LAYERS,
            "strategy": STRATEGY,
            "kernel": "K004 exact zero-tile tensor-core bypass with reusable flags",
            "selection_provenance": "410M A4-high/A7-high development occupancy",
            "runtime_density_dispatch": False,
            "additional_pruning": False,
            "attention_qk_pv": "unchanged dense SDPA",
            "other_linear_sites": "unchanged dense",
            "lm_head": "unchanged dense",
        }
