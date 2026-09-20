"""K016 frozen topology-dispatched Pythia-70M sparse inference policy."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from sparsity_research.pythia import topology_metadata


CANDIDATE_ID = "k016"
EXPECTED_LAYERS = 6
EXPECTED_HIDDEN_SIZE = 512
POLICIES = {
    "A0": {"layers": (), "sites": ()},
    "A1-H": {"layers": (4, 5), "sites": ("h",)},
    "A4-Z": {"layers": (3, 4, 5), "sites": ("a", "m", "h", "z")},
    "A7-Z-POST": {"layers": (3, 4, 5), "sites": ("h", "z")},
}


def _k001():
    source = Path(__file__).parents[1] / "k001/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_k016_inherited_k001", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Adapter:
    def __init__(self, model):
        layers = model.gpt_neox.layers
        if len(layers) != EXPECTED_LAYERS:
            raise ValueError("K016 requires the six-layer Pythia-70M architecture")
        if layers[0].attention.dense.in_features != EXPECTED_HIDDEN_SIZE:
            raise ValueError("K016 is frozen for Pythia-70M")
        self.topology_id = topology_metadata(model)["topology_id"]
        if self.topology_id not in POLICIES:
            raise ValueError(f"Unsupported K016 topology: {self.topology_id}")
        policy = POLICIES[self.topology_id]
        self.selected_layers = frozenset(policy["layers"])
        self.selected_sites = frozenset(policy["sites"])
        self.entries = []
        k001 = _k001()
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
        if mode not in {"native", "adapter_dense", CANDIDATE_ID}:
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
            "architecture": "Pythia-70M",
            "topology_id": self.topology_id,
            "selected_layer_indices": sorted(self.selected_layers),
            "selected_sites": sorted(self.selected_sites),
            "sparse_linear_count": len(self.entries),
            "native_fallback": not self.entries,
            "kernel": "qualified K001 fused signed-exact warp compaction",
            "topology_dispatch": True,
            "runtime_density_dispatch": False,
            "checkpoint_identity_dispatch": False,
            "kappa_dispatch": False,
            "additional_pruning": False,
            "attention_qk_pv": "unchanged dense SDPA",
            "lm_head": "unchanged dense",
        }
