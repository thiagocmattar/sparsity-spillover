import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run025_k009_candidate", HERE / "candidate.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def model(layer_count=6):
    layers = []
    for _ in range(layer_count):
        mlp = SimpleNamespace(
            dense_h_to_4h=torch.nn.Linear(4, 8), dense_4h_to_h=torch.nn.Linear(8, 4)
        )
        attention = SimpleNamespace(
            query_key_value=torch.nn.Linear(4, 12), dense=torch.nn.Linear(4, 4)
        )
        layers.append(SimpleNamespace(mlp=mlp, attention=attention))
    return SimpleNamespace(gpt_neox=SimpleNamespace(layers=layers))


def test_policy_is_frozen_to_last_two_of_six_layers():
    assert C.EXPECTED_LAYERS == 6
    assert C.SELECTED_LAYERS == {4, 5}
    with pytest.raises(ValueError, match="frozen 70M"):
        C.Adapter(model(5))


def test_adapter_changes_only_frozen_layers_and_restores_native():
    adapter = C.Adapter(model())
    adapter.set_mode(C.CANDIDATE_ID)
    for entry, selected in zip(adapter.entries, adapter.layer_selected, strict=True):
        parent, name, original, wrapped, _site = entry
        assert getattr(parent, name) is (wrapped if selected else original)
        if selected:
            assert wrapped.mode == "k001"
    adapter.set_mode("native")
    assert all(getattr(parent, name) is original for parent, name, original, _, _ in adapter.entries)


def test_coverage_records_search_provenance_and_static_dispatch():
    coverage = C.Adapter(model()).coverage()
    assert coverage["selected_layer_indices"] == [4, 5]
    assert coverage["runtime_density_dispatch"] is False
    assert coverage["selection_provenance"].startswith("K008")
