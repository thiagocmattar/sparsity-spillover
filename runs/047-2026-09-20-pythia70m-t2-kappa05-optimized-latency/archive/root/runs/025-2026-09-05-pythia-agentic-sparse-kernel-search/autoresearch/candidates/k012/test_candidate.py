import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run025_k012_candidate", HERE / "candidate.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def model(layer_count=6, hidden=128):
    layers = []
    for _ in range(layer_count):
        layers.append(
            SimpleNamespace(
                mlp=SimpleNamespace(
                    dense_h_to_4h=torch.nn.Linear(hidden, hidden * 4),
                    dense_4h_to_h=torch.nn.Linear(hidden * 4, hidden),
                ),
                attention=SimpleNamespace(
                    query_key_value=torch.nn.Linear(hidden, hidden * 3),
                    dense=torch.nn.Linear(hidden, hidden),
                ),
            )
        )
    return SimpleNamespace(gpt_neox=SimpleNamespace(layers=layers))


def test_k012_selects_only_layers_one_through_five():
    instance = model()
    adapter = C.Adapter(instance)
    adapter.set_mode(C.CANDIDATE_ID)
    for entry, selected in zip(adapter.entries, adapter.layer_selected, strict=True):
        parent, name, original, wrapped, _site = entry
        assert getattr(parent, name) is (wrapped if selected else original)
    assert C.SELECTED_LAYERS == {1, 2, 3, 4, 5}


def test_k012_restores_native_and_rejects_wrong_architecture():
    instance = model()
    adapter = C.Adapter(instance)
    adapter.set_mode(C.CANDIDATE_ID)
    adapter.set_mode("native")
    assert all(getattr(parent, name) is original for parent, name, original, _, _ in adapter.entries)
    with pytest.raises(ValueError, match="six-layer"):
        C.Adapter(model(layer_count=5))
    with pytest.raises(ValueError, match="14M"):
        C.Adapter(model(hidden=512))


def test_k012_coverage_records_static_policy():
    coverage = C.Adapter(model()).coverage()
    assert coverage["selected_layer_indices"] == [1, 2, 3, 4, 5]
    assert coverage["runtime_density_dispatch"] is False
    assert coverage["checkpoint_identity_dispatch"] is False
