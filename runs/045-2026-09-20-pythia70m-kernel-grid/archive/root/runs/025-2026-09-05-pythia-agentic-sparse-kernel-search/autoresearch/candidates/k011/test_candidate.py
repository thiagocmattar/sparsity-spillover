import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run025_k011_candidate", HERE / "candidate.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def model(layer_count=6, hidden=128):
    layers = []
    for _ in range(layer_count):
        mlp = SimpleNamespace(
            dense_h_to_4h=torch.nn.Linear(hidden, hidden * 4),
            dense_4h_to_h=torch.nn.Linear(hidden * 4, hidden),
        )
        attention = SimpleNamespace(
            query_key_value=torch.nn.Linear(hidden, hidden * 3),
            dense=torch.nn.Linear(hidden, hidden),
        )
        layers.append(SimpleNamespace(mlp=mlp, attention=attention))
    return SimpleNamespace(gpt_neox=SimpleNamespace(layers=layers))


def test_mask_space_is_bounded_unique_and_excludes_full_k001():
    assert len(C.MASKS) == 12
    assert len({tuple(value) for value in C.MASKS.values()}) == 12
    assert all(0 < len(mask) < C.EXPECTED_LAYERS for mask in C.MASKS.values())
    assert set(C.selected_layers("prefix3")) == {0, 1, 2}
    assert set(C.selected_layers("suffix2")) == {4, 5}
    with pytest.raises(ValueError):
        C.selected_layers("all")


def test_adapter_changes_only_selected_layers_and_restores_native():
    instance = model()
    adapter = C.Adapter(instance, mask_id="even")
    adapter.set_mode(C.CANDIDATE_ID)
    for entry, selected in zip(adapter.entries, adapter.layer_selected, strict=True):
        parent, name, original, wrapped, _site = entry
        assert getattr(parent, name) is (wrapped if selected else original)
        if selected:
            assert wrapped.mode == "k001"
    adapter.set_mode("native")
    assert all(getattr(parent, name) is original for parent, name, original, _, _ in adapter.entries)


def test_wrong_architecture_is_rejected():
    with pytest.raises(ValueError, match="six-layer"):
        C.Adapter(model(layer_count=5), mask_id="prefix1")
    with pytest.raises(ValueError, match="14M-only"):
        C.Adapter(model(hidden=512), mask_id="prefix1")


def test_coverage_records_static_mask_without_runtime_dispatch():
    coverage = C.Adapter(model(), mask_id="suffix3").coverage()
    assert coverage["selected_layer_indices"] == [3, 4, 5]
    assert coverage["runtime_density_dispatch"] is False
    assert coverage["checkpoint_identity_dispatch"] is False
