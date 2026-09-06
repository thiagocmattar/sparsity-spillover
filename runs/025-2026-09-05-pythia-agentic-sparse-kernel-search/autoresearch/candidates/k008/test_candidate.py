import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run025_k008_candidate", HERE / "candidate.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def model(layer_count=6):
    layers = []
    for _ in range(layer_count):
        mlp = SimpleNamespace(
            dense_h_to_4h=torch.nn.Linear(4, 8),
            dense_4h_to_h=torch.nn.Linear(8, 4),
        )
        attention = SimpleNamespace(
            query_key_value=torch.nn.Linear(4, 12), dense=torch.nn.Linear(4, 4)
        )
        layers.append(SimpleNamespace(mlp=mlp, attention=attention))
    return SimpleNamespace(gpt_neox=SimpleNamespace(layers=layers))


def test_mask_space_is_bounded_unique_and_excludes_full_k001():
    assert len(C.MASKS) == 12
    assert all(0 < len(mask) < C.EXPECTED_LAYERS for mask in C.MASKS.values())
    assert set(C.selected_layers("prefix3")) == {0, 1, 2}
    assert set(C.selected_layers("suffix2")) == {4, 5}
    with pytest.raises(ValueError):
        C.selected_layers("all")


def test_adapter_changes_only_selected_layers_and_restores_native():
    instance = model()
    adapter = C.Adapter(instance, mask_id="even")
    originals = [entry[2] for entry in adapter.entries]
    adapter.set_mode(C.CANDIDATE_ID)
    for index, (entry, selected) in enumerate(
        zip(adapter.entries, adapter.layer_selected, strict=True)
    ):
        parent, name, original, wrapped, _site = entry
        assert getattr(parent, name) is (wrapped if selected else original)
        if selected:
            assert wrapped.mode == "k001"
        assert wrapped.weight is original.weight
        assert originals[index] is original
    adapter.set_mode("native")
    assert all(getattr(parent, name) is original for parent, name, original, _, _ in adapter.entries)


def test_wrong_architecture_is_rejected():
    with pytest.raises(ValueError, match="70M-only"):
        C.Adapter(model(layer_count=5), mask_id="prefix1")


def test_coverage_records_static_mask_and_no_runtime_dispatch():
    coverage = C.Adapter(model(), mask_id="suffix3").coverage()
    assert coverage["selected_layer_indices"] == [3, 4, 5]
    assert coverage["runtime_density_dispatch"] is False
