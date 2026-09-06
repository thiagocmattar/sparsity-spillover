import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run025_k010_candidate", HERE / "candidate.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def model(layer_count=24):
    layers = [
        SimpleNamespace(attention=SimpleNamespace(dense=torch.nn.Linear(4, 4)))
        for _ in range(layer_count)
    ]
    return SimpleNamespace(gpt_neox=SimpleNamespace(layers=layers))


def test_policy_is_frozen_to_middle_z_layers_of_410m():
    assert C.EXPECTED_LAYERS == 24
    assert C.SELECTED_LAYERS == set(range(2, 21))
    assert C.SELECTED_SITE == "z"
    with pytest.raises(ValueError, match="410M"):
        C.Adapter(model(23))


def test_adapter_installs_exact_flagged_k004_and_restores_native():
    instance = model()
    originals = [layer.attention.dense for layer in instance.gpt_neox.layers]
    adapter = C.Adapter(instance)
    assert len(adapter.entries) == 19
    adapter.set_mode(C.CANDIDATE_ID)
    for index, layer in enumerate(instance.gpt_neox.layers):
        if index in C.SELECTED_LAYERS:
            assert layer.attention.dense is not originals[index]
            assert layer.attention.dense.mode == "k004"
            assert layer.attention.dense.strategy == "flagged"
        else:
            assert layer.attention.dense is originals[index]
    adapter.set_mode("native")
    assert all(
        layer.attention.dense is originals[index]
        for index, layer in enumerate(instance.gpt_neox.layers)
    )


def test_coverage_records_static_development_policy():
    coverage = C.Adapter(model()).coverage()
    assert coverage["selected_layer_indices"] == list(range(2, 21))
    assert coverage["selected_linear_count"] == 19
    assert coverage["runtime_density_dispatch"] is False
    assert coverage["additional_pruning"] is False
