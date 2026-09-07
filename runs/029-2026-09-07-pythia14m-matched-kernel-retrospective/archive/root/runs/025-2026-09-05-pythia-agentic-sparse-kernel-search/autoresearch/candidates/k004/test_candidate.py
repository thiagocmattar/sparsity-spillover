import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run025_k004_candidate", HERE / "candidate.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def model():
    mlp = SimpleNamespace(
        dense_h_to_4h=torch.nn.Linear(32, 128),
        dense_4h_to_h=torch.nn.Linear(128, 32),
    )
    attention = SimpleNamespace(
        query_key_value=torch.nn.Linear(32, 96), dense=torch.nn.Linear(32, 32)
    )
    return SimpleNamespace(
        gpt_neox=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp, attention=attention)])
    )


def test_exact_tile_activity_keeps_signed_nonzeros_and_pads_inactive_edges():
    x = torch.zeros(17, 33, dtype=torch.bfloat16)
    x[0, 0] = -1
    x[16, 32] = 2
    flags = C.tile_activity_reference(x)
    assert flags.shape == (2, 2)
    assert torch.equal(flags, torch.tensor([[True, False], [False, True]]))


def test_hybrid_strategy_is_site_specialized():
    assert C.strategy_for_site("hybrid", "h") == "inline"
    assert C.strategy_for_site("hybrid", "z") == "inline"
    assert C.strategy_for_site("hybrid", "a") == "flagged"
    assert C.strategy_for_site("hybrid", "m") == "flagged"
    with pytest.raises(ValueError):
        C.strategy_for_site("thresholded", "h")


def test_adapter_preserves_parameter_identity_and_native_restore():
    instance = model()
    adapter = C.Adapter(instance)
    originals = [(parent, name, original) for parent, name, original, _, _ in adapter.entries]
    adapter.set_mode("adapter_dense")
    for parent, name, original, wrapped, _ in adapter.entries:
        assert getattr(parent, name) is wrapped
        assert wrapped.weight is original.weight
        assert wrapped.bias is original.bias
    adapter.set_mode("native")
    for parent, name, original in originals:
        assert getattr(parent, name) is original


def test_kernel_source_has_dynamic_exact_zero_branch_and_tensor_dot():
    source = (HERE / "kernels.py").read_text()
    assert "activations != 0" in source
    assert "if active:" in source
    assert "tl.dot" in source
    assert "threshold" not in source
