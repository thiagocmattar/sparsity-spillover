import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run025_k007_candidate", HERE / "candidate.py")
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


def test_geometry_space_is_bounded_and_valid():
    assert len(C.CONFIGS) == 6
    assert len(set(C.CONFIGS.values())) == len(C.CONFIGS)
    for identifier, values in C.CONFIGS.items():
        assert C.geometry(identifier) == dict(
            zip(("block_m", "block_n", "block_k", "num_warps", "num_stages"), values)
        )
    with pytest.raises(ValueError):
        C.geometry("unbounded")


def test_exact_tile_activity_keeps_signed_nonzeros_and_respects_geometry():
    x = torch.zeros(33, 65, dtype=torch.bfloat16)
    x[0, 0] = -1
    x[32, 64] = 2
    flags = C.tile_activity_reference(x, config_id="m32n128k32w8")
    assert flags.shape == (2, 3)
    assert torch.equal(
        flags,
        torch.tensor([[True, False, False], [False, False, True]]),
    )


def test_adapter_preserves_parameter_identity_and_records_geometry():
    instance = model()
    adapter = C.Adapter(instance, config_id="m16n256k32w4")
    originals = [(parent, name, original) for parent, name, original, _, _ in adapter.entries]
    adapter.set_mode("adapter_dense")
    for parent, name, original, wrapped, _site in adapter.entries:
        assert getattr(parent, name) is wrapped
        assert wrapped.weight is original.weight
        assert wrapped.bias is original.bias
    adapter.set_mode("native")
    for parent, name, original in originals:
        assert getattr(parent, name) is original
    assert adapter.coverage()["config_id"] == "m16n256k32w4"


def test_kernel_source_preserves_exact_zero_branch_and_tensor_dot():
    source = (HERE / "kernels.py").read_text()
    assert "activations != 0" in source
    assert "if active:" in source
    assert "tl.dot" in source
    assert "threshold" not in source
