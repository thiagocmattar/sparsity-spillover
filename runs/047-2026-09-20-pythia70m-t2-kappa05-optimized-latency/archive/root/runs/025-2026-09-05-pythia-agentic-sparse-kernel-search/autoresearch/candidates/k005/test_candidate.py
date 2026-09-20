import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run025_k005_candidate", HERE / "candidate.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def test_width_dispatch_targets_pythia14m_output_shapes():
    assert C.output_plan(128) == {"vector_width": 4, "row_warps": 8}
    assert C.output_plan(384) == {"vector_width": 16, "row_warps": 4}
    assert C.output_plan(512) == {"vector_width": 16, "row_warps": 4}
    assert C.output_plan(1024) == {"vector_width": 16, "row_warps": 4}


def test_adapter_preserves_parameters_and_restores_native_modules():
    mlp = SimpleNamespace(
        dense_h_to_4h=torch.nn.Linear(32, 128),
        dense_4h_to_h=torch.nn.Linear(128, 32),
    )
    attention = SimpleNamespace(
        query_key_value=torch.nn.Linear(32, 96), dense=torch.nn.Linear(32, 32)
    )
    model = SimpleNamespace(
        gpt_neox=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp, attention=attention)])
    )
    adapter = C.Adapter(model)
    adapter.set_mode("adapter_dense")
    for parent, name, original, wrapped, _site in adapter.entries:
        assert getattr(parent, name) is wrapped
        assert wrapped.weight is original.weight and wrapped.bias is original.bias
    adapter.set_mode("native")
    for parent, name, original, _wrapped, _site in adapter.entries:
        assert getattr(parent, name) is original


def test_kernel_preserves_exact_signed_compaction_contract():
    source = (HERE / "kernel.cu").read_text()
    assert "lane_value != 0.f" in source
    assert "__ffs(remaining)" in source
    assert "fmaf" in source
    assert "launch<4, 8>" in source
    assert "launch<16, 4>" in source
