import importlib.util
from pathlib import Path
from types import SimpleNamespace

import torch


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("run025_k006_candidate", HERE / "candidate.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def test_shape_dispatch_is_fixed_and_sparse_identity_independent():
    assert C.backend_for_shape(128, 128) == "k005"
    assert C.backend_for_shape(2048, 512) == "k005"
    for shape in ((128, 384), (128, 512), (512, 128), (512, 512), (4096, 1024)):
        assert C.backend_for_shape(*shape) == "k001"


def test_adapter_installs_inherited_kernel_without_forward_wrapper():
    mlp = SimpleNamespace(
        dense_h_to_4h=torch.nn.Linear(128, 512),
        dense_4h_to_h=torch.nn.Linear(512, 128),
    )
    attention = SimpleNamespace(
        query_key_value=torch.nn.Linear(128, 384),
        dense=torch.nn.Linear(128, 128),
    )
    model = SimpleNamespace(
        gpt_neox=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp, attention=attention)])
    )
    adapter = C.Adapter(model)
    assert adapter.backends == ["k001", "k001", "k001", "k005"]
    adapter.set_mode("k006")
    for (parent, name, original, wrapped, _site), backend in zip(
        adapter.entries, adapter.backends, strict=True
    ):
        assert getattr(parent, name) is wrapped
        assert wrapped.mode == backend
        assert wrapped.weight is original.weight
    adapter.set_mode("native")
    for parent, name, original, _wrapped, _site in adapter.entries:
        assert getattr(parent, name) is original


def test_coverage_declares_no_density_dispatch_or_new_pruning():
    mlp = SimpleNamespace(
        dense_h_to_4h=torch.nn.Linear(128, 512),
        dense_4h_to_h=torch.nn.Linear(512, 128),
    )
    attention = SimpleNamespace(
        query_key_value=torch.nn.Linear(128, 384),
        dense=torch.nn.Linear(128, 128),
    )
    model = SimpleNamespace(
        gpt_neox=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp, attention=attention)])
    )
    coverage = C.Adapter(model).coverage()
    assert not coverage["runtime_density_dispatch"]
    assert not coverage["additional_pruning"]
    assert coverage["dispatch"]["z"] == ["k005"]
