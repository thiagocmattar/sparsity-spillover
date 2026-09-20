"""CPU/static K001 checks only. These do not qualify compiled CUDA behavior."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def load_candidate():
    path = Path(__file__).with_name("candidate.py")
    spec = importlib.util.spec_from_file_location("run025_k001_test_target", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


C = load_candidate()


def cpu_backend(x, weight, bias, out):
    value = x.float() @ weight.float()
    if bias.numel():
        value += bias.float()
    out.copy_(value)


@pytest.mark.parametrize("width", [1, 31, 32, 33, 128, 129, 256, 512, 1024, 4096])
def test_register_compaction_retains_signed_nonzeros_and_order(width):
    x = torch.arange(width).repeat(3, 1).float().sub(width / 2).to(torch.bfloat16)
    x[0].zero_()
    x[1, ::3] = 0
    x[1, -1] = -0.0
    for original, (columns, values) in zip(x, C.compaction_reference(x)):
        expected = torch.nonzero(original != 0).flatten()
        assert columns == expected.tolist()
        assert torch.equal(values.view(torch.int16), original[expected].view(torch.int16))
        restored = torch.zeros_like(original)
        restored[columns] = values
        assert torch.equal(restored, original)


@pytest.mark.parametrize("bias", [True, False])
def test_linear_wrapper_mathematical_cpu_oracle_and_output_shapes(bias):
    torch.manual_seed(2501)
    original = torch.nn.Linear(33, 129, bias=bias).bfloat16().eval()
    wrapped = C.FusedExactLinear(original, backend=cpu_backend).eval()
    for shape in [(2, 3, 33), (1, 7, 33)]:
        x = torch.randn(shape).bfloat16()
        x[x.abs() < 0.5] = 0
        with torch.inference_mode():
            assert torch.equal(wrapped(x), original(x))
            wrapped.mode = "k001"
            actual = wrapped(x).clone()
            reference = x.float() @ original.weight.float().T
            if bias:
                reference += original.bias.float()
            assert torch.equal(actual, reference.bfloat16())
            assert actual.shape == (*shape[:-1], 129)
            assert not hasattr(wrapped, "_packed")
            wrapped.mode = "adapter_dense"


def test_site_selection_preserves_native_fallback_and_restores_objects():
    mlp = SimpleNamespace(dense_h_to_4h=torch.nn.Linear(4, 16),
                          dense_4h_to_h=torch.nn.Linear(16, 4))
    attention = SimpleNamespace(query_key_value=torch.nn.Linear(4, 12),
                                dense=torch.nn.Linear(4, 4))
    model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=[
        SimpleNamespace(mlp=mlp, attention=attention)]))
    adapter = C.Adapter(model, sites=("h",), backend=cpu_backend)
    for mode in ("adapter_dense", "k001", "native"):
        adapter.set_mode(mode)
        for parent, name, original, wrapped, site in adapter.entries:
            assert getattr(parent, name) is (wrapped if site == "h" and mode != "native" else original)
    coverage = adapter.coverage()
    assert coverage["candidate_id"] == "k001"
    assert coverage["sparse_linear_count"] == 1
    assert coverage["dense_fallback_count"] == 3
    assert coverage["dense_fallback_sites"] == ["a", "m", "z"]
    with pytest.raises(ValueError):
        C.Adapter(model, sites=("q_post",))
    with pytest.raises(ValueError):
        adapter.set_mode("p0")


def test_inference_and_precision_guards():
    wrapped = C.FusedExactLinear(torch.nn.Linear(4, 4).bfloat16(), backend=cpu_backend).eval()
    wrapped.mode = "k001"
    with pytest.raises(RuntimeError):
        wrapped(torch.zeros(1, 4, dtype=torch.bfloat16))
    with torch.inference_mode(), pytest.raises(ValueError):
        wrapped(torch.zeros(1, 4))


def test_static_current_stream_and_no_lossy_work_or_fast_math():
    kernel = Path(__file__).with_name("kernel.cu").read_text()
    loader = Path(__file__).with_name("candidate.py").read_text()
    assert "getCurrentCUDAStream" in kernel
    assert "CUDAGuard" in kernel
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in kernel
    assert "remaining &= remaining - 1" in kernel
    assert "__ballot_sync(FULL, lane_value != 0.f)" in kernel
    assert "__shfl_sync(FULL, lane_value, source_lane)" in kernel
    assert "__float2bfloat16_rn(result)" in kernel
    assert "--use_fast_math" not in loader
