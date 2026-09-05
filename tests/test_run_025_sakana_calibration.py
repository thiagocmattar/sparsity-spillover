from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from sparsity_research.pythia import apply_activation_topology, topology_metadata

RUN = Path(__file__).resolve().parents[1] / "runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"


def load(name):
    spec = importlib.util.spec_from_file_location("run025_test_" + name, RUN / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P0, MEASURE, COMMON = load("p0"), load("measurement"), load("run025_common")


@pytest.mark.parametrize("k", [1, 33, 128, 255, 256, 257, 512, 1024, 2048, 4096])
def test_pack_roundtrip_no_truncation_signed_and_ragged(k):
    x = torch.randn(3, k, generator=torch.Generator().manual_seed(k)).bfloat16()
    x[0].zero_()
    x[1, ::3] = 0
    packed = P0.pack_reference(x)
    restored = P0.unpack_reference(packed, k)
    assert torch.equal(restored, x)
    assert torch.equal(packed[:, :, 0].sum(1), (x != 0).sum(1))
    if k >= 256:
        assert int(packed[2, 0, 0]) == 256  # no upstream 31-nonzero cap


def reference_backend(x, weight, bias, packed, out):
    packed.copy_(P0.pack_reference(x))
    restored = P0.unpack_reference(packed, x.shape[1])
    result = restored.float() @ weight.float()
    if bias.numel():
        result += bias.float()
    out.copy_(result)


def test_linear_signed_bias_reuse_and_changed_input():
    layer = torch.nn.Linear(128, 129).bfloat16().eval()
    wrapper = P0.ExactTwELLLinear(layer, backend=reference_backend).eval()
    x = torch.randn(2, 3, 128).bfloat16()
    wrapper.mode = "p0"
    with torch.inference_mode():
        output = wrapper(x).clone()
        buffer = wrapper._out.data_ptr()
        expected = (x.float() @ layer.weight.float().T + layer.bias.float()).bfloat16()
        assert torch.equal(output, expected)
        zeros = wrapper(torch.zeros_like(x)).clone()
        assert wrapper._out.data_ptr() == buffer
        assert torch.equal(zeros, layer.bias.expand_as(zeros))
    with pytest.raises(RuntimeError, match="inference-only"):
        wrapper(x)


@pytest.mark.parametrize("topology", ["A0", "A1-H", "A4-Z", "A7-Z-POST"])
def test_adapter_preserves_gates_state_keys_bias_and_restores_native(topology):
    # Gate definitions mirror operational placement, not released model weights.
    from sparsity_research.sites import TOPOLOGIES
    cfg = GPTNeoXConfig(hidden_size=32, intermediate_size=128, num_hidden_layers=1,
        num_attention_heads=4, vocab_size=128, max_position_embeddings=32)
    cfg.topology_id = topology
    cfg.site_gate = None if topology == "A0" else {"operator": "relu"}
    if topology in {"A4-Z", "A7-Z-POST"}:
        cfg.site_gate = None
        cfg.site_gates = {site: {"operator": "symmetric_threshold" if site in {"q_post", "k_post", "v"} else "one_sided_threshold", "kappa": .05}
                          for site in TOPOLOGIES[topology].active_sites}
    torch.manual_seed(2503)
    model = apply_activation_topology(GPTNeoXForCausalLM(cfg), torch=torch).bfloat16().eval()
    model.set_attn_implementation("sdpa")
    before = topology_metadata(model)
    keys = list(model.state_dict())
    adapter = P0.Adapter(model, backend=reference_backend)
    inputs = torch.arange(8).unsqueeze(0)
    with torch.inference_mode():
        adapter.set_mode("native")
        reference = model(input_ids=inputs, use_cache=False).logits.clone()
        adapter.set_mode("adapter_dense")
        assert torch.equal(model(input_ids=inputs, use_cache=False).logits, reference)
        assert list(model.state_dict()) == keys
        assert topology_metadata(model) == before
        adapter.set_mode("p0")
        actual = model(input_ids=inputs, use_cache=False).logits
        assert MEASURE.numerical_gate(reference, actual, relative_l2=.02, atol=.03, rtol=.02)["pass"]
        adapter.set_mode("native")
        assert torch.equal(model(input_ids=inputs, use_cache=False).logits, reference)


def test_rotating_timer_pairs_every_input_and_mode(monkeypatch):
    ticks = iter(range(1000))
    monkeypatch.setattr(MEASURE.time, "perf_counter", lambda: next(ticks) / 1000)
    seen = []
    active = [None]
    samples = MEASURE.paired_timing(lambda x: seen.append((active[0], x)),
        lambda mode: active.__setitem__(0, mode), list(range(5)), modes=["native", "p0"],
        passes=3, warmups=0, seed=2504, cuda=False)
    assert len(samples) == len(seen) == 30
    for mode in ("native", "p0"):
        assert sorted(x for m, x in seen if m == mode) == sorted(list(range(5)) * 3)
    assert MEASURE.paired_schedule(5, 3, ["native", "p0"], 2504) == MEASURE.paired_schedule(5, 3, ["native", "p0"], 2504)
    assert len(MEASURE.timing_summary(samples)) == 2


def test_gate_rejects_nan_zero_reference_and_bad_values():
    for actual in (torch.tensor([float("nan")]), torch.tensor([1.])):
        assert not MEASURE.numerical_gate(torch.zeros(1), actual, relative_l2=.02, atol=.25, rtol=.02)["pass"]
    assert MEASURE.numerical_gate(torch.zeros(1), torch.zeros(1), relative_l2=.02, atol=.25, rtol=.02)["pass"]


def test_allowlisted_path_and_atomic_record(tmp_path):
    with pytest.raises(ValueError, match="escapes"):
        COMMON.inside(tmp_path, "../outside")
    file = tmp_path / "evidence.json"
    COMMON.write_json(file, {"n": 7})
    row = COMMON.record(file, tmp_path)
    assert COMMON.verify_record(row, tmp_path) == file
    file.write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        COMMON.verify_record(row, tmp_path)


def test_sealed_upstream_git_identity_and_budget():
    cfg = COMMON.config()
    assert COMMON.sha256(RUN / "upstream/matmul_t2d.cu") == cfg["upstream_t2d_sha256"]
    assert cfg["budget"]["total_usd"] == 40
    assert cfg["budget"]["pilot_usd"] == 5
    assert len(cfg["calibration"]["conditions"]) == 8


def test_retained_inventory_partition_and_validation_coverage():
    manifest = COMMON.read_json(RUN / "prelaunch/input_manifest.json")
    assert len(manifest["checkpoints"]) == 36
    assert sum(row["partition"] == "development" for row in manifest["checkpoints"]) == 18
    assert len(set(manifest["development_source_block_ids"])) == 64
    assert manifest["validation"]["bytes"] == (338 * 2048 + 1444) * 4
    for row in manifest["checkpoints"]:
        assert len(row["files"]) == 4
        assert all("optimizer" not in file["path"] for file in row["files"])


def test_trial_record_is_immutable_and_checks_evaluator(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "run025_common", COMMON)
    trials = load("trial_record")
    source, evaluator = tmp_path / "kernel.txt", tmp_path / "evaluator.txt"
    source.write_text("source")
    evaluator.write_text("fixed")
    proposal = {"candidate_id": "DUMMY", "parent_id": "P0", "trajectory_id": 2501,
                "hypothesis": "CPU-only record smoke", "agent": {"type": "test"}, "budget": {"usd": 0}}
    result = {"status": "pass", "decision": "test_only", "elapsed_seconds": 0,
              "gpu_usd": 0, "agent_usd": 0, "token_usage": None, "evidence": []}
    directory = tmp_path / "trial"
    trials.begin(directory, proposal=proposal, source_paths=[source], evaluator_paths=[evaluator], root=tmp_path)
    # Simulate controller restart: reload from disk, not an in-memory proposal.
    trials = load("trial_record")
    trials.finish(directory, result=result, root=tmp_path)
    with pytest.raises(FileExistsError):
        trials.finish(directory, result=result, root=tmp_path)
    with pytest.raises(FileExistsError):
        trials.begin(directory, proposal=proposal, source_paths=[source], evaluator_paths=[evaluator], root=tmp_path)
    evaluator.write_text("changed")
    with pytest.raises(ValueError, match="identity"):
        trials.finish(directory, result=result, root=tmp_path)


def test_complete_validation_pools_shifted_tokens_and_rejects_tail(tmp_path, monkeypatch):
    import numpy as np
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules, "run025_common", COMMON)
    monkeypatch.setitem(sys.modules, "measurement", MEASURE)
    monkeypatch.setitem(sys.modules, "p0", P0)
    calibration = load("01_calibrate")

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.parameter = torch.nn.Parameter(torch.zeros(1))
            self.seen = []

        def forward(self, input_ids, use_cache):
            self.seen.append(input_ids.tolist())
            return SimpleNamespace(logits=torch.zeros(*input_ids.shape, 4))

    model = Model()
    cfg = COMMON.config()
    cfg["inputs"] = dict(cfg["inputs"], sequence_length=4, validation_blocks=3, validation_tail=2)
    progress = SimpleNamespace(check=lambda: None, emit=lambda *a, **kw: None)
    result = calibration.full_validation(model, SimpleNamespace(set_mode=lambda mode: None),
        np.arange(14, dtype=np.int32) % 4, cfg, progress, tmp_path / "val.json")
    assert result["complete"] and result["pass"]
    assert result["prediction_tokens"] == 9 and result["input_tokens"] == 12
    assert result["excluded_tail_tokens"] == 2
    assert result["loss"]["native"] == pytest.approx(np.log(4))
    assert len(model.seen) == 6
    with pytest.raises(ValueError, match="coverage"):
        calibration.full_validation(model, None, np.arange(13), cfg, progress, tmp_path / "bad.json")


def test_transfer_inventory_excludes_nonpilot_weights(monkeypatch):
    monkeypatch.setitem(sys.modules, "run025_common", COMMON)
    package = load("02_package")
    manifest = COMMON.read_json(RUN / "prelaunch/input_manifest.json")
    files = package.input_files(manifest, "calibration")
    weights = [row for row in files if row["path"].endswith("model.safetensors")]
    assert len(weights) == 8
    assert sum(row["bytes"] for row in weights) < 6_000_000_000
    assert not any(row["path"].endswith("train/tokens.int32.bin") for row in files)
