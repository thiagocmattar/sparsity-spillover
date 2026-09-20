"""CPU ownership, graph-selection and paired-evidence checks; no GPU qualification."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

spec = importlib.util.spec_from_file_location("run025_model_probe_tested", Path(__file__).with_name("probe_models.py"))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_development_only_and_site_contract():
    manifest = {"checkpoints": [{"id": "14m/a0", "partition": "development"},
                                {"id": "14m/a7-0p1", "partition": "heldout"}]}
    assert probe.development_condition(manifest, "14m/a0")["id"] == "14m/a0"
    for identifier in ("unknown", "14m/a7-0p1"):
        with pytest.raises(ValueError, match="held out"):
            probe.development_condition(manifest, identifier)
    assert probe.select_sites("active", {"active_sites": []}) == frozenset()
    assert probe.select_sites("active", {"active_sites": ["a", "m", "h", "z", "q_post", "k_post", "v"]}) == probe.LINEAR_SITES
    assert probe.select_sites("h", {}) == {"h"}
    assert probe.select_sites("all", {}) == probe.LINEAR_SITES


def test_p0_site_subset_preserves_original_modules_and_true_mode():
    from p0 import Adapter
    mlp = SimpleNamespace(dense_h_to_4h=torch.nn.Linear(4, 8), dense_4h_to_h=torch.nn.Linear(8, 4))
    attention = SimpleNamespace(query_key_value=torch.nn.Linear(4, 12), dense=torch.nn.Linear(4, 4))
    model = SimpleNamespace(gpt_neox=SimpleNamespace(layers=[SimpleNamespace(mlp=mlp, attention=attention)]))
    adapter = probe.SiteAdapter(Adapter(model), "p0", {"h"})
    for mode in ("candidate", "native", "candidate"):
        adapter.set_mode(mode)
        for parent, name, original, wrapped, site in adapter.adapter.entries:
            assert getattr(parent, name) is (wrapped if site == "h" and mode == "candidate" else original)
            if mode == "candidate":
                assert wrapped.mode == "p0"


def test_selection_precedes_prepare_and_stage_but_not_timed_call():
    events = []
    adapter = SimpleNamespace(set_mode=lambda mode: events.append(("select", mode)))
    runner = SimpleNamespace(prepare=lambda: events.append(("prepare",)),
                             stage=lambda ids: events.append(("stage", ids)),
                             __call__=lambda: events.append(("run",)))
    class CallableRunner:
        def prepare(self): runner.prepare()
        def stage(self, ids): runner.stage(ids)
        def __call__(self): return "logits"
    selected = probe.SelectedRunner(CallableRunner(), adapter, "candidate")
    selected.prepare()
    selected.stage(17)
    assert selected() == "logits"
    assert events == [("select", "candidate"), ("prepare",), ("select", "candidate"), ("stage", 17)]


def test_matched_graph_reference_is_graph_not_eager():
    samples = [{"repeat": 0, "input_index": 0, "mode": mode, "host_ms": latency}
               for mode, latency in (("native", 10), ("candidate", 5), ("native_graph", 2), ("candidate_graph", 4))]
    result = probe.matched_summaries(samples)
    assert result["eager"]["summary"]["candidate"]["paired_geomean_speedup"] == 2
    assert result["cuda_graph"]["reference"] == "native_graph"
    assert result["cuda_graph"]["summary"]["candidate_graph"]["paired_geomean_speedup"] == .5
    with pytest.raises(ValueError, match="both"):
        probe.matched_summaries(samples[:1])


def test_rotating_inputs_and_real_mode_selection_match_full_logits():
    state = {"mode": None}
    adapter = SimpleNamespace(set_mode=lambda mode: state.update(mode=mode))
    inputs = [torch.tensor([[0, 1, 2, 3]]), torch.tensor([[3, 2, 1, 0]])]
    static = inputs[0].clone()
    seen = []
    def forward(ids):
        seen.append((state["mode"], ids.clone()))
        return torch.nn.functional.one_hot(ids, num_classes=5).float()
    runners = {mode: probe.SelectedRunner(probe.DenseRunner(forward, static, "native", cuda=False), adapter, mode)
               for mode in ("native", "candidate")}
    samples = probe.paired_probe(runners, inputs, passes=2, seed=2504, cuda=False)
    for row, (actual_mode, ids) in zip(samples, seen, strict=True):
        assert row["mode"] == actual_mode
        assert torch.equal(ids, inputs[row["input_index"]])
        assert row["output_shape"] == [1, 4, 5]
    result = probe.compare_inputs(runners, inputs, probe.config()["calibration"])
    assert result["prediction_tokens"] == 6
    assert result["pass"] == {"native": True, "candidate": True}


def test_source_inventory_contains_evaluator_adapter_kernel_and_config():
    for implementation, suffix in (("p0", "kernels/twell_pythia.cu"), ("k001", "candidates/k001/kernel.cu"),
                                    ("k003", "candidates/k003/kernel.cu")):
        records = probe.source_records(implementation)
        paths = [row["path"] for row in records]
        assert any(path.endswith(suffix) for path in paths)
        assert any(path.endswith("dense_probe/probe.py") for path in paths)
        assert any(path.endswith("measurement.py") for path in paths)
        assert all(len(row["sha256"]) == 64 and row["bytes"] > 0 for row in records)
