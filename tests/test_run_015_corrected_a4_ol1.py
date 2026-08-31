import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from sparsity_research.pressure import activation_l1, parse_pressure_config
from sparsity_research.pythia import apply_activation_topology


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "015-2026-08-31-pythia14m-corrected-a4-ol1"
RUN012_DIR = ROOT / "runs" / "012-2026-08-30-pythia14m-full-pass-a4-ol1"


def _module(name):
    spec = importlib.util.spec_from_file_location(name, RUN_DIR / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def modules():
    names = (
        "_reuse_run004",
        "_reuse_run012",
        "run015_capture",
        "run_config",
        "initialization",
        "optimizer_boundary",
        "diagnostics",
        "smoke",
        "training",
        "verification",
    )
    previous = {name: sys.modules.get(name) for name in names}
    sys.path.insert(0, str(RUN_DIR))
    try:
        run_config = _module("run_config")
        boundary = _module("optimizer_boundary")
        smoke = _module("smoke")
        training = _module("training")
        verification = _module("verification")
        yield run_config, boundary, smoke, training, verification
    finally:
        sys.path.remove(str(RUN_DIR))
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_run015_locks_the_desired_five_condition_four_site_objective(modules):
    run_config, _, _, _, _ = modules
    config = run_config.load_config()
    rows = run_config.condition_specs(config)
    assert [row["gate_threshold"] for row in rows] == [0.0, 0.01, 0.05, 0.1, 0.5]
    assert run_config.EXPECTED_ACTIVE_SITES == ("a", "m", "h", "z")
    assert run_config.EXPECTED_PRESSURE_CAPTURE_TENSOR_COUNT == 24
    assert len(run_config.EXPECTED_PRESSURE_CAPTURE_NAMES) == 24
    assert all(row["topology_id"] == "A4-Z" for row in rows)
    assert all(row["active_sites"] == ["a", "m", "h", "z"] for row in rows)
    assert all(row["gate_operator"] == "one_sided_threshold" for row in rows)
    assert all(row["pressure_method"] == "orthogonal_l1" for row in rows)
    assert all(row["pressure_sites"] == ["a", "m", "h", "z"] for row in rows)
    assert all(row["pressure_weight"] == 1.0 for row in rows)
    assert all(row["step_budget"] == 1.0 for row in rows)


def test_run015_reuses_the_matched_recipe_but_has_a_distinct_record(modules):
    run_config, _, _, _, _ = modules
    import yaml

    current = run_config.load_config()
    historical = yaml.safe_load((RUN012_DIR / "config.yaml").read_text(encoding="utf-8"))
    assert current["name"] != historical["name"]
    current_without_name = {key: value for key, value in current.items() if key not in {"name", "runpod"}}
    historical_without_name = {key: value for key, value in historical.items() if key not in {"name", "runpod"}}
    assert current_without_name == historical_without_name
    current_runpod = dict(current["runpod"])
    historical_runpod = dict(historical["runpod"])
    assert current_runpod.pop("preflight_terminate_after_hours") == 1.5
    assert current_runpod.pop("scientific_terminate_after_hours") == 2.5
    historical_runpod.pop("seed_worker_terminate_after_hours")
    historical_runpod.pop("additional_worker_terminate_after_hours")
    assert current_runpod == historical_runpod
    assert current["comparison"]["source_run"].endswith("pythia14m-full-pass-a4z")


def test_training_adapter_replaces_frozen_h_capture_with_all_four_sites(modules):
    run_config, _, _, training, _ = modules
    capture = training.FourSitePressureCapture(object(), ["h"], torch=torch)
    assert tuple(capture.sites) == run_config.EXPECTED_ACTIVE_SITES
    with pytest.raises(RuntimeError, match="frozen Run 004 capture call changed"):
        training.FourSitePressureCapture(object(), ["a", "m", "h", "z"], torch=torch)


def test_adapter_realizes_every_post_gate_site_on_real_gpt_neox(modules):
    run_config, _, _, training, _ = modules
    config = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=2,
        num_attention_heads=2,
        max_position_embeddings=16,
        rotary_pct=0.25,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    config.topology_id = "A4-Z"
    config.site_gate = {"operator": "one_sided_threshold", "kappa": 0.05}
    model = apply_activation_topology(GPTNeoXForCausalLM(config), torch=torch)
    with training.FourSitePressureCapture(model, ["h"], torch=torch) as capture:
        input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
        output = model(input_ids=input_ids, labels=input_ids)
        pressure = activation_l1(capture.activations)
    assert torch.isfinite(output.loss)
    assert torch.isfinite(pressure)
    assert set(capture.activations) == {
        f"{site}.layer_{layer}"
        for site in run_config.EXPECTED_ACTIVE_SITES
        for layer in range(2)
    }
    assert all(
        torch.all(value[value != 0] >= 0.05)
        for value in capture.activations.values()
    )


class _Capture:
    def __init__(self):
        self.activations = {}

    def clear(self):
        self.activations.clear()


class _ToyModel(torch.nn.Module):
    def __init__(self, capture, *, omit_site=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
        self.capture = capture
        self.omit_site = omit_site
        self.config = SimpleNamespace(num_hidden_layers=1)

    def forward(self, input_ids, labels):
        hidden = input_ids.float() * self.weight
        multipliers = {"a": 1.0, "m": 2.0, "h": 3.0, "z": 4.0}
        for site, multiplier in multipliers.items():
            if site != self.omit_site:
                self.capture.activations[f"{site}.layer_0"] = hidden * multiplier
        return SimpleNamespace(loss=(hidden - 0.25).square().mean())


def _pressure():
    return parse_pressure_config({
        "method": "orthogonal_l1",
        "sites": ["a", "m", "h", "z"],
        "weight": 1.0,
        "step_budget": 1.0,
    })


def test_boundary_differentiates_one_unweighted_mean_over_all_four_sites(modules):
    run_config, boundary, _, _, _ = modules
    capture = _Capture()
    model = _ToyModel(capture)
    result = boundary.run_recipe_boundary(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0),
        batches=[torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 1.0]])],
        pressure=_pressure(),
        capture=capture,
        loss_scaler=boundary.DynamicLossScaler(scale=8.0),
        gradient_clip_norm=1.0,
        torch=torch,
        device=torch.device("cpu"),
    )
    assert result["pressure_loss"] == pytest.approx(5.625)
    assert result["weighted_pressure_loss"] == pytest.approx(5.625)
    assert result["pressure_capture_tensor_count"] == 4
    assert result["pressure_capture_names_sha256"] == boundary._capture_names_sha256(
        tuple(sorted(f"{site}.layer_0" for site in run_config.EXPECTED_ACTIVE_SITES))
    )
    assert result["ol1_correction_applied"] is True


def test_boundary_fails_closed_when_one_declared_site_is_not_realized(modules):
    _, boundary, _, _, _ = modules
    capture = _Capture()
    model = _ToyModel(capture, omit_site="z")
    with pytest.raises(RuntimeError, match="pressure capture mismatch.*z.layer_0"):
        boundary.run_recipe_boundary(
            model=model,
            optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
            batches=[torch.tensor([[1.0, 2.0]])],
            pressure=_pressure(),
            capture=capture,
            loss_scaler=boundary.DynamicLossScaler(scale=8.0),
            gradient_clip_norm=1.0,
            torch=torch,
            device=torch.device("cpu"),
        )


def test_preflight_requires_the_24_tensor_capture_identity(modules):
    run_config, _, smoke, _, _ = modules
    target = {
        "micro_batch_size": 32,
        "sequence_length": 2048,
        "gradient_accumulation_steps": 32,
        "memory_bytes": 80 * 1024**3,
    }
    rows = []
    for condition_id, threshold in zip(smoke.ENDPOINT_CONDITION_IDS, (0.0, 0.5)):
        rows.append({
            "condition_id": condition_id,
            "batch_size": 32,
            "status": "completed",
            "boundary_health": [{"optimizer_step_skipped": False, "gradient_overflow": False}],
            "last_boundary": {
                "task_loss": 5.0,
                "pressure_loss": 0.2,
                "adamw_gradient_norm_post_clip": 1.0,
                "pressure_to_task_ratio_raw": 1.2,
                "pressure_to_task_ratio_final": 1.0,
                "trust_scale": 0.8,
                "pressure_weight": 1.0,
                "ol1_correction_applied": True,
                "pressure_capture_tensor_count": 24,
                "pressure_capture_names_sha256": run_config.EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256,
            },
            "topology": {
                "topology_id": "A4-Z",
                "active_sites": ["a", "m", "h", "z"],
                "site_gate": {"operator": "one_sided_threshold", "kappa": threshold},
            },
            "peak_memory_reserved_bytes": 60 * 1024**3,
            "boundary_seconds": [6.0],
        })
    assert smoke._exact_target(rows, target, 2048, 32)["all_conditions_fit"] is True
    rows[1]["last_boundary"]["pressure_capture_tensor_count"] = 6
    assert smoke._exact_target(rows, target, 2048, 32)["all_conditions_fit"] is False


def test_verifier_rejects_h_only_event_history(modules, tmp_path):
    run_config, _, _, _, verification = modules
    attempt = tmp_path / "attempt"
    attempt.mkdir()
    event = {
        "event": "train",
        "step": 1,
        "pressure_capture_tensor_count": 6,
        "pressure_capture_names_sha256": "historical-h-only",
    }
    (attempt / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
    original = verification._ORIGINAL_REQUIRE_TRAIN_EVENTS
    verification._ORIGINAL_REQUIRE_TRAIN_EVENTS = lambda *_args: {"boundary_count": 1}
    try:
        with pytest.raises(ValueError, match="Four-site pressure-capture identity mismatch"):
            verification._require_train_events(attempt, {"id": "a4z-ol1-kappa-0"})
        event["pressure_capture_tensor_count"] = 24
        event["pressure_capture_names_sha256"] = run_config.EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256
        (attempt / "events.jsonl").write_text(json.dumps(event) + "\n", encoding="utf-8")
        result = verification._require_train_events(attempt, {"id": "a4z-ol1-kappa-0"})
        assert result["pressure_capture_tensor_count"] == 24
    finally:
        verification._ORIGINAL_REQUIRE_TRAIN_EVENTS = original


def test_run_code_inventory_locks_the_fix_and_historical_source(modules):
    run_config, _, _, _, _ = modules
    paths = [row["path"] for row in run_config.run_code_identity()["files"]]
    assert "run015_capture.py" in paths
    assert "optimizer_boundary.py" in paths
    assert "verification.py" in paths
    assert "../004-2026-08-29-pythia14m-full-pass-l1n/training.py" in paths
    assert "../012-2026-08-30-pythia14m-full-pass-a4-ol1/run_config.py" in paths
    assert "../011-2026-08-30-pythia14m-full-pass-a4z/artifacts/verification.json" in paths
