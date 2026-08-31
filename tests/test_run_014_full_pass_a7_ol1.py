import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparsity_research.ceilings import architecture_ceiling
from sparsity_research.pressure import parse_pressure_config


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "014-2026-08-31-pythia14m-full-pass-a7-ol1"
RUN013_DIR = ROOT / "runs" / "013-2026-08-30-pythia14m-full-pass-a7"


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
        "run014_capture",
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
        yield run_config, boundary, smoke, training
    finally:
        sys.path.remove(str(RUN_DIR))
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_five_a7_ol1_conditions_and_parallel_workers_are_locked(modules):
    run_config, _, _, _ = modules
    config = run_config.load_config()
    rows = run_config.condition_specs(config)
    assert [row["id"] for row in rows] == list(run_config.EXPECTED_CONDITION_IDS)
    assert [row["gate_threshold"] for row in rows] == [0.0, 0.01, 0.05, 0.1, 0.5]
    assert all(row["active_sites"] == list(run_config.EXPECTED_ACTIVE_SITES) for row in rows)
    assert all(row["one_sided_sites"] == ["a", "m", "h", "z"] for row in rows)
    assert all(row["symmetric_sites"] == ["q_post", "k_post", "v"] for row in rows)
    assert all(row["pressure_method"] == "orthogonal_l1" for row in rows)
    assert all(row["pressure_sites"] == list(run_config.EXPECTED_ACTIVE_SITES) for row in rows)
    assert all(row["pressure_weight"] == 1.0 and row["step_budget"] == 1.0 for row in rows)
    assert config["runpod"]["pod_count"] == 5
    assert config["runpod"]["parallelism"] == "one_condition_per_gpu"
    assert config["runpod"]["preflight_terminate_after_hours"] == 1.5
    assert config["runpod"]["scientific_terminate_after_hours"] == 2.5
    for row in rows:
        assert run_config.worker_conditions(config, row["id"]) == [row]


def test_every_condition_resolves_mixed_a7_gates_and_seven_site_ol1(modules):
    run_config, _, _, _ = modules
    config = run_config.load_config()
    for row in run_config.condition_specs(config):
        resolved = run_config.resolved_condition_config(config, row)
        assert resolved["model"]["topology_id"] == "A7-Z-POST"
        assert resolved["model"]["site_gate"] is None
        assert resolved["model"]["site_gates"] == run_config.site_gates(
            row["gate_threshold"]
        )
        assert resolved["activation_pressure"] == {
            "method": "orthogonal_l1",
            "sites": list(run_config.EXPECTED_ACTIVE_SITES),
            "weight": 1.0,
            "step_budget": 1.0,
            "eps": 1e-12,
        }


def test_run013_recipe_schedule_and_comparator_are_hash_locked(modules):
    run_config, _, _, _ = modules
    config = run_config.load_config()
    import yaml

    run013 = yaml.safe_load((RUN013_DIR / "config.yaml").read_text(encoding="utf-8"))
    for key in (
        "model",
        "recipe",
        "runtime",
        "data",
        "seeds",
        "validation",
        "checkpoints",
        "artifacts",
    ):
        assert config[key] == run013[key]
    actual_training = dict(config["training"])
    assert actual_training.pop("fp16_ol1_overflow_policy") == "skip_entire_boundary"
    assert actual_training == run013["training"]
    actual_diagnostics = dict(config["diagnostics"])
    assert actual_diagnostics.pop("ol1_adaptive_directions") is True
    assert actual_diagnostics.pop("gradient_interaction") is True
    expected_diagnostics = dict(run013["diagnostics"])
    assert expected_diagnostics.pop("gradient_interaction") is False
    assert actual_diagnostics == expected_diagnostics

    metadata = json.loads((ROOT / config["data"]["training_metadata"]).read_text())
    _, digest, schedule = run_config.build_schedule(config, metadata, np=np)
    assert digest == run_config.EXPECTED_SCHEDULE_SHA256
    assert schedule["scheduled_blocks"] == 729_088
    assert schedule["wrapped_blocks"] == 714

    source = json.loads(run_config.RUN013_VERIFICATION.read_text(encoding="utf-8"))
    assert source["status"] == "verified"
    assert source["evidence_label"] == "valid"
    assert source["initial_parameter_sha256"] == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    assert source["training_schedule_sha256"] == run_config.EXPECTED_SCHEDULE_SHA256
    assert source["run_code_sha256"] == run_config.EXPECTED_RUN013_CODE_SHA256
    assert [row["condition"]["id"] for row in source["conditions"]] == list(
        run_config.EXPECTED_RUN013_CONDITION_IDS
    )


def test_training_adapter_replaces_frozen_h_capture_with_all_seven_sites(modules):
    run_config, _, _, training = modules
    capture = training.SevenSitePressureCapture(object(), ["h"], torch=torch)
    assert tuple(capture.sites) == run_config.EXPECTED_ACTIVE_SITES
    with pytest.raises(RuntimeError, match="frozen Run 004 capture call changed"):
        training.SevenSitePressureCapture(object(), ["a"], torch=torch)


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
        for site in ("a", "m", "h", "q_post", "k_post", "v", "z"):
            if site != self.omit_site:
                self.capture.activations[f"{site}.layer_0"] = hidden
        return SimpleNamespace(loss=(hidden - 0.25).square().mean())


def _pressure(run_config):
    return parse_pressure_config({
        "method": "orthogonal_l1",
        "sites": list(run_config.EXPECTED_ACTIVE_SITES),
        "weight": 1.0,
        "step_budget": 0.2,
    })


def test_boundary_differentiates_exact_realized_seven_site_objective(modules):
    run_config, boundary, _, _ = modules
    capture = _Capture()
    model = _ToyModel(capture)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    )
    result = boundary.run_recipe_boundary(
        model=model,
        optimizer=optimizer,
        batches=[torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 1.0]])],
        pressure=_pressure(run_config),
        capture=capture,
        loss_scaler=boundary.DynamicLossScaler(scale=8.0),
        gradient_clip_norm=0.05,
        torch=torch,
        device=torch.device("cpu"),
    )
    assert result["optimizer_step_skipped"] is False
    assert result["ol1_correction_applied"] is True
    assert result["pressure_capture_tensor_count"] == 7
    assert result["pressure_to_task_ratio_final"] <= 0.2 + 1e-9

    broken_capture = _Capture()
    broken = _ToyModel(broken_capture, omit_site="v")
    with pytest.raises(RuntimeError, match="pressure capture mismatch"):
        boundary.run_recipe_boundary(
            model=broken,
            optimizer=torch.optim.AdamW(broken.parameters(), lr=0.01),
            batches=[torch.tensor([[1.0, 2.0]])],
            pressure=_pressure(run_config),
            capture=broken_capture,
            loss_scaler=boundary.DynamicLossScaler(scale=8.0),
            gradient_clip_norm=1.0,
            torch=torch,
            device=torch.device("cpu"),
        )


def test_exact_preflight_requires_mixed_gates_capture_identity_and_headroom(modules):
    run_config, _, smoke, _ = modules
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
            "boundary_health": [
                {"optimizer_step_skipped": False, "gradient_overflow": False}
            ],
            "last_boundary": {
                "task_loss": 5.0,
                "pressure_loss": 0.2,
                "adamw_gradient_norm_post_clip": 1.0,
                "pressure_to_task_ratio_raw": 1.2,
                "pressure_to_task_ratio_final": 1.0,
                "trust_scale": 0.8,
                "pressure_weight": 1.0,
                "ol1_correction_applied": True,
                "pressure_capture_tensor_count": 42,
                "pressure_capture_names_sha256": (
                    run_config.EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256
                ),
            },
            "topology": {
                "topology_id": "A7-Z-POST",
                "active_sites": list(run_config.EXPECTED_ACTIVE_SITES),
                "site_gate": None,
                "site_gates": run_config.site_gates(threshold),
            },
            "peak_memory_reserved_bytes": 60 * 1024**3,
            "boundary_seconds": [6.0],
        })
    assert smoke._exact_target(rows, target, 2048, 32)["all_conditions_fit"] is True
    rows[1]["last_boundary"]["pressure_capture_tensor_count"] = 6
    assert smoke._exact_target(rows, target, 2048, 32)["all_conditions_fit"] is False


def test_a7_integer_ceiling_matches_locked_contract(modules):
    run_config, _, _, _ = modules
    result = architecture_ceiling(
        "A7-Z-POST",
        layers=6,
        hidden_size=128,
        ffn_size=512,
        sequence_length=2048,
        vocabulary_size=50304,
    )
    assert result["reachable_product_count"] == run_config.EXPECTED_CEILING_NUMERATOR
    assert result["model_product_count"] == run_config.EXPECTED_CEILING_DENOMINATOR
    assert result["R_model_max_fraction"] == pytest.approx(
        run_config.EXPECTED_CEILING_FRACTION, abs=1e-15
    )


def test_run_code_inventory_covers_adapter_boundary_and_comparator(modules):
    run_config, _, _, _ = modules
    paths = [row["path"] for row in run_config.run_code_identity()["files"]]
    assert "training.py" in paths
    assert "optimizer_boundary.py" in paths
    assert "verification.py" in paths
    assert "../004-2026-08-29-pythia14m-full-pass-l1n/training.py" in paths
    assert "../013-2026-08-30-pythia14m-full-pass-a7/artifacts/verification.json" in paths
    assert "../../src/sparsity_research/capture.py" in paths
