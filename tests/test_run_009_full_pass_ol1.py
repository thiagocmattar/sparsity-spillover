import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sparsity_research.pressure import parse_pressure_config


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "009-2026-08-30-pythia14m-full-pass-ol1"
RUN004_DIR = ROOT / "runs" / "004-2026-08-29-pythia14m-full-pass-l1n"


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
        "_reuse_run004", "run_config", "initialization", "optimizer_boundary",
        "diagnostics", "smoke", "training", "verification",
    )
    previous = {name: sys.modules.get(name) for name in names}
    sys.path.insert(0, str(RUN_DIR))
    try:
        run_config = _module("run_config")
        boundary = _module("optimizer_boundary")
        smoke = _module("smoke")
        yield run_config, boundary, smoke
    finally:
        sys.path.remove(str(RUN_DIR))
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_four_condition_grid_and_parallel_workers_are_locked(modules):
    run_config, _, _ = modules
    config = run_config.load_config()
    rows = run_config.condition_specs(config)
    assert [row["id"] for row in rows] == list(run_config.EXPECTED_CONDITION_IDS)
    assert [row["pressure_weight"] for row in rows] == [0.05, 0.1, 0.5, 1.0]
    assert all(row["pressure_method"] == "orthogonal_l1" for row in rows)
    assert all(row["step_budget"] == 1.0 for row in rows)
    assert all(not row["is_control"] for row in rows)
    assert config["conditions"]["include_controls"] is False
    assert config["runpod"]["pod_count"] == 4
    for row in rows:
        assert run_config.worker_conditions(config, row["id"]) == [row]


def test_every_condition_resolves_relu_h_only_ol1(modules):
    run_config, _, _ = modules
    config = run_config.load_config()
    for row in run_config.condition_specs(config):
        resolved = run_config.resolved_condition_config(config, row)
        assert resolved["model"]["topology_id"] == "A1-H"
        assert resolved["model"]["site_gate"] == {"operator": "relu"}
        assert resolved["activation_pressure"] == {
            "method": "orthogonal_l1",
            "sites": ["h"],
            "weight": row["pressure_weight"],
            "step_budget": 1.0,
            "eps": 1e-12,
        }


def test_run004_recipe_schedule_diagnostics_and_checkpoints_are_matched(modules):
    run_config, _, _ = modules
    config = run_config.load_config()
    import yaml

    run004 = yaml.safe_load((RUN004_DIR / "config.yaml").read_text(encoding="utf-8"))
    for key in ("model", "recipe", "runtime", "data", "seeds", "validation", "checkpoints", "artifacts"):
        assert config[key] == run004[key]
    actual_diagnostics = dict(config["diagnostics"])
    assert actual_diagnostics.pop("ol1_adaptive_directions") is True
    assert actual_diagnostics == run004["diagnostics"]
    expected_training = dict(run004["training"])
    actual_training = dict(config["training"])
    assert actual_training.pop("fp16_ol1_overflow_policy") == "skip_entire_boundary"
    assert actual_training == expected_training
    metadata = json.loads((ROOT / config["data"]["training_metadata"]).read_text(encoding="utf-8"))
    _, digest, schedule = run_config.build_schedule(config, metadata, np=np)
    assert digest == run_config.EXPECTED_SCHEDULE_SHA256
    assert schedule["scheduled_blocks"] == 729_088
    assert schedule["wrapped_blocks"] == 714


def test_run004_comparators_are_valid_and_hash_locked(modules):
    run_config, _, _ = modules
    source = json.loads(run_config.RUN004_VERIFICATION.read_text(encoding="utf-8"))
    assert source["status"] == "verified"
    assert source["evidence_label"] == "valid"
    assert source["initial_parameter_sha256"] == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    assert source["training_schedule_sha256"] == run_config.EXPECTED_SCHEDULE_SHA256
    assert source["run_code_sha256"] == run_config.EXPECTED_RUN004_CODE_SHA256


class _Capture:
    def __init__(self):
        self.activations = {}

    def clear(self):
        self.activations.clear()


class _ToyModel(torch.nn.Module):
    def __init__(self, capture):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
        self.capture = capture

    def forward(self, input_ids, labels):
        hidden = input_ids.float() * self.weight
        self.capture.activations["h.layer_0"] = hidden
        return SimpleNamespace(loss=(hidden - 0.25).square().mean())


def test_fp16_recipe_boundary_clips_task_only_then_applies_ol1(modules):
    _, boundary, _ = modules
    capture = _Capture()
    model = _ToyModel(capture)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.01, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0
    )
    pressure = parse_pressure_config({
        "method": "orthogonal_l1", "sites": ["h"], "weight": 1.0,
        "step_budget": 0.2,
    })
    result = boundary.run_recipe_boundary(
        model=model,
        optimizer=optimizer,
        batches=[torch.tensor([[1.0, 2.0]]), torch.tensor([[2.0, 1.0]])],
        pressure=pressure,
        capture=capture,
        loss_scaler=boundary.DynamicLossScaler(scale=8.0),
        gradient_clip_norm=0.05,
        torch=torch,
        device=torch.device("cpu"),
    )
    assert result["optimizer_step_skipped"] is False
    assert result["gradient_overflow"] is False
    assert result["fp16_overflow_policy"] == "skip_entire_boundary"
    assert result["ol1_correction_applied"] is True
    assert result["adamw_gradient_norm_pre_clip"] > 0.05
    assert result["adamw_gradient_norm_post_clip"] == pytest.approx(0.05)
    assert result["task_gradient_norm"] == pytest.approx(0.05)
    assert result["pressure_to_task_ratio_final"] <= 0.2 + 1e-9
    assert optimizer.state[model.weight]["exp_avg"].norm().item() == pytest.approx(0.005)


class _InfiniteGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.sum()

    @staticmethod
    def backward(ctx, gradient):
        return torch.full((2,), float("inf"))


class _OverflowModel(torch.nn.Module):
    def __init__(self, capture):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        self.capture = capture

    def forward(self, input_ids, labels):
        self.capture.activations["h.layer_0"] = self.weight * input_ids.float()
        return SimpleNamespace(loss=_InfiniteGradient.apply(self.weight))


def test_nonfinite_component_gradient_skips_adamw_and_ol1_atomically(modules):
    _, boundary, _ = modules
    capture = _Capture()
    model = _OverflowModel(capture)
    before = model.weight.detach().clone()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.0)
    pressure = parse_pressure_config({
        "method": "orthogonal_l1", "sites": ["h"], "weight": 1.0,
        "step_budget": 1.0,
    })
    result = boundary.run_recipe_boundary(
        model=model,
        optimizer=optimizer,
        batches=[torch.tensor([[1.0, 1.0]])],
        pressure=pressure,
        capture=capture,
        loss_scaler=boundary.DynamicLossScaler(scale=8.0),
        gradient_clip_norm=1.0,
        torch=torch,
        device=torch.device("cpu"),
    )
    assert result["optimizer_step_skipped"] is True
    assert result["gradient_overflow"] is True
    assert result["ol1_correction_applied"] is False
    assert optimizer.state == {}
    assert torch.equal(model.weight.detach(), before)


def test_smoke_exact_target_requires_healthy_trust_bounded_ol1(modules):
    _, _, smoke = modules
    target = {
        "micro_batch_size": 32,
        "sequence_length": 2048,
        "gradient_accumulation_steps": 32,
        "memory_bytes": 80 * 1024**3,
    }
    row = {
        "condition_id": smoke.CONDITION_ID,
        "batch_size": 32,
        "sequence_length": 2048,
        "gradient_accumulation_steps": 32,
        "status": "completed",
        "boundary_health": [{"optimizer_step_skipped": False, "gradient_overflow": False}],
        "last_boundary": {
            "ol1_correction_applied": True,
            "pressure_to_task_ratio_final": 1.0,
        },
        "peak_memory_reserved_bytes": 60 * 1024**3,
        "boundary_seconds": [6.0],
    }
    exact = smoke._exact_target([row], target, 2048, 32)
    assert exact["all_conditions_fit"] is True
    row["last_boundary"]["pressure_to_task_ratio_final"] = 1.01
    assert smoke._exact_target([row], target, 2048, 32)["all_conditions_fit"] is False


def test_run_code_inventory_includes_all_reused_behavior(modules):
    run_config, _, _ = modules
    paths = [row["path"] for row in run_config.run_code_identity()["files"]]
    assert "_reuse_run004.py" in paths
    assert "../004-2026-08-29-pythia14m-full-pass-l1n/training.py" in paths
    assert "../004-2026-08-29-pythia14m-full-pass-l1n/optimizer_boundary.py" in paths
    assert "../../src/sparsity_research/pressure.py" in paths
