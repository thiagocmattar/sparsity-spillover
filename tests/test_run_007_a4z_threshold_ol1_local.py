import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "007-2026-08-29-pythia14m-a4z-threshold-ol1-local"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_config():
    return _load("run_007_config", RUN_DIR / "run_config.py")


def test_five_condition_a4z_threshold_ol1_matrix_is_locked():
    run_config = _run_config()
    config = run_config.load_config()
    assert config["model"]["topology_id"] == "A4-Z"
    assert config["model"]["site_gate"] == {
        "operator": "one_sided_threshold",
        "kappa": 0.0,
    }
    assert config["conditions"]["active_sites"] == ["a", "m", "h", "z"]
    assert config["conditions"]["pressure_method"] == "orthogonal_l1"
    assert config["conditions"]["pressure_sites"] == ["a", "m", "h", "z"]
    assert config["conditions"]["pressure_weight"] == 1.0
    assert config["conditions"]["step_budget"] == 1.0
    assert config["training"]["peak_learning_rate"] == 1.0e-3
    conditions = run_config.condition_specs(config)
    assert [row["id"] for row in conditions] == [
        "a4z-one-sided-kappa-0-ol1-1",
        "a4z-one-sided-kappa-0p01-ol1-1",
        "a4z-one-sided-kappa-0p05-ol1-1",
        "a4z-one-sided-kappa-0p1-ol1-1",
        "a4z-one-sided-kappa-0p5-ol1-1",
    ]
    assert [row["gate_threshold"] for row in conditions] == [0.0, 0.01, 0.05, 0.1, 0.5]


def test_every_condition_resolves_gate_with_all_site_post_threshold_ol1():
    run_config = _run_config()
    config = run_config.load_config()
    for condition in run_config.condition_specs(config):
        resolved = run_config.resolved_condition_config(config, condition)
        assert resolved["model"]["topology_id"] == "A4-Z"
        assert resolved["model"]["site_gate"] == {
            "operator": "one_sided_threshold",
            "kappa": condition["gate_threshold"],
        }
        assert resolved["activation_pressure"] == {
            "method": "orthogonal_l1",
            "sites": ["a", "m", "h", "z"],
            "weight": 1.0,
            "step_budget": 1.0,
            "eps": 1.0e-12,
        }


def test_schedule_exactly_matches_run006_token_budget_and_order():
    run_config = _run_config()
    config = run_config.load_config()
    metadata = json.loads((ROOT / config["data"]["training_metadata"]).read_text(encoding="utf-8"))
    starts, digest, schedule = run_config.build_schedule(config, metadata, np=np)
    assert starts.shape == (581, 16, 4)
    assert digest == run_config.EXPECTED_SCHEDULE_SHA256
    assert digest == "c254893f0ea521e5834405d7a4e6edaed74472733d533aff68fb119e600151d4"
    assert schedule["scheduled_blocks"] == 581 * 64
    assert schedule["wrapped_blocks"] == 0
    assert 581 * 64 * 2048 == 76_152_832
    assert 5 * 581 * 64 * 2048 == 380_764_160


def test_validation_diagnostics_and_retention_are_locked():
    run_config = _run_config()
    config = run_config.load_config()
    assert config["validation"] == {
        "documents": 500,
        "complete_sequences": 338,
        "input_tokens": 692224,
        "excluded_tail_tokens": 1444,
        "batch_size": 4,
        "at_step_one": True,
        "at_final_step": True,
    }
    assert config["diagnostics"]["activation_sites"] == [
        "a", "m", "h", "q_post", "k_post", "v", "z", "attention_output"
    ]
    assert config["diagnostics"]["gradient_interaction"] is True
    assert config["diagnostics"]["logical_products"] is True
    assert config["diagnostics"]["clipping_frontier"] is None
    assert config["artifacts"] == {
        "save_final_checkpoint": True,
        "save_optimizer": False,
        "retain_predictions": False,
    }


def test_calibration_estimate_accounts_for_all_five_threshold_conditions():
    saved = {name: sys.modules.get(name) for name in ("run_config", "diagnostics", "training")}
    sys.path.insert(0, str(RUN_DIR))
    try:
        run_config = _load("run_config", RUN_DIR / "run_config.py")
        sys.modules["run_config"] = run_config
        diagnostics = _load("diagnostics", RUN_DIR / "diagnostics.py")
        sys.modules["diagnostics"] = diagnostics
        training = _load("training", RUN_DIR / "training.py")
        sys.modules["training"] = training
        calibration = _load("run_007_calibration", RUN_DIR / "calibration.py")
    finally:
        sys.path.remove(str(RUN_DIR))
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    sample = {
        "model_setup_seconds": 1.0,
        "ordinary_full_validation_seconds": 2.0,
        "activation_diagnostic_seconds": 3.0,
        "logical_product_diagnostic_seconds": 4.0,
        "weight_statistics_seconds": 1.0,
        "checkpoint_save_hash_reload_seconds": 2.0,
        "optimizer_step_seconds": [1.0, 1.1, 1.2, 1.3],
    }
    measured = {
        "cache_verification_seconds": 5.0,
        "terminal_headroom_seconds": 25.0,
        "condition_counts": {"kappa_0": 1, "kappa_0p5": 4},
        "samples": {"kappa_0": sample, "kappa_0p5": sample},
    }
    estimate = calibration.estimate_cohort(measured, common_steps=581)
    assert estimate["condition_count"] == 5
    assert estimate["total_optimizer_steps"] == 2905
    assert estimate["p90_seconds"] > estimate["median_fixed_seconds"]


def test_training_boundary_contract_requires_ol1_metrics_and_budget():
    saved = {name: sys.modules.get(name) for name in ("run_config", "diagnostics")}
    sys.path.insert(0, str(RUN_DIR))
    try:
        run_config = _load("run_config", RUN_DIR / "run_config.py")
        sys.modules["run_config"] = run_config
        diagnostics = _load("diagnostics", RUN_DIR / "diagnostics.py")
        sys.modules["diagnostics"] = diagnostics
        training = _load("run_007_training", RUN_DIR / "training.py")
    finally:
        sys.path.remove(str(RUN_DIR))
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module

    boundary = {
        "adamw_gradient_clipping_enabled": True,
        "adamw_gradient_clip_norm": 1.0,
        "pressure_loss": 0.25,
        "pressure_gradient_norm": 0.5,
        "task_pressure_gradient_dot": -0.1,
        "task_direction_norm": 1.0,
        "pressure_direction_norm_raw": 2.0,
        "task_pressure_dot_before": -0.1,
        "task_pressure_dot_after": 0.0,
        "projection_applied": True,
        "pressure_to_task_ratio_raw": 2.0,
        "trust_scale": 0.5,
        "pressure_to_task_ratio_final": 1.0,
    }
    training._require_boundary_contract(boundary)
    boundary["pressure_to_task_ratio_final"] = 1.01
    with pytest.raises(RuntimeError, match="trust budget"):
        training._require_boundary_contract(boundary)


def test_code_inventory_launch_provenance_and_monitor_are_locked():
    run_config = _run_config()
    paths = [row["path"] for row in run_config.run_code_identity()["files"]]
    assert paths[:11] == [
        "01_calibrate.py", "02_train.py", "03_verify.py", "04_launch.ps1",
        "05_monitor.ps1", "calibration.py", "diagnostics.py", "launch_detached.py",
        "run_config.py", "training.py", "verification.py",
    ]
    assert "../../src/sparsity_research/sites.py" in paths
    launcher = (RUN_DIR / "04_launch.ps1").read_text(encoding="utf-8")
    assert "launch_approved" in launcher
    assert "launch-provenance.json" in launcher
    assert "launch_detached.py" in launcher
    detached_launcher = (RUN_DIR / "launch_detached.py").read_text(encoding="utf-8")
    assert "subprocess.DETACHED_PROCESS" in detached_launcher
    assert "subprocess.CREATE_NEW_PROCESS_GROUP" in detached_launcher
    monitor = (RUN_DIR / "05_monitor.ps1").read_text(encoding="utf-8")
    assert "[int]$IntervalSeconds = 1800" in monitor
    assert "[datetimeoffset]::Parse" in monitor
    assert "task_loss" in monitor
    assert "throughput_tokens_per_second" in monitor
    assert "etc_seconds" in monitor


def test_launch_packet_is_locked_approved_and_matches_the_design():
    plan = json.loads(
        (RUN_DIR / "prelaunch" / "launch-plan.json").read_text(encoding="utf-8")
    )
    assert plan["status"] == "approved_for_launch"
    assert plan["launch_approved"] is True
    assert plan["approved_at"] == "2026-08-29T22:28:05.1761202Z"
    definition = plan["scientific_definition"]
    assert definition["gate_thresholds"] == [0.0, 0.01, 0.05, 0.1, 0.5]
    assert definition["pressure_method"] == "orthogonal_l1"
    assert definition["pressure_sites"] == ["a", "m", "h", "z"]
    assert definition["pressure_placement"] == "post-threshold outputs in all six layers"
    assert definition["pressure_weight"] == 1.0
    assert definition["step_budget"] == 1.0
    assert definition["training_input_tokens_per_condition"] == 76_152_832
    assert plan["estimated_etc"]["p90_seconds"] < plan["estimated_etc"]["hard_ceiling_seconds"]
    assert plan["monitoring"]["interval_seconds"] == 1800
    calibration_path = RUN_DIR / plan["calibration"]["path"]
    assert calibration_path.is_file()
    assert not (RUN_DIR / "attempts").exists()
