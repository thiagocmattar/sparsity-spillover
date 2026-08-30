import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "008-2026-08-29-pythia14m-a7-z-post-mixed-threshold-local"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_config():
    return _load("run_008_config", RUN_DIR / "run_config.py")


def test_five_condition_mixed_threshold_matrix_is_locked():
    run_config = _run_config()
    config = run_config.load_config()
    assert config["model"]["topology_id"] == "A7-Z-POST"
    assert config["model"]["site_gate"] is None
    assert config["conditions"] == {
        "active_sites": ["a", "m", "h", "q_post", "k_post", "v", "z"],
        "one_sided_sites": ["a", "m", "h", "z"],
        "symmetric_sites": ["q_post", "k_post", "v"],
        "gate_thresholds": [0.0, 0.01, 0.05, 0.1, 0.5],
        "pressure_method": "none",
    }
    conditions = run_config.condition_specs(config)
    assert [row["id"] for row in conditions] == [
        "a7-z-post-mixed-kappa-0",
        "a7-z-post-mixed-kappa-0p01",
        "a7-z-post-mixed-kappa-0p05",
        "a7-z-post-mixed-kappa-0p1",
        "a7-z-post-mixed-kappa-0p5",
    ]
    assert [row["gate_threshold"] for row in conditions] == [0.0, 0.01, 0.05, 0.1, 0.5]


def test_every_condition_resolves_approved_per_site_gates_without_pressure():
    run_config = _run_config()
    config = run_config.load_config()
    for condition in run_config.condition_specs(config):
        resolved = run_config.resolved_condition_config(config, condition)
        kappa = condition["gate_threshold"]
        assert resolved["model"]["topology_id"] == "A7-Z-POST"
        assert resolved["model"]["site_gate"] is None
        assert resolved["model"]["site_gates"] == run_config.site_gates(kappa)
        for site in ("a", "m", "h", "z"):
            assert resolved["model"]["site_gates"][site] == {
                "operator": "one_sided_threshold",
                "kappa": kappa,
            }
        for site in ("q_post", "k_post", "v"):
            assert resolved["model"]["site_gates"][site] == {
                "operator": "symmetric_threshold",
                "kappa": kappa,
            }
        assert resolved["activation_pressure"] == {
            "method": "none",
            "sites": [],
            "weight": 0.0,
            "step_budget": None,
            "eps": 1.0e-12,
        }


def test_schedule_exactly_matches_run006_token_budget_and_order():
    run_config = _run_config()
    config = run_config.load_config()
    metadata = json.loads((ROOT / config["data"]["training_metadata"]).read_text(encoding="utf-8"))
    starts, digest, schedule = run_config.build_schedule(config, metadata, np=np)
    assert starts.shape == (581, 16, 4)
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
    assert config["diagnostics"]["gradient_interaction"] is False
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
        calibration = _load("run_008_calibration", RUN_DIR / "calibration.py")
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
    monitor = (RUN_DIR / "05_monitor.ps1").read_text(encoding="utf-8")
    assert "[int]$IntervalSeconds = 1800" in monitor
    assert "task_loss" in monitor
    assert "throughput_tokens_per_second" in monitor
    assert "etc_seconds" in monitor


def test_launch_packet_is_locked_approved_and_matches_calibration():
    run_config = _run_config()
    plan = json.loads((RUN_DIR / "prelaunch" / "launch-plan.json").read_text(encoding="utf-8"))
    calibration_path = RUN_DIR / plan["calibration"]["path"]
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    assert plan["status"] == "approved_for_launch"
    assert plan["launch_approved"] is True
    assert plan["approved_at"] == "2026-08-30T10:00:03.7027576Z"
    assert plan["run_code_content_sha256"] == run_config.run_code_identity()["content_sha256"]
    assert plan["training_schedule_sha256"] == run_config.EXPECTED_SCHEDULE_SHA256
    assert plan["initial_parameter_sha256"] == calibration["initial_parameter_sha256"]
    assert plan["estimated_etc"]["median_seconds"] == calibration["estimated_etc"]["median_seconds"]
    assert plan["estimated_etc"]["p90_seconds"] == calibration["estimated_etc"]["p90_seconds"]
    assert plan["estimated_etc"]["p90_seconds"] < plan["estimated_etc"]["planning_seconds"]
    assert plan["local_resource_fit"]["peak_reserved_bytes"] == calibration["device"]["peak_memory_reserved_bytes"]
    assert plan["architecture_ceiling"]["reachable_product_count"] == 5_638_717_440
