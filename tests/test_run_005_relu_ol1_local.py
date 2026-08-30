import importlib.util
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "005-2026-08-29-pythia14m-relu-ol1-local"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _run_config():
    return _load("run_005_config", RUN_DIR / "run_config.py")


def test_five_condition_relu_ol1_matrix_is_locked():
    run_config = _run_config()
    config = run_config.load_config()
    assert config["model"]["architecture"] == "EleutherAI/pythia-14m-deduped"
    assert config["model"]["topology_id"] == "A1-H"
    assert config["model"]["site_gate"] == {"operator": "relu"}
    assert config["training"]["peak_learning_rate"] == 1.0e-3
    assert config["training"]["target_cohort_seconds"] == 3600
    assert config["training"]["planning_cohort_seconds"] == 3300
    conditions = run_config.condition_specs(config)
    assert [row["id"] for row in conditions] == [
        "relu-control",
        "relu-ol1-0p01",
        "relu-ol1-0p1",
        "relu-ol1-0p5",
        "relu-ol1-1",
    ]
    assert [row["pressure_weight"] for row in conditions] == [0.0, 0.01, 0.1, 0.5, 1.0]


def test_control_and_ol1_pressure_are_resolved_independently_from_relu_gate():
    run_config = _run_config()
    config = run_config.load_config()
    control, pressured = run_config.condition_specs(config)[:2]
    control_config = run_config.resolved_condition_config(config, control)
    pressured_config = run_config.resolved_condition_config(config, pressured)
    assert control_config["model"]["topology_id"] == "A1-H"
    assert control_config["activation_pressure"] == {
        "method": "none",
        "sites": [],
        "weight": 0.0,
        "step_budget": None,
        "eps": 1.0e-12,
    }
    assert pressured_config["activation_pressure"] == {
        "method": "orthogonal_l1",
        "sites": ["h"],
        "weight": 0.01,
        "step_budget": 1.0,
        "eps": 1.0e-12,
    }


def test_schedule_validation_diagnostics_and_retention_are_locked():
    run_config = _run_config()
    config = run_config.load_config()
    metadata = json.loads(
        (ROOT / config["data"]["training_metadata"]).read_text(encoding="utf-8")
    )
    _, _, schedule = run_config.build_schedule(config, metadata, np=np)
    assert schedule["max_steps"] == config["training"]["max_steps"]
    assert schedule["wrapped_blocks"] == 0
    assert config["training"]["global_batch_size"] == 64
    assert config["training"]["micro_batch_size"] == 4
    assert config["training"]["gradient_accumulation_steps"] == 16
    assert config["validation"] == {
        "documents": 500,
        "complete_sequences": 338,
        "input_tokens": 692224,
        "excluded_tail_tokens": 1444,
        "batch_size": 4,
        "at_step_one": True,
        "at_final_step": True,
    }
    assert config["diagnostics"]["logical_products"] is True
    assert config["diagnostics"]["logical_product_batch_size"] == 1
    assert config["artifacts"] == {
        "save_final_checkpoint": True,
        "save_optimizer": False,
        "retain_predictions": False,
    }


def test_calibration_budget_accounts_for_one_control_and_four_ol1_conditions():
    saved = {name: sys.modules.get(name) for name in ("run_config", "diagnostics", "training")}
    sys.path.insert(0, str(RUN_DIR))
    try:
        run_config = _load("run_config", RUN_DIR / "run_config.py")
        sys.modules["run_config"] = run_config
        diagnostics = _load("diagnostics", RUN_DIR / "diagnostics.py")
        sys.modules["diagnostics"] = diagnostics
        training = _load("training", RUN_DIR / "training.py")
        sys.modules["training"] = training
        calibration = _load("run_005_calibration", RUN_DIR / "calibration.py")
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
        "condition_counts": {"control": 1, "ol1_pressure": 4},
        "samples": {"control": sample, "ol1_pressure": sample},
    }
    steps = calibration.budget_steps_from_calibration(measured, planning_seconds=1000.0)
    estimate = calibration.estimate_cohort(measured, common_steps=steps)
    assert estimate["condition_count"] == 5
    assert estimate["total_optimizer_steps"] == 5 * steps
    assert estimate["p90_seconds"] <= 1000.0


def test_code_inventory_and_monitor_cover_prelaunch_and_terminal_paths():
    run_config = _run_config()
    paths = [row["path"] for row in run_config.run_code_identity()["files"]]
    assert paths[:10] == [
        "01_calibrate.py",
        "02_train.py",
        "03_verify.py",
        "04_launch.ps1",
        "05_monitor.ps1",
        "calibration.py",
        "diagnostics.py",
        "run_config.py",
        "training.py",
        "verification.py",
    ]
    assert "../../src/sparsity_research/logical_capture.py" in paths
    launcher = (RUN_DIR / "04_launch.ps1").read_text(encoding="utf-8")
    assert "-WindowStyle Hidden" in launcher
    assert "cohort.stdout.log" in launcher
    monitor = (RUN_DIR / "05_monitor.ps1").read_text(encoding="utf-8")
    assert "Start-Sleep -Seconds $IntervalSeconds" in monitor
    assert "artifacts\\progress.json" in monitor
    assert "latest_event" in monitor
    assert "event_file_age_seconds" in monitor
