import importlib.util
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "003-2026-08-29-l1n-lambda5-no-gradient-clipping"


def _run_config():
    spec = importlib.util.spec_from_file_location(
        "run_003_config", RUN_DIR / "run_config.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_two_condition_matrix_and_unclipped_contract_are_locked():
    run_config = _run_config()
    config = run_config.load_config()
    assert config["training"]["gradient_clip_norm"] is None
    assert config["training"]["finite_gradient_check"] is True
    assert config["training"]["max_steps"] == 451
    assert config["training"]["peak_learning_rate"] == 0.004
    assert config["training"]["global_batch_size"] == 64
    assert config["training"]["micro_batch_size"] == 4
    assert config["training"]["gradient_accumulation_steps"] == 16
    assert run_config.condition_specs(config) == [
        {
            "id": "gelu-l1n-5-no-clip",
            "order": 1,
            "activation": "gelu",
            "pressure_method": "l1_naive",
            "pressure_sites": ["h"],
            "pressure_weight": 5.0,
            "gradient_clip_norm": None,
            "label": "gelu lambda=5, no clipping",
        },
        {
            "id": "relu-l1n-5-no-clip",
            "order": 2,
            "activation": "relu",
            "pressure_method": "l1_naive",
            "pressure_sites": ["h"],
            "pressure_weight": 5.0,
            "gradient_clip_norm": None,
            "label": "relu lambda=5, no clipping",
        },
    ]


def test_gate_pressure_and_terminal_diagnostics_are_independent():
    run_config = _run_config()
    config = run_config.load_config()
    gelu, relu = run_config.condition_specs(config)
    gelu_resolved = run_config.resolved_condition_config(config, gelu)
    relu_resolved = run_config.resolved_condition_config(config, relu)
    assert gelu_resolved["model"]["topology_id"] == "A0"
    assert gelu_resolved["model"]["site_gate"] is None
    assert relu_resolved["model"]["topology_id"] == "A1-H"
    assert relu_resolved["model"]["site_gate"] == {"operator": "relu"}
    for resolved in (gelu_resolved, relu_resolved):
        assert resolved["activation_pressure"] == {
            "method": "l1_naive",
            "sites": ["h"],
            "weight": 5.0,
            "step_budget": None,
            "eps": 1e-12,
        }
    assert config["diagnostics"]["activation_sites"] == [
        "h",
        "q_post",
        "k_post",
        "v",
        "m",
        "attention_output",
    ]
    assert config["diagnostics"]["logical_products"] is False
    assert config["diagnostics"]["clipping_frontier"] is None


def test_schedule_and_initialization_comparator_are_exactly_run002_matched():
    run_config = _run_config()
    config = run_config.load_config()
    metadata_path = ROOT / config["data"]["training_metadata"]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    _, schedule_hash, schedule_metadata = run_config.build_schedule(
        config, metadata, np=np
    )
    baseline = json.loads(
        (ROOT / config["comparison"]["baseline_verification"]).read_text(
            encoding="utf-8"
        )
    )
    assert schedule_hash == baseline["training_schedule_sha256"]
    assert schedule_metadata["max_steps"] == 451
    assert config["seeds"] == {"model": 0, "data_order": 0}
    assert run_config.baseline_identity(config)["sha256"] == config["comparison"][
        "baseline_verification_sha256"
    ]


def test_code_inventory_and_monitor_cover_terminal_workflow():
    run_config = _run_config()
    identity = run_config.run_code_identity()
    assert [row["path"] for row in identity["files"]] == [
        "02_train.py",
        "03_verify.py",
        "run_config.py",
        "training.py",
        "verification.py",
        "../../src/sparsity_research/artifacts.py",
        "../../src/sparsity_research/capture.py",
        "../../src/sparsity_research/data.py",
        "../../src/sparsity_research/evaluation.py",
        "../../src/sparsity_research/metrics.py",
        "../../src/sparsity_research/optimization.py",
        "../../src/sparsity_research/pressure.py",
        "../../src/sparsity_research/pythia.py",
        "../../src/sparsity_research/sites.py",
    ]
    monitor = (RUN_DIR / "04_monitor.ps1").read_text(encoding="utf-8")
    assert "Start-Sleep -Seconds $IntervalSeconds" in monitor
    assert "artifacts\\progress.json" in monitor
