import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from sparsity_research.ceilings import architecture_ceiling


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "013-2026-08-30-pythia14m-full-pass-a7"
RUN011_DIR = ROOT / "runs" / "011-2026-08-30-pythia14m-full-pass-a4z"


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
        "_reuse_run011",
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
        smoke = _module("smoke")
        yield run_config, smoke
    finally:
        sys.path.remove(str(RUN_DIR))
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def test_five_condition_grid_and_parallel_workers_are_locked(modules):
    run_config, _ = modules
    config = run_config.load_config()
    rows = run_config.condition_specs(config)
    assert [row["id"] for row in rows] == list(run_config.EXPECTED_CONDITION_IDS)
    assert [row["gate_threshold"] for row in rows] == [0.0, 0.01, 0.05, 0.1, 0.5]
    assert all(row["active_sites"] == list(run_config.EXPECTED_ACTIVE_SITES) for row in rows)
    assert all(row["one_sided_sites"] == ["a", "m", "h", "z"] for row in rows)
    assert all(row["symmetric_sites"] == ["q_post", "k_post", "v"] for row in rows)
    assert all(row["pressure_method"] == "none" for row in rows)
    assert config["runpod"]["pod_count"] == 5
    assert config["runpod"]["parallelism"] == "one_condition_per_gpu"
    assert config["runpod"]["preflight_terminate_after_hours"] == 1.5
    assert config["runpod"]["scientific_terminate_after_hours"] == 2.5
    for row in rows:
        assert run_config.worker_conditions(config, row["id"]) == [row]


def test_every_condition_resolves_exact_mixed_a7_gates_without_pressure(modules):
    run_config, _ = modules
    config = run_config.load_config()
    for row in run_config.condition_specs(config):
        resolved = run_config.resolved_condition_config(config, row)
        assert resolved["model"]["topology_id"] == "A7-Z-POST"
        assert resolved["model"]["site_gate"] is None
        assert resolved["model"]["site_gates"] == run_config.site_gates(
            row["gate_threshold"]
        )
        assert resolved["activation_pressure"] == {
            "method": "none",
            "sites": [],
            "weight": 0.0,
            "step_budget": None,
            "eps": 1e-12,
        }


def test_run011_recipe_schedule_validation_diagnostics_and_checkpoints_are_matched(modules):
    run_config, _ = modules
    config = run_config.load_config()
    import yaml

    run011 = yaml.safe_load((RUN011_DIR / "config.yaml").read_text(encoding="utf-8"))
    for key in ("recipe", "runtime", "data", "seeds", "training", "validation", "diagnostics", "checkpoints"):
        assert config[key] == run011[key]
    actual_model = dict(config["model"])
    actual_model.pop("topology_id")
    actual_model.pop("site_gate")
    actual_model.pop("site_gates")
    source_model = dict(run011["model"])
    source_model.pop("topology_id")
    source_model.pop("site_gate")
    assert actual_model == source_model
    metadata = json.loads((ROOT / config["data"]["training_metadata"]).read_text(encoding="utf-8"))
    _, digest, schedule = run_config.build_schedule(config, metadata, np=np)
    assert digest == run_config.EXPECTED_SCHEDULE_SHA256
    assert schedule["scheduled_blocks"] == 729_088
    assert schedule["wrapped_blocks"] == 714


def test_run011_matched_a4_comparator_is_valid_and_hash_locked(modules):
    run_config, _ = modules
    source = json.loads(run_config.RUN011_VERIFICATION.read_text(encoding="utf-8"))
    assert source["status"] == "verified"
    assert source["evidence_label"] == "valid"
    assert source["condition_count"] == 5
    assert source["initial_parameter_sha256"] == run_config.EXPECTED_INITIAL_PARAMETER_SHA256
    assert source["training_schedule_sha256"] == run_config.EXPECTED_SCHEDULE_SHA256
    assert source["run_code_sha256"] == run_config.EXPECTED_RUN011_CODE_SHA256
    assert [row["condition"]["id"] for row in source["conditions"]] == list(
        run_config.EXPECTED_A4_CONDITION_IDS
    )


def test_a7_integer_ceiling_matches_locked_contract(modules):
    run_config, _ = modules
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


def test_exact_preflight_requires_both_mixed_gate_endpoints_and_headroom(modules):
    run_config, smoke = modules
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
                "adamw_gradient_norm_post_clip": 1.0,
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
    exact = smoke._exact_target(rows, target, 2048, 32)
    assert exact["all_conditions_fit"] is True
    assert smoke._exact_target(rows[:1], target, 2048, 32)["all_conditions_fit"] is False
    rows[1]["peak_memory_reserved_bytes"] = 73 * 1024**3
    assert smoke._exact_target(rows, target, 2048, 32)["all_conditions_fit"] is False


def test_run_code_inventory_includes_reused_and_mixed_gate_behavior(modules):
    run_config, _ = modules
    paths = [row["path"] for row in run_config.run_code_identity()["files"]]
    assert "_reuse_run004.py" in paths
    assert "_reuse_run011.py" in paths
    assert "../004-2026-08-29-pythia14m-full-pass-l1n/training.py" in paths
    assert "../011-2026-08-30-pythia14m-full-pass-a4z/run_config.py" in paths
    assert "../../src/sparsity_research/pythia.py" in paths
    assert "verification.py" in paths
