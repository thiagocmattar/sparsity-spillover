import importlib.util
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "001-2026-08-28-local-lr-pipeline-calibration"


def _pipeline():
    spec = importlib.util.spec_from_file_location("run_001_pipeline", RUN_DIR / "pipeline.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_approved_grid_batch_and_complete_validation_are_locked():
    pipeline = _pipeline()
    config = pipeline.load_config(RUN_DIR / "config.yaml")
    assert config["conditions"]["peak_learning_rates"] == [0.0005, 0.001, 0.002, 0.004]
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


def test_resolved_conditions_change_only_the_named_peak_lr_fields():
    pipeline = _pipeline()
    config = pipeline.load_config(RUN_DIR / "config.yaml")
    first = pipeline.resolved_condition_config(config, 0.0005)
    second = pipeline.resolved_condition_config(config, 0.001)
    assert first["condition"] == {"id": "lr-5e-4", "peak_learning_rate": 0.0005}
    assert second["condition"] == {"id": "lr-1e-3", "peak_learning_rate": 0.001}
    first["condition"] = second["condition"]
    first["training"]["peak_learning_rate"] = second["training"]["peak_learning_rate"]
    assert first == second


def test_common_step_budget_uses_p90_and_preserves_equal_steps():
    pipeline = _pipeline()
    calibration = {
        "condition_count": 4,
        "validation_passes_per_condition": 2,
        "optimizer_step_seconds": [1.0, 1.1, 1.2, 1.3, 1.4],
        "full_validation_seconds": [2.0, 2.2, 2.4],
        "setup_seconds": 20.0,
        "diagnostics_seconds": 10.0,
        "checkpoint_seconds": 10.0,
    }
    steps = pipeline.budget_steps_from_calibration(calibration, planning_seconds=1000.0)
    estimate = pipeline.estimate_cohort(calibration, common_steps=steps)
    assert steps == 168
    assert estimate["total_optimizer_steps"] == 4 * steps
    assert estimate["p90_seconds"] <= 1000.0


def test_microbatch_materialization_follows_explicit_block_starts():
    pipeline = _pipeline()
    tokens = np.arange(40, dtype=np.int32)
    starts = np.array([[0, 4], [8, 12]], dtype=np.int64)
    batches = pipeline.microbatches_for_step(
        tokens,
        starts,
        block_size=4,
        device=torch.device("cpu"),
        torch=torch,
        np=np,
    )
    assert [batch.tolist() for batch in batches] == [
        [[0, 1, 2, 3], [4, 5, 6, 7]],
        [[8, 9, 10, 11], [12, 13, 14, 15]],
    ]


def test_locked_launch_schedule_matches_config_and_launch_plan():
    pipeline = _pipeline()
    config = pipeline.load_config(RUN_DIR / "config.yaml")
    starts, schedule_hash, metadata = pipeline.build_training_schedule(
        np,
        token_count=1_491_711_416,
        block_size=2048,
        max_steps=config["training"]["max_steps"],
        gradient_accumulation_steps=config["training"]["gradient_accumulation_steps"],
        micro_batch_size=config["training"]["micro_batch_size"],
        seed=config["seeds"]["data_order"],
    )
    assert starts.shape == (449, 16, 4)
    assert metadata["scheduled_blocks"] == 28_736
    assert metadata["wrapped_blocks"] == 0
    assert schedule_hash == "d61d355668223d092d2d0f1b04daf9c614c45d6bffe2670ab4a6c63b1ae47523"
    assert pipeline.warmup_steps(449, 0.01) == 5


def test_launch_manifest_code_inventory_covers_every_run_module():
    pipeline = _pipeline()
    identity = pipeline.run_code_identity()
    assert [row["path"] for row in identity["files"]] == [
        "01_calibrate.py",
        "02_train.py",
        "lr_calibration.py",
        "lr_run_config.py",
        "lr_training.py",
        "pipeline.py",
    ]
    assert len(identity["content_sha256"]) == 64
