import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs" / "002-2026-08-29-l1n-spillover-local"
sys.path.insert(0, str(RUN_DIR))

import calibration  # noqa: E402
import plotting  # noqa: E402
import run_config  # noqa: E402


def test_approved_ten_condition_matrix_and_order_are_locked():
    config = run_config.load_config()
    conditions = run_config.condition_specs(config)
    assert [row["id"] for row in conditions] == [
        "gelu-control",
        "relu-control",
        "gelu-l1n-0p1",
        "relu-l1n-0p1",
        "gelu-l1n-0p5",
        "relu-l1n-0p5",
        "gelu-l1n-1",
        "relu-l1n-1",
        "gelu-l1n-5",
        "relu-l1n-5",
    ]
    assert sum(row["is_control"] for row in conditions) == 2
    assert [row["pressure_weight"] for row in conditions[2::2]] == [0.1, 0.5, 1.0, 5.0]


def test_resolved_conditions_separate_gate_from_h_only_pressure():
    config = run_config.load_config()
    conditions = run_config.condition_specs(config)
    gelu_control = run_config.resolved_condition_config(config, conditions[0])
    relu_control = run_config.resolved_condition_config(config, conditions[1])
    gelu_pressured = run_config.resolved_condition_config(config, conditions[2])
    relu_pressured = run_config.resolved_condition_config(config, conditions[3])
    assert gelu_control["model"]["topology_id"] == "A0"
    assert gelu_control["model"]["site_gate"] is None
    assert relu_control["model"] == {
        **config["model"],
        "topology_id": "A1-H",
        "site_gate": {"operator": "relu"},
    }
    assert gelu_pressured["activation_pressure"]["sites"] == ["h"]
    assert gelu_pressured["activation_pressure"]["method"] == "l1_naive"
    assert relu_pressured["activation_pressure"]["sites"] == ["h"]
    assert relu_pressured["activation_pressure"]["weight"] == 0.1


def test_batch_validation_and_figure_estimand_are_locked():
    config = run_config.load_config()
    assert config["training"]["peak_learning_rate"] == 0.004
    assert config["training"]["global_batch_size"] == 64
    assert config["training"]["micro_batch_size"] == 4
    assert config["training"]["gradient_accumulation_steps"] == 16
    assert config["validation"]["documents"] == 500
    assert config["validation"]["complete_sequences"] == 338
    assert config["validation"]["input_tokens"] == 692224
    assert config["validation"]["excluded_tail_tokens"] == 1444
    assert config["figure"]["near_zero_threshold"] == 0.001
    assert config["figure"]["x_site"] == "h"
    assert config["figure"]["y_sites"] == ["q_post", "k_post", "v"]


def test_common_step_budget_accounts_for_all_four_runtime_classes():
    sample = {
        "model_setup_seconds": 1.0,
        "ordinary_full_validation_seconds": 2.0,
        "diagnostic_full_validation_seconds": 3.0,
        "weight_statistics_seconds": 1.0,
        "checkpoint_save_hash_reload_seconds": 3.0,
        "optimizer_step_seconds": [1.0, 1.1, 1.2, 1.3],
    }
    measured = {
        "cache_verification_seconds": 5.0,
        "terminal_headroom_seconds": 25.0,
        "condition_counts": {
            "gelu_control": 1,
            "relu_control": 1,
            "gelu_pressure": 4,
            "relu_pressure": 4,
        },
        "samples": {
            "gelu_control": sample,
            "relu_control": sample,
            "gelu_pressure": sample,
            "relu_pressure": sample,
        },
    }
    steps = calibration.budget_steps_from_calibration(measured, planning_seconds=1000)
    estimate = calibration.estimate_cohort(measured, common_steps=steps)
    assert steps == 66
    assert estimate["total_optimizer_steps"] == 660
    assert estimate["p90_seconds"] <= 1000


def test_figure_reduction_keeps_two_five_point_lines_and_averages_sites():
    conditions = run_config.condition_specs(run_config.load_config())
    rows = []
    for index, condition in enumerate(conditions):
        rows.append(
            {
                "condition": condition,
                "completed_steps": 10,
                "h_near_zero_fraction_epsilon_0p001": 0.10 + index / 100,
                "attention_mean_near_zero_fraction_epsilon_0p001": 0.02 + index / 1000,
                "attention_site_near_zero_fractions_epsilon_0p001": {
                    "q_post": 0.01,
                    "k_post": 0.02,
                    "v": 0.03,
                },
                "final_validation_loss": 6.0,
            }
        )
    grouped = plotting.reduce_points({"status": "verified", "conditions": rows})
    assert list(grouped) == ["gelu", "relu"]
    assert len(grouped["gelu"]) == len(grouped["relu"]) == 5
    assert grouped["gelu"][0]["label"] == "control"
    assert grouped["relu"][-1]["label"] == "lambda=5"
    assert grouped["gelu"][0]["attention_mean_near_zero_percent"] == 2.0


def test_code_inventory_and_monitor_cover_terminal_workflow():
    identity = run_config.run_code_identity()
    assert [row["path"] for row in identity["files"]] == [
        "02_train.py",
        "03_verify.py",
        "04_plot.py",
        "run_config.py",
        "training.py",
        "verification.py",
        "plotting.py",
    ]
    monitor = (RUN_DIR / "05_monitor.ps1").read_text(encoding="utf-8")
    assert "Start-Sleep -Seconds $IntervalSeconds" in monitor
    assert "$milestone = 20" in monitor
    assert "$milestone += 20" in monitor
