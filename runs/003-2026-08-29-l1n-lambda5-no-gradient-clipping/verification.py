"""Terminal verification and clipped-baseline comparison for Run 003."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from sparsity_research.artifacts import verify_transfer_inventory

from run_config import (
    DEFAULT_CONFIG,
    REPO_ROOT,
    RUN_DIR,
    baseline_identity,
    condition_specs,
    inventory_content_sha256,
    load_config,
    mapping,
    run_code_identity,
    write_json,
)


def verify_cohort(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    config = load_config(config_path)
    expected_conditions = condition_specs(config)
    attempts_root = RUN_DIR / "artifacts" / "attempts"
    attempts = sorted(path for path in attempts_root.iterdir() if path.is_dir())
    _require(len(attempts) == 2, "Run 003 requires exactly two attempts.")
    expected_code = run_code_identity()["content_sha256"]
    baseline_record = baseline_identity(config)
    baseline = _read_json(REPO_ROOT / baseline_record["path"])
    _require(baseline.get("status") == "verified", "Run 002 baseline is not verified.")
    baseline_by_id = {row["attempt_id"]: row for row in baseline["conditions"]}

    rows = []
    initial_hashes: set[str] = set()
    schedule_hashes: set[str] = set()
    code_hashes: set[str] = set()
    git_commits: set[str | None] = set()
    git_dirty_states: set[bool | None] = set()
    checkpoint_bytes = 0

    for attempt_dir, expected_condition in zip(attempts, expected_conditions, strict=True):
        label = attempt_dir.name
        manifest = _read_json(attempt_dir / "manifest.json")
        metrics = _read_json(attempt_dir / "metrics.json")
        snapshot = _read_yaml(attempt_dir / "config.yaml")
        events = _read_jsonl(attempt_dir / "events.jsonl")
        condition = manifest.get("condition")
        _require(manifest.get("status") == "completed", f"{label}: manifest status")
        _require(condition == expected_condition, f"{label}: condition order/identity")
        _require(snapshot.get("condition") == expected_condition, f"{label}: config condition")
        _require(
            manifest.get("gradient_clipping")
            == {"enabled": False, "max_norm": None, "finite_gradient_check": True},
            f"{label}: unclipped manifest contract",
        )
        _require(
            manifest.get("comparison_baseline") == baseline_record,
            f"{label}: baseline identity",
        )

        training = metrics["training"]
        expected_steps = int(mapping(config, "training")["max_steps"])
        _require(training["completed_steps"] == expected_steps, f"{label}: steps")
        _require(
            training["gradient_clipping"]
            == {"enabled": False, "max_norm": None, "finite_gradient_check": True},
            f"{label}: unclipped metrics contract",
        )
        train_events = [event for event in events if event.get("event") == "train"]
        validation_events = [
            event for event in events if event.get("event") == "validation"
        ]
        _require(len(train_events) == expected_steps, f"{label}: train event count")
        _require(
            [event["step"] for event in train_events]
            == list(range(1, expected_steps + 1)),
            f"{label}: train step sequence",
        )
        _require(
            [event["step"] for event in validation_events] == [1, expected_steps],
            f"{label}: validation events",
        )
        for event in train_events:
            for key in (
                "task_loss",
                "pressure_loss",
                "weighted_pressure_loss",
                "augmented_loss",
                "task_gradient_norm",
                "pressure_gradient_norm",
                "task_pressure_gradient_dot",
                "task_pressure_gradient_cosine",
                "adamw_gradient_norm_pre_clip",
                "adamw_gradient_norm_post_clip",
                "learning_rate",
                "step_wall_seconds",
                "tokens_per_second",
            ):
                _finite(event[key], f"{label}: {key}")
            _require(
                event["adamw_gradient_clipping_enabled"] is False
                and event["adamw_gradient_clip_norm"] is None
                and event["adamw_gradient_was_clipped"] is False,
                f"{label}: clipping event fields",
            )
            _require(
                float(event["adamw_gradient_norm_pre_clip"])
                == float(event["adamw_gradient_norm_post_clip"]),
                f"{label}: gradient norm changed",
            )
        for event in validation_events:
            _finite(event["loss"], f"{label}: validation loss")
            _require_validation(event, label)

        statistics = _read_json(
            attempt_dir / "diagnostics" / "activation_statistics.json"
        )
        expected_sites = tuple(mapping(config, "diagnostics")["activation_sites"])
        _require(len(statistics["rows"]) == 36, f"{label}: activation rows")
        _require(len(statistics["pooled_by_site"]) == 6, f"{label}: pooled sites")
        expected_names = {
            f"{site}.layer_{layer}" for site in expected_sites for layer in range(6)
        }
        _require(
            {row["name"] for row in statistics["rows"]} == expected_names,
            f"{label}: activation names",
        )
        for row in statistics["rows"] + statistics["pooled_by_site"]:
            _require(row["finite"] == row["total"], f"{label}: finite {row['name']}")
            _require(
                set(row["threshold_hits"]) == {"0", "0.001", "0.01"},
                f"{label}: thresholds {row['name']}",
            )
            _require(
                all(
                    0 <= int(value) <= int(row["total"])
                    for value in row["threshold_hits"].values()
                ),
                f"{label}: threshold counts {row['name']}",
            )
        boundary = statistics["attention_output_boundary_equivalence"]
        _require(
            boundary["exactly_equal"] is True
            and boundary["comparisons"] == 510
            and float(boundary["maximum_absolute_difference"]) == 0.0,
            f"{label}: attention-output boundary",
        )

        weights = _read_json(attempt_dir / "diagnostics" / "weight_statistics.json")
        _require(len(weights["rows"]) == 76, f"{label}: weight rows")
        _require(
            weights["pooled"]["finite"] == weights["pooled"]["elements"],
            f"{label}: weight finiteness",
        )
        inventory = _read_json(
            attempt_dir / "diagnostics" / "checkpoint_inventory.json"
        )
        checkpoint_dir = attempt_dir / "checkpoints" / "final"
        verify_transfer_inventory(checkpoint_dir, inventory)
        checkpoint_hash = inventory_content_sha256(inventory)
        _require(
            checkpoint_hash == metrics["checkpoint"]["content_sha256"]
            == manifest["checkpoint"]["content_sha256"],
            f"{label}: checkpoint hash",
        )

        baseline_id = baseline_record[f"{condition['activation']}_attempt_id"]
        source = baseline_by_id[baseline_id]
        pooled = {row["name"]: row for row in statistics["pooled_by_site"]}
        near_zero = {
            threshold: {
                site: int(pooled[site]["threshold_hits"][threshold])
                / int(pooled[site]["total"])
                for site in expected_sites
            }
            for threshold in ("0", "0.001", "0.01")
        }
        attention_mean = sum(
            near_zero["0.001"][site] for site in ("q_post", "k_post", "v")
        ) / 3.0
        conflict_rate = sum(
            bool(event["gradient_conflict"]) for event in train_events
        ) / len(train_events)
        gradient_norms = sorted(
            float(event["adamw_gradient_norm_pre_clip"]) for event in train_events
        )
        row = {
            "attempt_id": label,
            "condition": condition,
            "completed_steps": expected_steps,
            "input_tokens": int(training["input_tokens"]),
            "final_train_loss": float(training["task_loss_final"]),
            "step_one_validation_loss": float(metrics["validation"]["step_one"]["loss"]),
            "final_validation_loss": float(metrics["validation"]["final"]["loss"]),
            "median_step_seconds": float(training["median_step_seconds"]),
            "median_tokens_per_second": float(training["median_tokens_per_second"]),
            "total_seconds": float(metrics["timing"]["total_seconds"]),
            "gradient_conflict_rate": conflict_rate,
            "combined_gradient_norm": {
                "median": gradient_norms[(len(gradient_norms) - 1) // 2],
                "maximum": max(gradient_norms),
                "steps_above_run002_clip_norm": sum(
                    value > baseline_record["gradient_clip_norm"]
                    for value in gradient_norms
                ),
            },
            "near_zero_fractions": near_zero,
            "attention_mean_near_zero_fraction_epsilon_0p001": attention_mean,
            "baseline": {
                "run": "002",
                "attempt_id": baseline_id,
                "gradient_clip_norm": baseline_record["gradient_clip_norm"],
                "final_validation_loss": float(source["final_validation_loss"]),
                "h_near_zero_fraction_epsilon_0p001": float(
                    source["h_near_zero_fraction_epsilon_0p001"]
                ),
                "attention_mean_near_zero_fraction_epsilon_0p001": float(
                    source["attention_mean_near_zero_fraction_epsilon_0p001"]
                ),
            },
            "delta_vs_clipped_baseline": {
                "final_validation_loss": float(metrics["validation"]["final"]["loss"])
                - float(source["final_validation_loss"]),
                "h_near_zero_fraction_epsilon_0p001": near_zero["0.001"]["h"]
                - float(source["h_near_zero_fraction_epsilon_0p001"]),
                "attention_mean_near_zero_fraction_epsilon_0p001": attention_mean
                - float(source["attention_mean_near_zero_fraction_epsilon_0p001"]),
            },
            "checkpoint_bytes": int(metrics["checkpoint"]["bytes"]),
            "checkpoint_content_sha256": checkpoint_hash,
        }
        rows.append(row)
        checkpoint_bytes += row["checkpoint_bytes"]
        initial_hashes.add(manifest["initial_parameter_sha256"])
        schedule_hashes.add(manifest["training_schedule_hash"])
        code_hashes.add(manifest["run_code"]["content_sha256"])
        git_commits.add(manifest["code"]["git_commit"])
        git_dirty_states.add(manifest["code"]["git_dirty"])

    _require(initial_hashes == {baseline["initial_parameter_sha256"]}, "Initial hash mismatch.")
    _require(schedule_hashes == {baseline["training_schedule_sha256"]}, "Schedule mismatch.")
    _require(code_hashes == {expected_code}, "Executed run-code identities differ.")
    _require(None not in git_commits and len(git_commits) == 1, "Git commit missing/mixed.")
    _require(
        None not in git_dirty_states and len(git_dirty_states) == 1,
        "Git dirty state missing/mixed.",
    )

    verifier_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verifier": {"path": Path(__file__).name, "sha256": verifier_sha},
        "status": "verified",
        "evidence_label": "valid",
        "question": "effect of disabling global gradient clipping at lambda=5",
        "gradient_clipping": {
            "enabled": False,
            "max_norm": None,
            "finite_gradient_check": True,
        },
        "comparison_baseline": baseline_record,
        "attempt_count": len(rows),
        "condition_count": len(rows),
        "total_optimizer_steps": sum(row["completed_steps"] for row in rows),
        "total_training_input_tokens": sum(row["input_tokens"] for row in rows),
        "complete_validation_passes": 2 * len(rows),
        "initial_parameter_sha256": next(iter(initial_hashes)),
        "training_schedule_sha256": next(iter(schedule_hashes)),
        "run_code_content_sha256": next(iter(code_hashes)),
        "git_commit": next(iter(git_commits)),
        "git_dirty": next(iter(git_dirty_states)),
        "checkpoint_total_bytes": checkpoint_bytes,
        "conditions": rows,
    }
    output = RUN_DIR / "artifacts" / "verification.json"
    write_json(output, summary)
    write_json(
        RUN_DIR / "artifacts" / "progress.json",
        {
            "status": "verified",
            "condition_count": 2,
            "completed_conditions": 2,
            "evidence_label": "valid",
        },
    )
    return output


def _require_validation(value: Mapping[str, Any], label: str) -> None:
    expected = {
        "sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "complete_block_coverage": True,
    }
    _require(
        all(value.get(key) == expected_value for key, expected_value in expected.items()),
        f"{label}: validation coverage",
    )


def _finite(value: Any, label: str) -> None:
    _require(math.isfinite(float(value)), f"{label}: nonfinite")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)
