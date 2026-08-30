"""Terminal verification and count-first figure reduction for Run 002."""

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
    RUN_DIR,
    condition_specs,
    inventory_content_sha256,
    load_config,
    resolved_condition_config,
    run_code_identity,
    write_json,
)


def verify_cohort(config_path: str | Path = DEFAULT_CONFIG) -> Path:
    config = load_config(config_path)
    expected_conditions = condition_specs(config)
    expected_steps = int(config["training"]["max_steps"])
    attempts_root = RUN_DIR / "artifacts" / "attempts"
    attempts = sorted(path for path in attempts_root.iterdir() if path.is_dir())
    _require(
        len(attempts) == len(expected_conditions),
        f"Expected ten attempts, found {len(attempts)}.",
    )
    expected_code = run_code_identity()["content_sha256"]
    initial_hashes = set()
    schedule_hashes = set()
    code_hashes = set()
    git_commits = set()
    git_dirty_states = set()
    started = []
    finished = []
    rows = []

    for attempt_dir, condition in zip(attempts, expected_conditions, strict=True):
        manifest = _read_json(attempt_dir / "manifest.json")
        metrics = _read_json(attempt_dir / "metrics.json")
        resolved = _read_yaml(attempt_dir / "config.yaml")
        events = _read_jsonl(attempt_dir / "events.jsonl")
        expected_resolved = resolved_condition_config(config, condition)

        label = attempt_dir.name
        _require(manifest["status"] == "completed", f"{label}: status")
        _require(manifest["condition"] == condition, f"{label}: condition identity")
        _require(resolved["condition"] == condition, f"{label}: config condition")
        _require(resolved["model"] == expected_resolved["model"], f"{label}: model")
        _require(
            resolved["activation_pressure"] == expected_resolved["activation_pressure"],
            f"{label}: pressure",
        )
        _require(
            resolved["training"]["max_steps"] == expected_steps,
            f"{label}: max steps",
        )
        _require(
            float(resolved["training"]["peak_learning_rate"]) == 0.004,
            f"{label}: peak LR",
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
        _require(len(validation_events) == 2, f"{label}: validation event count")
        _require(
            [event["step"] for event in validation_events] == [1, expected_steps],
            f"{label}: validation step sequence",
        )
        for event in train_events:
            for key in (
                "task_loss",
                "learning_rate",
                "step_wall_seconds",
                "tokens_per_second",
                "adamw_gradient_norm_pre_clip",
                "adamw_gradient_norm_post_clip",
            ):
                _finite(event[key], f"{label}: {key}")
            if condition["is_control"]:
                _require("pressure_loss" not in event, f"{label}: control pressure metric")
            else:
                for key in (
                    "pressure_loss",
                    "weighted_pressure_loss",
                    "augmented_loss",
                    "task_gradient_norm",
                    "pressure_gradient_norm",
                    "pressure_to_task_gradient_norm_ratio",
                    "task_pressure_gradient_dot",
                    "task_pressure_gradient_cosine",
                ):
                    _finite(event[key], f"{label}: {key}")
                _require(
                    float(event["pressure_weight"]) == condition["pressure_weight"],
                    f"{label}: event pressure weight",
                )
                _require(
                    isinstance(event["gradient_conflict"], bool),
                    f"{label}: conflict flag",
                )
        for event in validation_events:
            _finite(event["loss"], f"{label}: validation loss")
            _require_validation(event, label)
        _require_validation(metrics["validation"]["step_one"], label)
        _require_validation(metrics["validation"]["final"], label)

        activation = _read_json(
            attempt_dir / "diagnostics" / "activation_statistics.json"
        )
        weights = _read_json(attempt_dir / "diagnostics" / "weight_statistics.json")
        _require(len(activation["rows"]) == 24, f"{label}: activation rows")
        _require(len(activation["pooled_by_site"]) == 4, f"{label}: pooled sites")
        expected_names = {
            f"{site}.layer_{layer}"
            for site in ("h", "q_post", "k_post", "v")
            for layer in range(6)
        }
        _require(
            {row["name"] for row in activation["rows"]} == expected_names,
            f"{label}: activation names",
        )
        for row in activation["rows"] + activation["pooled_by_site"]:
            _require(row["finite"] == row["total"], f"{label}: {row['name']} finite")
            _require(
                set(row["threshold_hits"]) == {"0", "0.001", "0.01"},
                f"{label}: {row['name']} thresholds",
            )
            _require(
                all(0 <= int(value) <= int(row["total"]) for value in row["threshold_hits"].values()),
                f"{label}: {row['name']} threshold counts",
            )
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
            checkpoint_hash == metrics["checkpoint"]["content_sha256"],
            f"{label}: checkpoint metrics hash",
        )
        _require(
            checkpoint_hash == manifest["checkpoint"]["content_sha256"],
            f"{label}: checkpoint manifest hash",
        )

        pooled = {row["name"]: row for row in activation["pooled_by_site"]}
        threshold = "0.001"
        h_fraction = int(pooled["h"]["threshold_hits"][threshold]) / int(
            pooled["h"]["total"]
        )
        attention_fractions = {
            site: int(pooled[site]["threshold_hits"][threshold])
            / int(pooled[site]["total"])
            for site in ("q_post", "k_post", "v")
        }
        attention_mean = sum(attention_fractions.values()) / len(attention_fractions)
        conflict_rate = None
        if not condition["is_control"]:
            conflict_rate = sum(bool(event["gradient_conflict"]) for event in train_events) / len(
                train_events
            )

        initial_hashes.add(manifest["initial_parameter_sha256"])
        schedule_hashes.add(manifest["training_schedule_hash"])
        code_hashes.add(manifest["run_code"]["content_sha256"])
        git_commits.add(manifest["code"]["git_commit"])
        git_dirty_states.add(manifest["code"]["git_dirty"])
        started.append(datetime.fromisoformat(manifest["started_at"]))
        finished.append(datetime.fromisoformat(manifest["finished_at"]))
        rows.append(
            {
                "attempt_id": attempt_dir.name,
                "condition": condition,
                "completed_steps": metrics["training"]["completed_steps"],
                "input_tokens": metrics["training"]["input_tokens"],
                "final_train_loss": metrics["training"]["task_loss_final"],
                "step_one_validation_loss": metrics["validation"]["step_one"]["loss"],
                "final_validation_loss": metrics["validation"]["final"]["loss"],
                "median_step_seconds": metrics["training"]["median_step_seconds"],
                "median_tokens_per_second": metrics["training"]["median_tokens_per_second"],
                "total_seconds": metrics["timing"]["total_seconds"],
                "h_near_zero_fraction_epsilon_0p001": h_fraction,
                "attention_site_near_zero_fractions_epsilon_0p001": attention_fractions,
                "attention_mean_near_zero_fraction_epsilon_0p001": attention_mean,
                "gradient_conflict_rate": conflict_rate,
                "checkpoint_bytes": metrics["checkpoint"]["bytes"],
                "checkpoint_content_sha256": checkpoint_hash,
            }
        )

    _require(len(initial_hashes) == 1, "Initial parameter hashes differ.")
    _require(len(schedule_hashes) == 1, "Training schedule hashes differ.")
    _require(code_hashes == {expected_code}, "Executed run-code identities differ.")
    _require(None not in git_commits and len(git_commits) == 1, "Git commit is missing or mixed.")
    _require(None not in git_dirty_states and len(git_dirty_states) == 1, "Git dirty state is missing or mixed.")
    verifier_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verifier": {"path": Path(__file__).name, "sha256": verifier_sha},
        "status": "verified",
        "evidence_label": "valid",
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
        "cohort_wall_seconds": (max(finished) - min(started)).total_seconds(),
        "checkpoint_total_bytes": sum(row["checkpoint_bytes"] for row in rows),
        "figure_estimand": {
            "epsilon": 0.001,
            "x": "count-pooled h near-zero fraction",
            "y": "unweighted mean of separately count-pooled q_post, k_post, and v near-zero fractions",
        },
        "conditions": rows,
    }
    output = RUN_DIR / "artifacts" / "verification.json"
    write_json(output, summary)
    write_json(
        RUN_DIR / "artifacts" / "progress.json",
        {
            "status": "verified_pending_figure",
            "condition_count": len(rows),
            "completed_conditions": len(rows),
            "elapsed_seconds": summary["cohort_wall_seconds"],
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
