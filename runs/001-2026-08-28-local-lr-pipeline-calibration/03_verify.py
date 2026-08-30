"""Read-only terminal verification for the completed four-condition cohort."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

from sparsity_research.artifacts import verify_transfer_inventory  # noqa: E402
from lr_run_config import inventory_content_sha256, write_json  # noqa: E402


EXPECTED = [
    ("lr-5e-4", 0.0005),
    ("lr-1e-3", 0.001),
    ("lr-2e-3", 0.002),
    ("lr-4e-3", 0.004),
]
EXPECTED_STEPS = 449
EXPECTED_SCHEDULE = "d61d355668223d092d2d0f1b04daf9c614c45d6bffe2670ab4a6c63b1ae47523"
EXPECTED_CODE = "29160e3cef6af808743025c3e3ec2e50906bd0bd5c6755e86fdfaeb2e1002765"


def main() -> None:
    attempts_root = RUN_DIR / "artifacts" / "attempts"
    attempts = sorted(path for path in attempts_root.iterdir() if path.is_dir())
    if len(attempts) != len(EXPECTED):
        raise RuntimeError(f"Expected four attempts, found {len(attempts)}.")

    rows = []
    initial_hashes = set()
    schedule_hashes = set()
    code_hashes = set()
    started = []
    finished = []
    for attempt_dir, (condition_id, peak_lr) in zip(attempts, EXPECTED, strict=True):
        manifest = _read_json(attempt_dir / "manifest.json")
        metrics = _read_json(attempt_dir / "metrics.json")
        config = _read_yaml_as_json(attempt_dir / "config.yaml")
        events = _read_jsonl(attempt_dir / "events.jsonl")

        _require(manifest["status"] == "completed", f"{attempt_dir.name}: not completed")
        _require(manifest["condition"]["id"] == condition_id, f"{attempt_dir.name}: condition")
        _require(
            float(manifest["condition"]["peak_learning_rate"]) == peak_lr,
            f"{attempt_dir.name}: learning rate",
        )
        _require(config["training"]["max_steps"] == EXPECTED_STEPS, f"{attempt_dir.name}: steps")
        _require(
            config["training"]["peak_learning_rate"] == peak_lr,
            f"{attempt_dir.name}: resolved peak LR",
        )

        train_events = [event for event in events if event.get("event") == "train"]
        validation_events = [event for event in events if event.get("event") == "validation"]
        _require(len(train_events) == EXPECTED_STEPS, f"{attempt_dir.name}: train event count")
        _require(
            [event["step"] for event in train_events] == list(range(1, EXPECTED_STEPS + 1)),
            f"{attempt_dir.name}: step sequence",
        )
        _require(len(validation_events) == 2, f"{attempt_dir.name}: validation event count")
        _require(
            [event["step"] for event in validation_events] == [1, EXPECTED_STEPS],
            f"{attempt_dir.name}: validation steps",
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
                _require(math.isfinite(float(event[key])), f"{attempt_dir.name}: nonfinite {key}")
        for event in validation_events:
            _require(math.isfinite(float(event["loss"])), f"{attempt_dir.name}: validation loss")
            _require_validation(event, attempt_dir.name)
        _require_validation(metrics["validation"]["step_one"], attempt_dir.name)
        _require_validation(metrics["validation"]["final"], attempt_dir.name)

        activation = _read_json(attempt_dir / "diagnostics" / "activation_statistics.json")
        weights = _read_json(attempt_dir / "diagnostics" / "weight_statistics.json")
        _require(len(activation["rows"]) == 18, f"{attempt_dir.name}: activation rows")
        _require(len(activation["pooled_by_site"]) == 3, f"{attempt_dir.name}: pooled rows")
        _require(len(weights["rows"]) == 76, f"{attempt_dir.name}: weight rows")
        _require(weights["pooled"]["finite"] == weights["pooled"]["elements"], f"{attempt_dir.name}: weights")
        for row in activation["rows"]:
            _require(row["finite"] == row["total"], f"{attempt_dir.name}: {row['name']} finite")
            _require(set(row["threshold_hits"]) == {"0", "0.001", "0.01"}, f"{attempt_dir.name}: thresholds")

        inventory_path = attempt_dir / "diagnostics" / "checkpoint_inventory.json"
        inventory = _read_json(inventory_path)
        checkpoint_dir = attempt_dir / "checkpoints" / "final"
        verify_transfer_inventory(checkpoint_dir, inventory)
        content_hash = inventory_content_sha256(inventory)
        _require(
            content_hash == metrics["checkpoint"]["content_sha256"],
            f"{attempt_dir.name}: checkpoint content hash",
        )
        _require(
            content_hash == manifest["checkpoint"]["content_sha256"],
            f"{attempt_dir.name}: manifest checkpoint hash",
        )

        initial_hashes.add(manifest["initial_parameter_sha256"])
        schedule_hashes.add(manifest["training_schedule_hash"])
        code_hashes.add(manifest["run_code"]["content_sha256"])
        started.append(datetime.fromisoformat(manifest["started_at"]))
        finished.append(datetime.fromisoformat(manifest["finished_at"]))
        rows.append(
            {
                "attempt_id": attempt_dir.name,
                "condition_id": condition_id,
                "peak_learning_rate": peak_lr,
                "final_train_loss": metrics["training"]["task_loss_final"],
                "final_validation_loss": metrics["validation"]["final"]["loss"],
                "step_one_validation_loss": metrics["validation"]["step_one"]["loss"],
                "completed_steps": metrics["training"]["completed_steps"],
                "input_tokens": metrics["training"]["input_tokens"],
                "median_step_seconds": metrics["training"]["median_step_seconds"],
                "median_tokens_per_second": metrics["training"]["median_tokens_per_second"],
                "total_seconds": metrics["timing"]["total_seconds"],
                "checkpoint_bytes": metrics["checkpoint"]["bytes"],
                "checkpoint_content_sha256": content_hash,
                "git_commit": manifest["code"]["git_commit"],
                "git_dirty": manifest["code"]["git_dirty"],
            }
        )

    _require(len(initial_hashes) == 1, "Initial parameter hashes differ.")
    _require(schedule_hashes == {EXPECTED_SCHEDULE}, "Training schedule hashes differ.")
    _require(code_hashes == {EXPECTED_CODE}, "Run code hashes differ.")
    verifier_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    summary = {
        "schema_version": 1,
        "verified_at": datetime.now(timezone.utc).isoformat(),
        "verifier": {"path": Path(__file__).name, "sha256": verifier_sha},
        "status": "verified",
        "evidence_label": "provisional",
        "provisional_reason": (
            "Detached attempt manifests have null Git commit/dirty fields; exact resolved "
            "configs and identical run-code content hashes are present."
        ),
        "attempt_count": len(rows),
        "condition_count": len(rows),
        "total_optimizer_steps": sum(row["completed_steps"] for row in rows),
        "total_training_input_tokens": sum(row["input_tokens"] for row in rows),
        "complete_validation_passes": 2 * len(rows),
        "initial_parameter_sha256": next(iter(initial_hashes)),
        "training_schedule_sha256": next(iter(schedule_hashes)),
        "run_code_content_sha256": next(iter(code_hashes)),
        "cohort_wall_seconds": (max(finished) - min(started)).total_seconds(),
        "checkpoint_total_bytes": sum(row["checkpoint_bytes"] for row in rows),
        "conditions": rows,
    }
    output = RUN_DIR / "artifacts" / "verification.json"
    write_json(output, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def _require_validation(value: dict, label: str) -> None:
    expected = {
        "sequences": 338,
        "input_tokens": 692224,
        "excluded_tail_tokens": 1444,
        "complete_block_coverage": True,
    }
    _require(all(value.get(key) == expected_value for key, expected_value in expected.items()), f"{label}: validation coverage")


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _read_yaml_as_json(path: Path):
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


if __name__ == "__main__":
    main()
