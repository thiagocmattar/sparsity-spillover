"""Resume pending Run 007 conditions without changing locked scientific code."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import gc
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Mapping


RETRY_DIR = Path(__file__).resolve().parent
RUN_DIR = RETRY_DIR.parents[1]
REPO_ROOT = RUN_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(RUN_DIR))

from sparsity_research.artifacts import (  # noqa: E402
    attempt_lifecycle as immutable_attempt_lifecycle,
    config_sha256,
)
from sparsity_research.data import file_sha256  # noqa: E402
import training  # noqa: E402
import verification  # noqa: E402


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _attempt_rows() -> list[tuple[Path, dict[str, Any]]]:
    attempts_root = RUN_DIR / "artifacts" / "attempts"
    rows = []
    for attempt_dir in sorted(path for path in attempts_root.iterdir() if path.is_dir()):
        rows.append((attempt_dir, _json(attempt_dir / "manifest.json")))
    return rows


def _completed_for_condition(condition_id: str) -> list[Path]:
    return [
        attempt_dir
        for attempt_dir, manifest in _attempt_rows()
        if manifest.get("status") == "completed"
        and manifest.get("condition", {}).get("id") == condition_id
    ]


def _validate_locked_state() -> tuple[
    dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any]
]:
    config = training.load_config()
    conditions = training.condition_specs(config)
    provenance = _json(RUN_DIR / "artifacts" / "launch-provenance.json")
    code_identity = training.run_code_identity()
    if provenance.get("config_sha256") != config_sha256(config):
        raise RuntimeError("Base config differs from launch provenance.")
    if provenance.get("run_code_content_sha256") != code_identity["content_sha256"]:
        raise RuntimeError("Locked run code differs from launch provenance.")
    calibration = provenance.get("calibration")
    if not isinstance(calibration, Mapping):
        raise RuntimeError("Calibration identity is missing from launch provenance.")
    if file_sha256(RUN_DIR / str(calibration.get("path"))) != calibration.get("sha256"):
        raise RuntimeError("Calibration artifact differs from launch provenance.")
    for _, manifest in _attempt_rows():
        if manifest.get("status") == "running":
            raise RuntimeError("A Run 007 attempt is already running.")
    for condition in conditions:
        completed = _completed_for_condition(str(condition["id"]))
        if len(completed) > 1:
            raise RuntimeError(f"Multiple completed attempts exist for {condition['id']}.")
    return config, conditions, provenance, code_identity


@contextmanager
def _next_sequence_attempt(
    run_dir: str | Path,
    *,
    config: Mapping[str, Any],
    command: str,
    mode: str,
    extra_identity: Mapping[str, Any] | None = None,
    attempt_sequence: int | None = None,
) -> Iterator[Any]:
    del attempt_sequence
    with immutable_attempt_lifecycle(
        run_dir,
        config=config,
        command=command,
        mode=mode,
        extra_identity=extra_identity,
        attempt_sequence=None,
    ) as attempt:
        yield attempt


def _completed_attempt_for_order(root: Path, order: int) -> Path:
    conditions = {int(row["order"]): row for row in training.condition_specs(training.load_config())}
    condition = conditions.get(int(order))
    if condition is None:
        raise RuntimeError(f"Unknown condition order {order}.")
    matches = _completed_for_condition(str(condition["id"]))
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected one completed attempt for {condition['id']}, found {len(matches)}."
        )
    return matches[0]


def check() -> dict[str, Any]:
    config, conditions, provenance, code_identity = _validate_locked_state()
    completed = [
        condition["id"]
        for condition in conditions
        if _completed_for_condition(str(condition["id"]))
    ]
    pending = [condition["id"] for condition in conditions if condition["id"] not in completed]
    return {
        "status": "ready" if pending else "nothing_pending",
        "completed_conditions": completed,
        "pending_conditions": pending,
        "config_sha256": config_sha256(config),
        "run_code_content_sha256": code_identity["content_sha256"],
        "launch_run_code_content_sha256": provenance["run_code_content_sha256"],
    }


def run() -> Path:
    import numpy as np
    import torch

    config, conditions, provenance, code_identity = _validate_locked_state()
    training.require_cuda(torch)
    pending = [
        condition
        for condition in conditions
        if not _completed_for_condition(str(condition["id"]))
    ]
    if not pending:
        raise RuntimeError("Run 007 has no pending conditions.")

    train_tokens, validation_tokens, train_meta, validation_meta, cache_seconds = (
        training.load_verified_caches(config, np=np)
    )
    starts, schedule_hash, schedule_metadata = training.build_schedule(config, train_meta, np=np)
    if provenance.get("training_schedule_sha256") != schedule_hash:
        raise RuntimeError("Training schedule differs from launch provenance.")

    original_lifecycle = training.attempt_lifecycle
    original_selector = verification._one_attempt
    training.attempt_lifecycle = _next_sequence_attempt
    verification._one_attempt = _completed_attempt_for_order
    retry_started = datetime.now(timezone.utc).isoformat()
    training.write_json(
        RETRY_DIR / "driver.json",
        {
            "status": "running",
            "started_at": retry_started,
            "pending_conditions": [row["id"] for row in pending],
        },
    )
    try:
        for condition in pending:
            completed_count = len(conditions) - len(pending) + pending.index(condition)
            training.write_json(
                RUN_DIR / "artifacts" / "progress.json",
                {
                    "status": "infrastructure_retry_running",
                    "condition_count": len(conditions),
                    "completed_conditions": completed_count,
                    "completed_optimizer_steps": completed_count
                    * int(training.mapping(config, "training")["max_steps"]),
                    "total_optimizer_steps": len(conditions)
                    * int(training.mapping(config, "training")["max_steps"]),
                    "current_condition": condition["id"],
                    "infrastructure_retry": "001",
                },
            )
            training.run_condition(
                config=config,
                condition=condition,
                train_tokens=train_tokens,
                validation_tokens=validation_tokens,
                train_metadata=train_meta,
                validation_metadata=validation_meta,
                starts=starts,
                schedule_hash=schedule_hash,
                schedule_metadata=schedule_metadata,
                cache_verification_seconds=cache_seconds,
                code_identity=code_identity,
                launch_provenance=provenance,
                torch=torch,
                np=np,
            )
            cache_seconds = 0.0
            gc.collect()
            torch.cuda.empty_cache()

        verification_path = verification.verify_cohort()
        training.write_json(
            RETRY_DIR / "driver.json",
            {
                "status": "completed",
                "started_at": retry_started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "verification": verification_path.relative_to(RUN_DIR).as_posix(),
            },
        )
        return verification_path
    except BaseException as error:
        training.write_json(
            RETRY_DIR / "driver.json",
            {
                "status": "failed",
                "started_at": retry_started,
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "failure": {"type": type(error).__qualname__, "message": str(error)},
            },
        )
        raise
    finally:
        training.attempt_lifecycle = original_lifecycle
        verification._one_attempt = original_selector


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check(), indent=2, sort_keys=True))
    else:
        print(run(), flush=True)
