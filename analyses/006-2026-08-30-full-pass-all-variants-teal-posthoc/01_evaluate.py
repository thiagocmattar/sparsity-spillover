"""Evaluate uniform TEAL-style clipping on every trained Analysis 004 endpoint."""

from __future__ import annotations

import argparse
import gc
import hashlib
from datetime import datetime, timezone
from importlib.util import module_from_spec, spec_from_file_location
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Mapping


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_DIR = ANALYSIS_DIR.parents[1]
ANALYSIS_004 = (
    REPO_DIR / "analyses" / "004-2026-08-30-full-pass-quality-logical-frontier"
)
ANALYSIS_005 = REPO_DIR / "analyses" / "005-2026-08-30-run004-controls-teal-posthoc"
COMPARISON_PATH = ANALYSIS_004 / "comparison.json"
CONTROL_RESULT_PATH = ANALYSIS_005 / "teal_frontier.json"
CONTROL_EVALUATOR_PATH = ANALYSIS_005 / "01_evaluate.py"
TRAIN_METADATA_PATH = (
    REPO_DIR
    / "data"
    / "tokenized"
    / "minipile-pythia-14m-full"
    / "train"
    / "metadata.json"
)
VALIDATION_METADATA_PATH = TRAIN_METADATA_PATH.parent.parent / "validation" / "metadata.json"
ARTIFACT_DIR = ANALYSIS_DIR / "artifacts"
CALIBRATION_DIR = ARTIFACT_DIR / "calibration"
PROGRESS_PATH = ARTIFACT_DIR / "progress.jsonl"
RESULT_PATH = ANALYSIS_DIR / "teal_all_variants.json"
SMOKE_PATH = ANALYSIS_DIR / "prelaunch" / "smoke.json"
RUN_STATE_PATH = ARTIFACT_DIR / "run-state.json"

RUN_DIRS = {
    "run004": REPO_DIR / "runs" / "004-2026-08-29-pythia14m-full-pass-l1n",
    "run009": REPO_DIR / "runs" / "009-2026-08-30-pythia14m-full-pass-ol1",
    "run011": REPO_DIR / "runs" / "011-2026-08-30-pythia14m-full-pass-a4z",
}
CONTROL_CONDITION_IDS = ("gelu-control", "relu-control")
NEW_CONDITION_IDS = (
    "relu-l1n-0p05",
    "relu-l1n-0p1",
    "relu-l1n-0p5",
    "relu-l1n-1",
    "relu-ol1-0p05",
    "relu-ol1-0p1",
    "relu-ol1-0p5",
    "relu-ol1-1",
    "a4z-one-sided-kappa-0",
    "a4z-one-sided-kappa-0p01",
    "a4z-one-sided-kappa-0p05",
    "a4z-one-sided-kappa-0p1",
    "a4z-one-sided-kappa-0p5",
)
ALL_CONDITION_IDS = CONTROL_CONDITION_IDS + NEW_CONDITION_IDS
TARGET_SPARSITIES = tuple(index / 10 for index in range(10))
SITES = ("a", "m", "h", "z")
EXPECTED_VALIDATION = {
    "sequences": 338,
    "input_tokens": 692_224,
    "source_tokens": 693_668,
    "excluded_tail_tokens": 1_444,
    "complete_block_coverage": True,
}
SMOKE_VALIDATION_BLOCKS = 32
SMOKE_TARGET = 0.2


def _load_control_evaluator() -> Any:
    spec = spec_from_file_location("analysis005_teal_evaluator", CONTROL_EVALUATOR_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not load the frozen Analysis 005 TEAL evaluator.")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEAL = _load_control_evaluator()


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_content_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in directory.rglob("*") if item.is_file()):
        relative = path.relative_to(directory).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(_sha256(path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source_verification(run_id: str) -> tuple[dict[str, Any], Path]:
    path = RUN_DIRS[run_id] / "artifacts" / "verification.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "verified" or value.get("evidence_label") != "valid":
        raise ValueError(f"{run_id} does not provide valid verified evidence.")
    return value, path


def _comparison() -> dict[str, Any]:
    value = json.loads(COMPARISON_PATH.read_text(encoding="utf-8"))
    if (
        value.get("schema_version") != 1
        or value.get("evidence_status") != "complete_verified_matched_cohorts"
        or len(value.get("conditions", [])) != 15
    ):
        raise ValueError("Analysis 004 is not complete verified 15-checkpoint evidence.")
    ids = tuple(row["condition_id"] for row in value["conditions"])
    if set(ids) != set(ALL_CONDITION_IDS):
        raise ValueError("Analysis 004 checkpoint set differs from the approved design.")
    return value


def load_source_descriptors(*, verify_checkpoint_content: bool = True) -> list[dict[str, Any]]:
    """Load and reconcile the thirteen newly evaluated checkpoint descriptors."""

    comparison = _comparison()
    matched = comparison["matched_identity"]
    verifications = {run_id: _source_verification(run_id) for run_id in RUN_DIRS}
    rows_by_id = {row["condition_id"]: row for row in comparison["conditions"]}
    sources: list[dict[str, Any]] = []
    for condition_id in NEW_CONDITION_IDS:
        row = rows_by_id[condition_id]
        run_id = str(row["run"])
        verification, verification_path = verifications[run_id]
        if verification["initial_parameter_sha256"] != matched["initial_parameter_sha256"]:
            raise ValueError(f"Initial-parameter mismatch for {run_id}.")
        if verification["training_schedule_sha256"] != matched["training_schedule_sha256"]:
            raise ValueError(f"Training-schedule mismatch for {run_id}.")
        verified_by_id = {
            item["condition"]["id"]: item for item in verification["conditions"]
        }
        verified = verified_by_id.get(condition_id)
        if verified is None or verified["attempt_id"] != row["attempt_id"]:
            raise ValueError(f"Verified attempt mismatch for {condition_id}.")

        attempt_dir = RUN_DIRS[run_id] / "artifacts" / "attempts" / row["attempt_id"]
        manifest_path = attempt_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed" or manifest["condition"]["id"] != condition_id:
            raise ValueError(f"Incomplete or mismatched source manifest for {condition_id}.")
        mismatches = [
            key
            for key, expected in EXPECTED_VALIDATION.items()
            if manifest["validation_coverage"].get(key) != expected
        ]
        if mismatches:
            raise ValueError(f"Validation coverage mismatch for {condition_id}: {mismatches}")
        if not math.isclose(
            float(manifest["validation_coverage"]["loss"]),
            float(row["final_validation_loss"]),
            rel_tol=0.0,
            abs_tol=1e-14,
        ):
            raise ValueError(f"Canonical loss mismatch for {condition_id}.")

        checkpoint = attempt_dir / manifest["checkpoints"]["final"]["path"]
        expected_checkpoint_hash = str(verified["checkpoint_content_sha256"])
        if manifest["checkpoints"]["final"]["content_sha256"] != expected_checkpoint_hash:
            raise ValueError(f"Checkpoint identity mismatch for {condition_id}.")
        if verify_checkpoint_content:
            actual_checkpoint_hash = _inventory_content_sha256(checkpoint)
            if actual_checkpoint_hash != expected_checkpoint_hash:
                raise ValueError(f"Checkpoint content changed for {condition_id}.")

        sources.append(
            {
                "condition_id": condition_id,
                "run": run_id,
                "attempt_id": row["attempt_id"],
                "checkpoint": checkpoint,
                "checkpoint_content_sha256": expected_checkpoint_hash,
                "manifest_path": manifest_path,
                "manifest_sha256": _sha256(manifest_path),
                "verification_path": verification_path,
                "verification_sha256": _sha256(verification_path),
                "source_final_validation_loss": float(row["final_validation_loss"]),
                "source_R_model": float(row["R_model"]),
                "series_id": row["series_id"],
                "series_label": row["series_label"],
                "activation": row["activation"],
                "topology": manifest["topology"],
                "pressure_method": row["pressure_method"],
                "dose_name": row["dose_name"],
                "dose": float(row["dose"]),
            }
        )
    return sources


def load_reused_controls() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    result = json.loads(CONTROL_RESULT_PATH.read_text(encoding="utf-8"))
    if result.get("schema_version") != 1 or result.get("status") != "complete_verified":
        raise ValueError("Analysis 005 controls are not complete verified evidence.")
    method = result.get("method", {})
    if (
        tuple(method.get("sites", [])) != SITES
        or tuple(float(value) for value in method.get("target_sparsities", []))
        != TARGET_SPARSITIES
    ):
        raise ValueError("Analysis 005 controls use a different clipping protocol.")
    rows = result.get("conditions", [])
    if len(rows) != 20:
        raise ValueError("Analysis 005 must contain twenty reused control points.")
    for condition_id in CONTROL_CONDITION_IDS:
        selected = sorted(
            (row for row in rows if row["condition_id"] == condition_id),
            key=lambda row: float(row["target_sparsity"]),
        )
        if tuple(float(row["target_sparsity"]) for row in selected) != TARGET_SPARSITIES:
            raise ValueError(f"Incomplete reused control sweep for {condition_id}.")
        for row in selected:
            for key, expected in EXPECTED_VALIDATION.items():
                if row["validation"].get(key) != expected:
                    raise ValueError(f"Reused control coverage mismatch at {condition_id}/{key}.")
            _validate_logical_products(row["logical_products"])
    return rows, result


def _validate_logical_products(logical: Mapping[str, Any]) -> None:
    numerator = sum(
        int(row["zero_product_count"]) for row in logical["per_operation"].values()
    )
    denominator = sum(
        int(row["product_count"]) for row in logical["per_operation"].values()
    )
    if numerator != int(logical["block_zero_product_count"]):
        raise ValueError("Logical zero-product numerator does not reconcile.")
    if denominator != int(logical["block_product_count"]):
        raise ValueError("Logical block denominator does not reconcile.")
    if numerator < 0 or numerator > denominator:
        raise ValueError("Logical zero-product numerator is outside its denominator.")
    expected = numerator / int(logical["model_product_count"])
    if not math.isclose(expected, float(logical["R_model"]), rel_tol=0.0, abs_tol=1e-16):
        raise ValueError("R_model is not derived from stored integer counts.")


def _protocol_sha256(
    sources: list[dict[str, Any]],
    train_metadata: Mapping[str, Any],
    validation_metadata: Mapping[str, Any],
) -> str:
    protocol = {
        "analysis_code_sha256": _sha256(Path(__file__)),
        "frozen_analysis005_evaluator_sha256": _sha256(CONTROL_EVALUATOR_PATH),
        "analysis004_comparison_sha256": _sha256(COMPARISON_PATH),
        "reused_control_result_sha256": _sha256(CONTROL_RESULT_PATH),
        "conditions": [
            {
                "condition_id": source["condition_id"],
                "checkpoint_content_sha256": source["checkpoint_content_sha256"],
                "manifest_sha256": source["manifest_sha256"],
            }
            for source in sources
        ],
        "sites": SITES,
        "target_sparsities": TARGET_SPARSITIES,
        "calibration_blocks": TEAL.CALIBRATION_BLOCKS,
        "calibration_split": "train",
        "calibration_block_selection": "first ten complete source-order blocks",
        "block_size": TEAL.BLOCK_SIZE,
        "evaluation_batch_size": TEAL.EVALUATION_BATCH_SIZE,
        "autocast_precision": TEAL.AUTOCAST_PRECISION,
        "train_cache_sha256": train_metadata["tokens_sha256"],
        "validation_cache_sha256": validation_metadata["tokens_sha256"],
        "threshold_rule": "smallest empirical absolute-value order statistic reaching target; abs(x) <= t becomes zero",
        "matrix_inputs": {"a": "QKV", "m": "W1", "h": "W2", "z": "W_o"},
    }
    return hashlib.sha256(_canonical_json(protocol).encode("utf-8")).hexdigest()


def _load_progress(protocol_sha256: str) -> dict[tuple[str, float], dict[str, Any]]:
    if not PROGRESS_PATH.exists():
        return {}
    rows: dict[tuple[str, float], dict[str, Any]] = {}
    for line_number, line in enumerate(PROGRESS_PATH.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if row.get("protocol_sha256") != protocol_sha256:
            raise ValueError("Existing progress belongs to a different evaluation protocol.")
        key = (str(row["condition_id"]), float(row["target_sparsity"]))
        if key in rows:
            raise ValueError(f"Duplicate progress row at line {line_number}: {key}")
        rows[key] = row
    return rows


def _append_progress(row: Mapping[str, Any]) -> None:
    PROGRESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROGRESS_PATH.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical_json(row) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _calibration_path(condition_id: str) -> Path:
    return CALIBRATION_DIR / f"{condition_id}.json"


def _source_public(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.relative_to(REPO_DIR).as_posix() if isinstance(value, Path) else value
        for key, value in source.items()
    }


def _tag_row(row: Mapping[str, Any], source: Mapping[str, Any], origin: str) -> dict[str, Any]:
    value = dict(row)
    topology = source["topology"]
    topology_id = topology["topology_id"] if isinstance(topology, Mapping) else topology
    value.update(
        {
            "run": source["run"],
            "series_id": source["series_id"],
            "series_label": source["series_label"],
            "source_topology_id": topology_id,
            "source_pressure_method": source["pressure_method"],
            "dose_name": source["dose_name"],
            "dose": source["dose"],
            "evidence_origin": origin,
        }
    )
    return value


def _result_rows(
    comparison: Mapping[str, Any],
    progress: Mapping[tuple[str, float], dict[str, Any]],
    sources: list[dict[str, Any]],
    reused_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    comparison_by_id = {row["condition_id"]: row for row in comparison["conditions"]}
    source_by_id = {source["condition_id"]: source for source in sources}
    reused_by_key = {
        (row["condition_id"], float(row["target_sparsity"])): row for row in reused_rows
    }
    conditions: list[dict[str, Any]] = []
    for condition in comparison["conditions"]:
        condition_id = condition["condition_id"]
        source_metadata = {
            "run": condition["run"],
            "series_id": condition["series_id"],
            "series_label": condition["series_label"],
            "topology": condition["topology"],
            "pressure_method": condition["pressure_method"],
            "dose_name": condition["dose_name"],
            "dose": condition["dose"],
        }
        rows = []
        for target in TARGET_SPARSITIES:
            if condition_id in CONTROL_CONDITION_IDS:
                row = _tag_row(
                    reused_by_key[(condition_id, target)],
                    source_metadata,
                    "analysis_005_reused_control",
                )
            else:
                row = _tag_row(
                    progress[(condition_id, target)],
                    source_by_id[condition_id],
                    "analysis_006_new_evaluation",
                )
            rows.append(row)
        baseline_loss = float(rows[0]["validation"]["loss"])
        for row, flag in zip(rows, TEAL.nondominated(rows), strict=True):
            row["loss_delta_from_zero_threshold"] = (
                float(row["validation"]["loss"]) - baseline_loss
            )
            row["nondominated_within_condition"] = flag
            conditions.append(row)

    global_flags = TEAL.nondominated(conditions)
    for row, flag in zip(conditions, global_flags, strict=True):
        row["nondominated_global"] = flag
    if set(comparison_by_id) != set(ALL_CONDITION_IDS):
        raise ValueError("Combined result does not cover the approved source set.")
    return conditions


def _evaluate_smoke(
    model: Any,
    validation_tokens: Any,
    thresholds: Mapping[str, float],
    *,
    device: Any,
    torch: Any,
    np: Any,
    autocast_dtype: Any,
) -> dict[str, Any]:
    from sparsity_research.evaluation import evaluate_complete_blocks
    from sparsity_research.logical_capture import LogicalProductAccumulator, capture_logical_products
    from sparsity_research.metrics import ActivationAccumulator

    subset_tokens = validation_tokens[: SMOKE_VALIDATION_BLOCKS * TEAL.BLOCK_SIZE]
    activation = ActivationAccumulator((0.0,))
    logical = LogicalProductAccumulator()
    started = perf_counter()
    with TEAL.threshold_hooks(model, thresholds, torch=torch) as captured:

        def consume(_output: Any, _sequences: int) -> None:
            if len(captured) != 6 * len(SITES):
                raise ValueError("Smoke did not capture every clipped matrix input.")
            activation.update(captured, torch=torch)
            captured.clear()

        with capture_logical_products(model, accumulator=logical, torch=torch):
            coverage = evaluate_complete_blocks(
                model=model,
                tokens=subset_tokens,
                block_size=TEAL.BLOCK_SIZE,
                batch_size=1,
                device=device,
                torch=torch,
                np=np,
                autocast_dtype=autocast_dtype,
                after_batch=consume,
            )
    seconds = perf_counter() - started
    summary = logical.summary(model=model, total_input_tokens=coverage["input_tokens"])
    if coverage["sequences"] != SMOKE_VALIDATION_BLOCKS:
        raise ValueError("Smoke validation coverage is incomplete.")
    if sum(
        int(row["zero_product_count"]) for row in summary["per_operation"].values()
    ) != int(summary["block_zero_product_count"]):
        raise ValueError("Smoke logical counts do not reconcile.")
    return {
        "validation": coverage,
        "logical_products": summary,
        "activation_rows": activation.rows(),
        "evaluation_seconds": seconds,
        "input_tokens_per_second": coverage["input_tokens"] / seconds,
    }


def smoke() -> Path:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata

    if not torch.cuda.is_available():
        raise RuntimeError("The representative smoke requires the local CUDA GPU.")
    sources = load_source_descriptors(verify_checkpoint_content=True)
    source = sources[0]
    train_tokens, train_metadata, _ = TEAL._cache(TRAIN_METADATA_PATH, np=np)
    validation_tokens, validation_metadata, _ = TEAL._cache(VALIDATION_METADATA_PATH, np=np)
    device = torch.device("cuda")
    model = load_checkpoint_pythia(AutoModelForCausalLM, source["checkpoint"], torch=torch)
    model.config.use_cache = False
    model.to(device=device, dtype=torch.float32)
    if topology_metadata(model) != source["topology"]:
        raise ValueError("Smoke checkpoint topology differs from its manifest.")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    calibration = TEAL._calibrate(
        model,
        train_tokens,
        device=device,
        torch=torch,
        np=np,
        autocast_dtype=torch.float16,
    )
    thresholds = TEAL._thresholds_for_target(calibration, SMOKE_TARGET)
    point = _evaluate_smoke(
        model,
        validation_tokens,
        thresholds,
        device=device,
        torch=torch,
        np=np,
        autocast_dtype=torch.float16,
    )
    torch.cuda.synchronize()
    result = {
        "schema_version": 1,
        "status": "passed_non_evidence_smoke",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "analysis_code_sha256": _sha256(Path(__file__)),
        "frozen_analysis005_evaluator_sha256": _sha256(CONTROL_EVALUATOR_PATH),
        "analysis004_comparison_sha256": _sha256(COMPARISON_PATH),
        "verified_new_checkpoint_count": len(sources),
        "condition_id": source["condition_id"],
        "attempt_id": source["attempt_id"],
        "checkpoint_content_sha256": source["checkpoint_content_sha256"],
        "target_sparsity": SMOKE_TARGET,
        "calibration_blocks": TEAL.CALIBRATION_BLOCKS,
        "calibration_input_tokens": TEAL.CALIBRATION_BLOCKS * TEAL.BLOCK_SIZE,
        "calibration_seconds": calibration["wall_seconds"],
        "threshold_tensor_count": len(thresholds),
        "validation_blocks": SMOKE_VALIDATION_BLOCKS,
        "validation_input_tokens": SMOKE_VALIDATION_BLOCKS * TEAL.BLOCK_SIZE,
        "validation_loss": point["validation"]["loss"],
        "R_model": point["logical_products"]["R_model"],
        "evaluation_seconds": point["evaluation_seconds"],
        "input_tokens_per_second": point["input_tokens_per_second"],
        "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "transformers": __import__("transformers").__version__,
        "train_cache_sha256": train_metadata["tokens_sha256"],
        "validation_cache_sha256": validation_metadata["tokens_sha256"],
        "notes": "This bounded smoke is implementation/resource evidence, not a scientific result.",
    }
    _write_json(SMOKE_PATH, result)
    print(_canonical_json(result), flush=True)
    return SMOKE_PATH


def evaluate() -> Path:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata

    if not torch.cuda.is_available():
        raise RuntimeError("The approved execution requires the local CUDA GPU.")
    started = perf_counter()
    comparison = _comparison()
    sources = load_source_descriptors(verify_checkpoint_content=True)
    reused_rows, reused_result = load_reused_controls()
    train_tokens, train_metadata, train_path = TEAL._cache(TRAIN_METADATA_PATH, np=np)
    validation_tokens, validation_metadata, validation_path = TEAL._cache(
        VALIDATION_METADATA_PATH, np=np
    )
    protocol_sha256 = _protocol_sha256(sources, train_metadata, validation_metadata)
    progress = _load_progress(protocol_sha256)
    total_points = len(sources) * len(TARGET_SPARSITIES)
    completed_points = len(progress)
    started_at = datetime.now(timezone.utc).isoformat()
    _write_json(
        RUN_STATE_PATH,
        {
            "schema_version": 1,
            "status": "running",
            "pid": os.getpid(),
            "started_at": started_at,
            "protocol_sha256": protocol_sha256,
            "completed_points_at_start": completed_points,
            "total_points": total_points,
        },
    )
    print(
        _canonical_json(
            {
                "event": "start",
                "device": torch.cuda.get_device_name(0),
                "protocol_sha256": protocol_sha256,
                "completed_points": completed_points,
                "total_points": total_points,
            }
        ),
        flush=True,
    )

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    calibrations: dict[str, dict[str, Any]] = {}
    for source in sources:
        condition_id = source["condition_id"]
        path = _calibration_path(condition_id)
        if path.exists():
            artifact = json.loads(path.read_text(encoding="utf-8"))
            if artifact.get("protocol_sha256") != protocol_sha256:
                raise ValueError(f"Existing calibration has a different protocol: {condition_id}")
            calibrations[condition_id] = artifact["calibration"]
            continue
        model = load_checkpoint_pythia(AutoModelForCausalLM, source["checkpoint"], torch=torch)
        model.config.use_cache = False
        model.to(device="cuda", dtype=torch.float32)
        if topology_metadata(model) != source["topology"]:
            raise ValueError(f"Reloaded topology differs for {condition_id}.")
        torch.cuda.synchronize()
        calibration = TEAL._calibrate(
            model,
            train_tokens,
            device=torch.device("cuda"),
            torch=torch,
            np=np,
            autocast_dtype=torch.float16,
        )
        torch.cuda.synchronize()
        _write_json(
            path,
            {
                "schema_version": 1,
                "protocol_sha256": protocol_sha256,
                "condition_id": condition_id,
                "attempt_id": source["attempt_id"],
                "checkpoint_content_sha256": source["checkpoint_content_sha256"],
                "calibration": calibration,
            },
        )
        calibrations[condition_id] = calibration
        print(
            _canonical_json(
                {
                    "event": "calibration_complete",
                    "condition_id": condition_id,
                    "wall_seconds": calibration["wall_seconds"],
                }
            ),
            flush=True,
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    durations = [float(row["evaluation_seconds"]) for row in progress.values()]
    for source in sources:
        condition_id = source["condition_id"]
        remaining = [
            target for target in TARGET_SPARSITIES if (condition_id, target) not in progress
        ]
        if not remaining:
            continue
        model = load_checkpoint_pythia(AutoModelForCausalLM, source["checkpoint"], torch=torch)
        model.config.use_cache = False
        model.to(device="cuda", dtype=torch.float32)
        if topology_metadata(model) != source["topology"]:
            raise ValueError(f"Reloaded topology differs for {condition_id}.")
        for target in remaining:
            thresholds = TEAL._thresholds_for_target(calibrations[condition_id], target)
            torch.cuda.synchronize()
            point = TEAL._evaluate_point(
                model,
                validation_tokens,
                thresholds,
                device=torch.device("cuda"),
                torch=torch,
                np=np,
                autocast_dtype=torch.float16,
            )
            torch.cuda.synchronize()
            row = {
                "protocol_sha256": protocol_sha256,
                "condition_id": condition_id,
                "activation": source["activation"],
                "attempt_id": source["attempt_id"],
                "checkpoint_content_sha256": source["checkpoint_content_sha256"],
                "target_sparsity": target,
                **point,
            }
            _validate_logical_products(row["logical_products"])
            if target == 0.0 and abs(
                float(point["validation"]["loss"])
                - source["source_final_validation_loss"]
            ) > 5e-4:
                raise ValueError(f"Zero-threshold loss did not reproduce {condition_id}.")
            _append_progress(row)
            progress[(condition_id, target)] = row
            durations.append(float(row["evaluation_seconds"]))
            completed_points += 1
            median_seconds = float(np.median(durations))
            print(
                _canonical_json(
                    {
                        "event": "point_complete",
                        "condition_id": condition_id,
                        "target_sparsity": target,
                        "loss": point["validation"]["loss"],
                        "R_model": point["logical_products"]["R_model"],
                        "input_tokens_per_second": point["input_tokens_per_second"],
                        "completed_points": completed_points,
                        "total_points": total_points,
                        "estimated_remaining_seconds": median_seconds
                        * (total_points - completed_points),
                    }
                ),
                flush=True,
            )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    conditions = _result_rows(comparison, progress, sources, reused_rows)
    result = {
        "schema_version": 1,
        "status": "complete_verified",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha256,
        "question": "How does uniform TEAL-style post-hoc clipping change every full-pass A1-H and A4-Z checkpoint frontier?",
        "method": reused_result["method"],
        "source_counts": {
            "checkpoints": 15,
            "new_checkpoints": 13,
            "reused_control_checkpoints": 2,
            "points": 150,
            "new_points": 130,
            "reused_control_points": 20,
        },
        "source_artifacts": {
            "analysis_004_comparison": {
                "path": COMPARISON_PATH.relative_to(REPO_DIR).as_posix(),
                "sha256": _sha256(COMPARISON_PATH),
            },
            "analysis_005_controls": {
                "path": CONTROL_RESULT_PATH.relative_to(REPO_DIR).as_posix(),
                "sha256": _sha256(CONTROL_RESULT_PATH),
                "protocol_sha256": reused_result["protocol_sha256"],
            },
            "frozen_analysis_005_evaluator": {
                "path": CONTROL_EVALUATOR_PATH.relative_to(REPO_DIR).as_posix(),
                "sha256": _sha256(CONTROL_EVALUATOR_PATH),
            },
        },
        "sources": [_source_public(source) for source in sources],
        "caches": {
            "train": {
                "path": train_path.relative_to(REPO_DIR).as_posix(),
                "metadata": train_metadata,
            },
            "validation": {
                "path": validation_path.relative_to(REPO_DIR).as_posix(),
                "metadata": validation_metadata,
            },
        },
        "coverage": {"documents": 500, **EXPECTED_VALIDATION, "seed_count": 1},
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "autocast_dtype": "torch.float16",
            "new_evaluation_wall_seconds": perf_counter() - started,
        },
        "calibration_artifacts": {
            source["condition_id"]: _calibration_path(source["condition_id"])
            .relative_to(ANALYSIS_DIR)
            .as_posix()
            for source in sources
        },
        "conditions": conditions,
    }
    _write_json(RESULT_PATH, result)
    _write_json(
        RUN_STATE_PATH,
        {
            "schema_version": 1,
            "status": "completed",
            "pid": os.getpid(),
            "started_at": started_at,
            "finished_at": result["generated_at"],
            "protocol_sha256": protocol_sha256,
            "completed_points": total_points,
            "total_points": total_points,
            "result": RESULT_PATH.relative_to(ANALYSIS_DIR).as_posix(),
        },
    )
    print(_canonical_json({"event": "complete", "result": str(RESULT_PATH)}), flush=True)
    return RESULT_PATH


def main() -> Path:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a bounded non-evidence calibration/evaluation smoke.",
    )
    args = parser.parse_args()
    return smoke() if args.smoke else evaluate()


if __name__ == "__main__":
    main()
