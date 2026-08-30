"""Evaluate a uniform TEAL-style post-hoc clipping frontier on Run 004 controls."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_DIR = ANALYSIS_DIR.parents[1]
RUN_DIR = REPO_DIR / "runs" / "004-2026-08-29-pythia14m-full-pass-l1n"
VERIFICATION_PATH = RUN_DIR / "artifacts" / "verification.json"
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
RESULT_PATH = ANALYSIS_DIR / "teal_frontier.json"

CONDITION_IDS = ("gelu-control", "relu-control")
SITES = ("a", "m", "h", "z")
TARGET_SPARSITIES = tuple(index / 10 for index in range(10))
CALIBRATION_BLOCKS = 10
BLOCK_SIZE = 2048
EVALUATION_BATCH_SIZE = 1
AUTOCAST_PRECISION = "float16"
EXPECTED_VALIDATION = {
    "sequences": 338,
    "input_tokens": 692_224,
    "source_tokens": 693_668,
    "excluded_tail_tokens": 1_444,
    "complete_block_coverage": True,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _cache(metadata_path: Path, *, np: Any) -> tuple[Any, dict[str, Any], Path]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    token_path = metadata_path.parent / str(metadata["tokens_path"])
    if token_path.stat().st_size != int(metadata["tokens_bytes"]):
        raise ValueError(f"Token-cache byte count changed: {token_path}")
    if _sha256(token_path) != metadata["tokens_sha256"]:
        raise ValueError(f"Token-cache SHA-256 changed: {token_path}")
    tokens = np.memmap(
        token_path,
        dtype=np.int32,
        mode="r",
        shape=(int(metadata["tokens"]),),
    )
    return tokens, metadata, token_path


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


def _load_sources() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    verification = json.loads(VERIFICATION_PATH.read_text(encoding="utf-8"))
    if verification.get("status") != "verified" or verification.get("evidence_label") != "valid":
        raise ValueError("Run 004 does not provide valid verified source evidence.")
    verified = {
        row["condition"]["id"]: row
        for row in verification["conditions"]
        if row["condition"]["id"] in CONDITION_IDS
    }
    if tuple(condition for condition in CONDITION_IDS if condition in verified) != CONDITION_IDS:
        raise ValueError("Run 004 verification does not contain both requested controls.")

    sources: list[dict[str, Any]] = []
    for condition_id in CONDITION_IDS:
        source = verified[condition_id]
        attempt_dir = RUN_DIR / "artifacts" / "attempts" / source["attempt_id"]
        manifest_path = attempt_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"Source attempt is not complete: {source['attempt_id']}")
        if manifest["condition"]["id"] != condition_id:
            raise ValueError(f"Source condition mismatch: {source['attempt_id']}")
        coverage_mismatches = [
            key
            for key, expected in EXPECTED_VALIDATION.items()
            if manifest["validation_coverage"].get(key) != expected
        ]
        if coverage_mismatches:
            raise ValueError(
                f"Source validation coverage changed for {source['attempt_id']}: "
                + ", ".join(coverage_mismatches)
            )
        checkpoint = attempt_dir / manifest["checkpoints"]["final"]["path"]
        checkpoint_hash = _inventory_content_sha256(checkpoint)
        expected_hash = source["checkpoint_content_sha256"]
        if checkpoint_hash != expected_hash or manifest["checkpoints"]["final"]["content_sha256"] != expected_hash:
            raise ValueError(f"Final checkpoint content changed: {source['attempt_id']}")
        sources.append(
            {
                "condition_id": condition_id,
                "activation": source["condition"]["activation"],
                "attempt_id": source["attempt_id"],
                "checkpoint": checkpoint,
                "checkpoint_content_sha256": checkpoint_hash,
                "source_final_validation_loss": float(source["final_validation_loss"]),
                "manifest_path": manifest_path,
                "manifest_sha256": _sha256(manifest_path),
                "topology": manifest["topology"],
            }
        )
    return sources, verification


def _protocol_sha256(
    sources: list[dict[str, Any]], train_metadata: Mapping[str, Any], validation_metadata: Mapping[str, Any]
) -> str:
    protocol = {
        "analysis_code_sha256": _sha256(Path(__file__)),
        "conditions": [
            {
                "condition_id": source["condition_id"],
                "checkpoint_content_sha256": source["checkpoint_content_sha256"],
            }
            for source in sources
        ],
        "sites": SITES,
        "target_sparsities": TARGET_SPARSITIES,
        "calibration_blocks": CALIBRATION_BLOCKS,
        "calibration_split": "train",
        "calibration_block_selection": "first ten complete source-order blocks",
        "block_size": BLOCK_SIZE,
        "evaluation_batch_size": EVALUATION_BATCH_SIZE,
        "autocast_precision": AUTOCAST_PRECISION,
        "train_cache_sha256": train_metadata["tokens_sha256"],
        "validation_cache_sha256": validation_metadata["tokens_sha256"],
        "threshold_rule": "smallest empirical absolute-value order statistic reaching target; abs(x) <= t becomes zero",
        "teal_all_matrix_input_mapping": {
            "a": "fused QKV projection",
            "m": "MLP W1/up projection",
            "h": "MLP W2/down projection",
            "z": "attention output projection",
        },
    }
    return hashlib.sha256(_canonical_json(protocol).encode("utf-8")).hexdigest()


def _module_map(model: Any) -> dict[str, tuple[Any, str]]:
    modules: dict[str, tuple[Any, str]] = {}
    for layer_index, layer in enumerate(model.gpt_neox.layers):
        resolved = {
            "a": (layer.input_layernorm, "output"),
            "m": (layer.post_attention_layernorm, "output"),
            "h": (layer.mlp.act, "output"),
            "z": (layer.attention.dense, "input"),
        }
        for site, module_and_port in resolved.items():
            modules[f"{site}.layer_{layer_index}"] = module_and_port
    if len(modules) != 6 * len(SITES):
        raise ValueError("Expected four TEAL matrix-input sites in all six Pythia blocks.")
    return modules


def _first_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    raise TypeError("Hook output did not contain a tensor.")


def empirical_thresholds(values: Any, targets: tuple[float, ...], *, np: Any) -> dict[str, Any]:
    """Return exact empirical thresholds using the smallest qualifying order statistic."""

    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if flat.size <= 0 or not bool(np.isfinite(flat).all()) or bool((flat < 0).any()):
        raise ValueError("Calibration requires finite absolute activation values.")
    target_indices = {
        target: None if target == 0.0 else max(0, math.ceil(target * flat.size) - 1)
        for target in targets
    }
    kth = sorted({index for index in target_indices.values() if index is not None})
    partitioned = np.partition(flat, kth) if kth else flat
    rows = []
    for target in targets:
        index = target_indices[target]
        threshold = 0.0 if index is None else float(partitioned[index])
        hits = int(np.count_nonzero(flat <= threshold))
        rows.append(
            {
                "target_sparsity": target,
                "threshold": threshold,
                "calibration_zero_count": hits,
                "calibration_total": int(flat.size),
                "calibration_fraction": hits / int(flat.size),
            }
        )
    return {
        "total": int(flat.size),
        "natural_exact_zero_count": int(np.count_nonzero(flat == 0.0)),
        "targets": rows,
    }


def _calibrate(
    model: Any,
    train_tokens: Any,
    *,
    device: Any,
    torch: Any,
    np: Any,
    autocast_dtype: Any,
) -> dict[str, Any]:
    modules = _module_map(model)
    captured: dict[str, list[Any]] = {name: [] for name in modules}
    handles = []

    def output_hook(name: str) -> Any:
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = _first_tensor(output)
            captured[name].append(tensor.detach().abs().float().cpu().reshape(-1).numpy())

        return capture

    def input_hook(name: str) -> Any:
        def capture(_module: Any, inputs: Any) -> None:
            tensor = _first_tensor(inputs)
            captured[name].append(tensor.detach().abs().float().cpu().reshape(-1).numpy())

        return capture

    for name, (module, port) in modules.items():
        if port == "output":
            handles.append(module.register_forward_hook(output_hook(name)))
        elif port == "input":
            handles.append(module.register_forward_pre_hook(input_hook(name)))
        else:
            raise ValueError(f"Unsupported capture port for {name}: {port}")
    started = perf_counter()
    model.eval()
    try:
        with torch.no_grad():
            for block_index in range(CALIBRATION_BLOCKS):
                start = block_index * BLOCK_SIZE
                array = np.asarray(train_tokens[start : start + BLOCK_SIZE], dtype=np.int64)[None, :]
                input_ids = torch.as_tensor(array, dtype=torch.long, device=device)
                with torch.autocast(device_type=device.type, dtype=autocast_dtype):
                    model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()

    statistics = {}
    for name in sorted(captured):
        if len(captured[name]) != CALIBRATION_BLOCKS:
            raise ValueError(f"Calibration capture is incomplete for {name}.")
        values = np.concatenate(captured[name])
        statistics[name] = empirical_thresholds(values, TARGET_SPARSITIES, np=np)
        del values
    return {
        "blocks": CALIBRATION_BLOCKS,
        "input_tokens": CALIBRATION_BLOCKS * BLOCK_SIZE,
        "source_order_block_indices": list(range(CALIBRATION_BLOCKS)),
        "wall_seconds": perf_counter() - started,
        "statistics": statistics,
    }


def _thresholds_for_target(calibration: Mapping[str, Any], target: float) -> dict[str, float]:
    thresholds = {}
    for name, statistics in calibration["statistics"].items():
        matches = [
            row for row in statistics["targets"] if math.isclose(row["target_sparsity"], target)
        ]
        if len(matches) != 1:
            raise ValueError(f"Calibration target {target} is missing for {name}.")
        thresholds[name] = float(matches[0]["threshold"])
    return thresholds


@contextmanager
def threshold_hooks(
    model: Any, thresholds: Mapping[str, float], *, torch: Any
) -> Iterator[dict[str, Any]]:
    modules = _module_map(model)
    if set(modules) != set(thresholds):
        raise ValueError("Threshold map does not exactly cover every TEAL matrix-input tensor.")
    handles = []

    activations: dict[str, Any] = {}

    def output_hook(name: str) -> Any:
        threshold = float(thresholds[name])
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"Invalid threshold for {name}: {threshold}")

        def clip(_module: Any, _inputs: Any, output: Any) -> Any:
            value = _first_tensor(output)
            clipped = value.masked_fill(value.detach().abs() <= threshold, 0.0)
            activations[name] = clipped
            if value is output:
                return clipped
            raise TypeError("TEAL matrix-input hooks require direct tensor module outputs.")

        return clip

    def input_hook(name: str) -> Any:
        threshold = float(thresholds[name])
        if not math.isfinite(threshold) or threshold < 0.0:
            raise ValueError(f"Invalid threshold for {name}: {threshold}")

        def clip(_module: Any, inputs: Any) -> Any:
            value = _first_tensor(inputs)
            clipped = value.masked_fill(value.detach().abs() <= threshold, 0.0)
            activations[name] = clipped
            if not isinstance(inputs, tuple) or not inputs or inputs[0] is not value:
                raise TypeError("TEAL matrix-input pre-hooks require a first tensor input.")
            return (clipped, *inputs[1:])

        return clip

    try:
        for name, (module, port) in modules.items():
            if port == "output":
                handles.append(module.register_forward_hook(output_hook(name)))
            elif port == "input":
                handles.append(module.register_forward_pre_hook(input_hook(name)))
            else:
                raise ValueError(f"Unsupported clipping port for {name}: {port}")
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def _pooled_site_counts(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pooled = []
    for site in SITES:
        selected = [row for row in rows if row["name"].startswith(f"{site}.layer_")]
        if len(selected) != 6:
            raise ValueError(f"Expected six activation rows for site {site}.")
        zero_count = sum(int(row["exact_zero_count"]) for row in selected)
        total = sum(int(row["total"]) for row in selected)
        pooled.append(
            {
                "site": site,
                "exact_zero_count": zero_count,
                "total": total,
                "exact_zero_fraction": zero_count / total,
            }
        )
    return pooled


def _evaluate_point(
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

    activation = ActivationAccumulator((0.0,))
    logical = LogicalProductAccumulator()
    started = perf_counter()
    with threshold_hooks(model, thresholds, torch=torch) as captured:

        def consume(_output: Any, _sequences: int) -> None:
            if len(captured) != 6 * len(SITES):
                raise ValueError("TEAL clipping did not capture every matrix input.")
            activation.update(captured, torch=torch)
            captured.clear()

        with capture_logical_products(model, accumulator=logical, torch=torch):
            coverage = evaluate_complete_blocks(
                model=model,
                tokens=validation_tokens,
                block_size=BLOCK_SIZE,
                batch_size=EVALUATION_BATCH_SIZE,
                device=device,
                torch=torch,
                np=np,
                autocast_dtype=autocast_dtype,
                after_batch=consume,
            )
    for key, expected in EXPECTED_VALIDATION.items():
        if coverage.get(key) != expected:
            raise ValueError(f"Complete-validation coverage mismatch at {key}.")
    layer_rows = activation.rows()
    logical_summary = logical.summary(model=model, total_input_tokens=coverage["input_tokens"])
    operation_rows = logical_summary["per_operation"].values()
    block_zero_count = sum(int(row["zero_product_count"]) for row in operation_rows)
    block_product_count = sum(
        int(row["product_count"]) for row in logical_summary["per_operation"].values()
    )
    if block_zero_count != int(logical_summary["block_zero_product_count"]):
        raise ValueError("Logical zero-product counts do not reconcile.")
    if block_product_count != int(logical_summary["block_product_count"]):
        raise ValueError("Logical product denominators do not reconcile.")
    seconds = perf_counter() - started
    return {
        "validation": coverage,
        "thresholds_by_site_layer": dict(sorted(thresholds.items())),
        "activation_rows": layer_rows,
        "activations_by_site": _pooled_site_counts(layer_rows),
        "logical_products": logical_summary,
        "evaluation_seconds": seconds,
        "input_tokens_per_second": coverage["input_tokens"] / seconds,
    }


def nondominated(rows: list[Mapping[str, Any]]) -> list[bool]:
    """Mark points nondominated under lower loss and higher measured R_model."""

    result = []
    for index, row in enumerate(rows):
        loss = float(row["validation"]["loss"])
        opportunity = float(row["logical_products"]["R_model"])
        dominated = any(
            other_index != index
            and float(other["validation"]["loss"]) <= loss
            and float(other["logical_products"]["R_model"]) >= opportunity
            and (
                float(other["validation"]["loss"]) < loss
                or float(other["logical_products"]["R_model"]) > opportunity
            )
            for other_index, other in enumerate(rows)
        )
        result.append(not dominated)
    return result


def _load_progress(protocol_sha256: str) -> dict[tuple[str, float], dict[str, Any]]:
    if not PROGRESS_PATH.exists():
        return {}
    rows = {}
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


def main() -> Path:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata

    if not torch.cuda.is_available():
        raise RuntimeError("The approved local execution requires the available CUDA GPU.")
    device = torch.device("cuda")
    autocast_dtype = torch.float16
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)

    started = perf_counter()
    sources, verification = _load_sources()
    train_tokens, train_metadata, train_path = _cache(TRAIN_METADATA_PATH, np=np)
    validation_tokens, validation_metadata, validation_path = _cache(VALIDATION_METADATA_PATH, np=np)
    protocol_sha256 = _protocol_sha256(sources, train_metadata, validation_metadata)
    progress = _load_progress(protocol_sha256)
    total_points = len(sources) * len(TARGET_SPARSITIES)
    completed_points = len(progress)
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

    calibrations = {}
    for source in sources:
        condition_id = source["condition_id"]
        calibration_path = _calibration_path(condition_id)
        if calibration_path.exists():
            calibration_artifact = json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration_artifact.get("protocol_sha256") != protocol_sha256:
                raise ValueError(f"Existing calibration has a different protocol: {condition_id}")
            calibrations[condition_id] = calibration_artifact["calibration"]
            continue

        model = load_checkpoint_pythia(AutoModelForCausalLM, source["checkpoint"], torch=torch)
        model.config.use_cache = False
        model.to(device=device, dtype=torch.float32)
        if topology_metadata(model) != source["topology"]:
            raise ValueError(f"Reloaded topology differs for {condition_id}.")
        torch.cuda.synchronize()
        calibration = _calibrate(
            model,
            train_tokens,
            device=device,
            torch=torch,
            np=np,
            autocast_dtype=autocast_dtype,
        )
        torch.cuda.synchronize()
        artifact = {
            "schema_version": 1,
            "protocol_sha256": protocol_sha256,
            "condition_id": condition_id,
            "attempt_id": source["attempt_id"],
            "checkpoint_content_sha256": source["checkpoint_content_sha256"],
            "calibration": calibration,
        }
        _write_json(calibration_path, artifact)
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

    point_durations = [float(row["evaluation_seconds"]) for row in progress.values()]
    for source in sources:
        condition_id = source["condition_id"]
        remaining_targets = [
            target for target in TARGET_SPARSITIES if (condition_id, target) not in progress
        ]
        if not remaining_targets:
            continue
        model = load_checkpoint_pythia(AutoModelForCausalLM, source["checkpoint"], torch=torch)
        model.config.use_cache = False
        model.to(device=device, dtype=torch.float32)
        if topology_metadata(model) != source["topology"]:
            raise ValueError(f"Reloaded topology differs for {condition_id}.")
        for target in remaining_targets:
            thresholds = _thresholds_for_target(calibrations[condition_id], target)
            torch.cuda.synchronize()
            point = _evaluate_point(
                model,
                validation_tokens,
                thresholds,
                device=device,
                torch=torch,
                np=np,
                autocast_dtype=autocast_dtype,
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
            if target == 0.0 and abs(
                float(point["validation"]["loss"]) - source["source_final_validation_loss"]
            ) > 5e-4:
                raise ValueError(f"Zero-threshold loss did not reproduce Run 004 for {condition_id}.")
            _append_progress(row)
            progress[(condition_id, target)] = row
            point_durations.append(float(row["evaluation_seconds"]))
            completed_points += 1
            median_seconds = float(np.median(point_durations))
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

    conditions = []
    for source in sources:
        rows = [progress[(source["condition_id"], target)] for target in TARGET_SPARSITIES]
        baseline_loss = float(rows[0]["validation"]["loss"])
        flags = nondominated(rows)
        for row, flag in zip(rows, flags, strict=True):
            row = dict(row)
            row["loss_delta_from_zero_threshold"] = float(row["validation"]["loss"]) - baseline_loss
            row["nondominated_within_condition"] = flag
            conditions.append(row)

    result = {
        "schema_version": 1,
        "status": "complete_verified",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol_sha256,
        "question": "How do uniform TEAL-style post-hoc thresholds change measured R_model versus full-validation loss for the Run 004 GeLU and ReLU controls?",
        "method": {
            "name": "uniform TEAL-style input magnitude clipping",
            "sites": list(SITES),
            "matrix_mapping": {
                "a": "fused QKV projection",
                "m": "MLP W1/up projection",
                "h": "MLP W2/down projection",
                "z": "attention output projection",
            },
            "target_sparsities": list(TARGET_SPARSITIES),
            "calibration": {
                "split": "train",
                "blocks": CALIBRATION_BLOCKS,
                "input_tokens": CALIBRATION_BLOCKS * BLOCK_SIZE,
                "selection": "first ten complete source-order blocks",
                "threshold_scope": "one empirical absolute-value threshold per matrix-input site and Pythia block",
            },
            "differences_from_original_teal": [
                "Pythia uses one fused QKV matrix rather than separate Q, K, and V matrices.",
                "This evaluates full-sequence pretraining loss rather than batch-one autoregressive decoding.",
                "This is the uniform target-sparsity variant, not TEAL block-wise greedy allocation.",
                "No sparse kernel or runtime speedup is evaluated.",
            ],
        },
        "source_verification": {
            "path": VERIFICATION_PATH.relative_to(REPO_DIR).as_posix(),
            "sha256": _sha256(VERIFICATION_PATH),
            "status": verification["status"],
            "evidence_label": verification["evidence_label"],
        },
        "sources": [
            {
                key: (
                    value.relative_to(REPO_DIR).as_posix()
                    if isinstance(value, Path)
                    else value
                )
                for key, value in source.items()
            }
            for source in sources
        ],
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
        "coverage": {
            "documents": 500,
            **EXPECTED_VALIDATION,
            "seed_count": 1,
        },
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "autocast_dtype": str(autocast_dtype),
            "wall_seconds": perf_counter() - started,
        },
        "calibration_artifacts": {
            condition_id: _calibration_path(condition_id).relative_to(ANALYSIS_DIR).as_posix()
            for condition_id in CONDITION_IDS
        },
        "conditions": conditions,
    }
    _write_json(RESULT_PATH, result)
    print(_canonical_json({"event": "complete", "result": str(RESULT_PATH)}), flush=True)
    return RESULT_PATH


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
