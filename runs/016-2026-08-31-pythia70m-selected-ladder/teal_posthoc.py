"""Scale-generic Analysis 005/006 TEAL frontiers for Run 016 A0 and A1-H."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
from time import perf_counter
from typing import Any, Iterator, Mapping

from sparsity_research.artifacts import build_transfer_inventory

from run_config import (
    RUN_DIR,
    inventory_content_sha256,
    load_config,
    load_verified_caches,
    mapping,
    write_json,
)
from verification import verify_attempt


TEAL_ROOT = RUN_DIR / "artifacts" / "teal"
SITES = ("a", "m", "h", "z")
EXPECTED_VALIDATION = {
    "sequences": 338,
    "input_tokens": 692_224,
    "excluded_tail_tokens": 1_444,
    "complete_block_coverage": True,
}


def evaluate_condition(condition_id: str) -> Path:
    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata

    config = load_config()
    teal = mapping(config, "teal_posthoc")
    allowed = tuple(str(value) for value in teal["condition_ids"])
    if condition_id not in allowed:
        raise ValueError(f"TEAL post-hoc is approved only for {allowed}.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the approved TEAL evaluation.")
    source = _source(condition_id)
    train_tokens, validation_tokens, train_metadata, validation_metadata, _ = load_verified_caches(
        config, np=np
    )
    protocol = _protocol_sha256(config, source, train_metadata, validation_metadata)
    progress_path = TEAL_ROOT / condition_id / "progress.jsonl"
    calibration_path = TEAL_ROOT / condition_id / "calibration.json"
    result_path = TEAL_ROOT / f"{condition_id}.json"
    progress = _load_progress(progress_path, protocol)
    targets = tuple(float(value) for value in teal["target_sparsities"])
    device = torch.device("cuda")
    model = load_checkpoint_pythia(AutoModelForCausalLM, source["checkpoint"], torch=torch)
    model.config.use_cache = False
    model.to(device=device, dtype=torch.float32)
    if topology_metadata(model) != source["topology"]:
        raise ValueError(f"Reloaded topology differs for {condition_id}.")

    if calibration_path.exists():
        artifact = _json(calibration_path)
        if artifact.get("protocol_sha256") != protocol:
            raise ValueError("Existing calibration belongs to a different TEAL protocol.")
        calibration = artifact["calibration"]
    else:
        calibration = _calibrate(
            model,
            train_tokens,
            targets=targets,
            blocks=int(teal["calibration_blocks"]),
            block_size=int(mapping(config, "data")["sequence_length"]),
            device=device,
            torch=torch,
            np=np,
        )
        write_json(
            calibration_path,
            {
                "schema_version": 1,
                "protocol_sha256": protocol,
                "condition_id": condition_id,
                "checkpoint_content_sha256": source["checkpoint_content_sha256"],
                "calibration": calibration,
            },
        )

    started = perf_counter()
    for target in targets:
        if target in progress:
            continue
        point = _evaluate_point(
            model,
            validation_tokens,
            _thresholds_for_target(calibration, target),
            block_size=int(mapping(config, "data")["sequence_length"]),
            batch_size=int(teal["evaluation_batch_size"]),
            device=device,
            torch=torch,
            np=np,
        )
        row = {
            "protocol_sha256": protocol,
            "condition_id": condition_id,
            "attempt_id": source["attempt_id"],
            "checkpoint_content_sha256": source["checkpoint_content_sha256"],
            "target_sparsity": target,
            **point,
        }
        tolerance = float(teal["zero_threshold_loss_tolerance"])
        if target == 0.0 and abs(
            float(point["validation"]["loss"]) - float(source["source_final_validation_loss"])
        ) > tolerance:
            raise ValueError(f"p=0 loss did not reproduce the source endpoint for {condition_id}.")
        _append_progress(progress_path, row)
        progress[target] = row
        print(
            json.dumps(
                {
                    "event": "teal_point_complete",
                    "condition_id": condition_id,
                    "target_sparsity": target,
                    "loss": point["validation"]["loss"],
                    "R_model": point["logical_products"]["R_model"],
                },
                sort_keys=True,
            ),
            flush=True,
        )

    rows = [progress[target] for target in targets]
    flags = nondominated(rows)
    baseline = float(rows[0]["validation"]["loss"])
    output_rows = []
    for row, flag in zip(rows, flags, strict=True):
        enriched = dict(row)
        enriched["loss_delta_from_zero_threshold"] = float(row["validation"]["loss"]) - baseline
        enriched["nondominated_within_condition"] = flag
        output_rows.append(enriched)
    result = {
        "schema_version": 1,
        "status": "complete_verified",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_sha256": protocol,
        "condition_id": condition_id,
        "question": "How does uniform TEAL-style post-hoc clipping change R_model versus full-validation loss?",
        "method": {
            "source_protocol": "analyses/005 and analyses/006",
            "sites": list(SITES),
            "target_sparsities": list(targets),
            "calibration_blocks": int(teal["calibration_blocks"]),
            "calibration_selection": "first complete source-order training blocks",
            "threshold_scope": "one empirical absolute-value threshold per site and model layer",
            "evaluation_only": True,
            "measured_speedup": False,
        },
        "source": _serializable_source(source),
        "cache_identity": {
            "train_tokens_sha256": train_metadata["tokens_sha256"],
            "validation_tokens_sha256": validation_metadata["tokens_sha256"],
        },
        "runtime": {
            "device": torch.cuda.get_device_name(0),
            "torch": torch.__version__,
            "transformers": __import__("transformers").__version__,
            "wall_seconds_this_invocation": perf_counter() - started,
        },
        "weight_statistics": source["weight_statistics"],
        "calibration_path": calibration_path.relative_to(RUN_DIR).as_posix(),
        "points": output_rows,
    }
    write_json(result_path, result)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return result_path


def consolidate() -> Path:
    config = load_config()
    ids = tuple(str(value) for value in mapping(config, "teal_posthoc")["condition_ids"])
    sources = [_json(TEAL_ROOT / f"{condition_id}.json") for condition_id in ids]
    if any(row.get("status") != "complete_verified" for row in sources):
        raise ValueError("Both TEAL condition frontiers must be complete and verified.")
    points = [point for source in sources for point in source["points"]]
    expected_targets = tuple(float(value) for value in mapping(config, "teal_posthoc")["target_sparsities"])
    for condition_id in ids:
        selected = [row for row in points if row["condition_id"] == condition_id]
        if tuple(float(row["target_sparsity"]) for row in selected) != expected_targets:
            raise ValueError(f"Incomplete TEAL grid for {condition_id}.")
    flags = nondominated(points)
    combined = []
    for row, flag in zip(points, flags, strict=True):
        value = dict(row)
        value["nondominated_across_controls"] = flag
        combined.append(value)
    output = TEAL_ROOT / "teal_frontiers.json"
    write_json(
        output,
        {
            "schema_version": 1,
            "status": "complete_verified",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "condition_ids": list(ids),
            "point_count": len(combined),
            "points": combined,
            "interpretation": "R_model is a logical-product opportunity, not runtime speedup.",
        },
    )
    return output


def empirical_thresholds(values: Any, targets: tuple[float, ...], *, np: Any) -> dict[str, Any]:
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


def _module_map(model: Any) -> dict[str, tuple[Any, str]]:
    layers = getattr(getattr(model, "gpt_neox", None), "layers", None)
    if layers is None or len(layers) <= 0:
        raise ValueError("TEAL module mapping requires GPT-NeoX layers.")
    modules = {}
    for layer_index, layer in enumerate(layers):
        resolved = {
            "a": (layer.input_layernorm, "output"),
            "m": (layer.post_attention_layernorm, "output"),
            "h": (layer.mlp.act, "output"),
            "z": (layer.attention.dense, "input"),
        }
        for site, module_and_port in resolved.items():
            modules[f"{site}.layer_{layer_index}"] = module_and_port
    if len(modules) != len(layers) * len(SITES):
        raise ValueError("TEAL mapping did not cover four sites in every model layer.")
    return modules


def _calibrate(
    model: Any,
    train_tokens: Any,
    *,
    targets: tuple[float, ...],
    blocks: int,
    block_size: int,
    device: Any,
    torch: Any,
    np: Any,
) -> dict[str, Any]:
    modules = _module_map(model)
    captured: dict[str, list[Any]] = {name: [] for name in modules}
    handles = []

    def hook(name: str, port: str):
        def capture(_module: Any, inputs: Any, output: Any = None) -> None:
            value = _first_tensor(output if port == "output" else inputs)
            captured[name].append(value.detach().abs().float().cpu().reshape(-1).numpy())

        return capture

    for name, (module, port) in modules.items():
        handles.append(
            module.register_forward_hook(hook(name, port))
            if port == "output"
            else module.register_forward_pre_hook(hook(name, port))
        )
    started = perf_counter()
    model.eval()
    try:
        with torch.no_grad():
            for block_index in range(blocks):
                start = block_index * block_size
                array = np.asarray(train_tokens[start : start + block_size], dtype=np.int64)[None, :]
                input_ids = torch.as_tensor(array, dtype=torch.long, device=device)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(input_ids=input_ids)
    finally:
        for handle in handles:
            handle.remove()
    statistics = {}
    for name in sorted(captured):
        if len(captured[name]) != blocks:
            raise ValueError(f"Calibration capture is incomplete for {name}.")
        values = np.concatenate(captured[name])
        statistics[name] = empirical_thresholds(values, targets, np=np)
    return {
        "blocks": blocks,
        "input_tokens": blocks * block_size,
        "source_order_block_indices": list(range(blocks)),
        "wall_seconds": perf_counter() - started,
        "statistics": statistics,
    }


@contextmanager
def threshold_hooks(
    model: Any, thresholds: Mapping[str, float]
) -> Iterator[dict[str, Any]]:
    modules = _module_map(model)
    if set(modules) != set(thresholds):
        raise ValueError("Thresholds must exactly cover every TEAL site-layer tensor.")
    handles = []
    activations: dict[str, Any] = {}

    def output_hook(name: str):
        threshold = _valid_threshold(thresholds[name], name)

        def clip(_module: Any, _inputs: Any, output: Any) -> Any:
            value = _first_tensor(output)
            clipped = value.masked_fill(value.detach().abs() <= threshold, 0.0)
            activations[name] = clipped
            if value is not output:
                raise TypeError("TEAL output hook requires a direct tensor output.")
            return clipped

        return clip

    def input_hook(name: str):
        threshold = _valid_threshold(thresholds[name], name)

        def clip(_module: Any, inputs: Any) -> Any:
            value = _first_tensor(inputs)
            clipped = value.masked_fill(value.detach().abs() <= threshold, 0.0)
            activations[name] = clipped
            if not isinstance(inputs, tuple) or not inputs or inputs[0] is not value:
                raise TypeError("TEAL input hook requires the first tuple element to be the tensor.")
            return (clipped, *inputs[1:])

        return clip

    try:
        for name, (module, port) in modules.items():
            handles.append(
                module.register_forward_hook(output_hook(name))
                if port == "output"
                else module.register_forward_pre_hook(input_hook(name))
            )
        yield activations
    finally:
        for handle in handles:
            handle.remove()


def _evaluate_point(
    model: Any,
    validation_tokens: Any,
    thresholds: Mapping[str, float],
    *,
    block_size: int,
    batch_size: int,
    device: Any,
    torch: Any,
    np: Any,
) -> dict[str, Any]:
    from sparsity_research.evaluation import evaluate_complete_blocks
    from sparsity_research.logical_capture import LogicalProductAccumulator, capture_logical_products
    from sparsity_research.metrics import ActivationAccumulator

    activation = ActivationAccumulator((0.0,))
    logical = LogicalProductAccumulator()
    expected_captures = len(_module_map(model))
    started = perf_counter()
    with threshold_hooks(model, thresholds) as captured:

        def consume(_output: Any, _sequences: int) -> None:
            if len(captured) != expected_captures:
                raise ValueError("TEAL clipping did not capture every matrix input.")
            activation.update(captured, torch=torch)
            captured.clear()

        with capture_logical_products(model, accumulator=logical, torch=torch):
            coverage = evaluate_complete_blocks(
                model=model,
                tokens=validation_tokens,
                block_size=block_size,
                batch_size=batch_size,
                device=device,
                torch=torch,
                np=np,
                autocast_dtype=torch.float16,
                after_batch=consume,
            )
    if any(coverage.get(key) != value for key, value in EXPECTED_VALIDATION.items()):
        raise ValueError("TEAL complete-validation coverage changed.")
    layer_rows = activation.rows()
    pooled = activation.pooled_by_site()
    if len(layer_rows) != expected_captures or {row["name"] for row in pooled} != set(SITES):
        raise ValueError("TEAL activation rows are incomplete.")
    logical_summary = logical.summary(model=model, total_input_tokens=int(coverage["input_tokens"]))
    operations = logical_summary["per_operation"].values()
    if sum(int(row["zero_product_count"]) for row in operations) != int(logical_summary["block_zero_product_count"]):
        raise ValueError("TEAL logical zero-product counts do not reconcile.")
    if sum(int(row["product_count"]) for row in logical_summary["per_operation"].values()) != int(logical_summary["block_product_count"]):
        raise ValueError("TEAL logical product denominators do not reconcile.")
    seconds = perf_counter() - started
    return {
        "validation": coverage,
        "thresholds_by_site_layer": dict(sorted(thresholds.items())),
        "activation_rows": layer_rows,
        "activations_by_site": pooled,
        "logical_products": logical_summary,
        "evaluation_seconds": seconds,
        "input_tokens_per_second": int(coverage["input_tokens"]) / seconds,
    }


def nondominated(rows: list[Mapping[str, Any]]) -> list[bool]:
    result = []
    for index, row in enumerate(rows):
        loss = float(row["validation"]["loss"])
        opportunity = float(row["logical_products"]["R_model"])
        result.append(
            not any(
                other_index != index
                and float(other["validation"]["loss"]) <= loss
                and float(other["logical_products"]["R_model"]) >= opportunity
                and (
                    float(other["validation"]["loss"]) < loss
                    or float(other["logical_products"]["R_model"]) > opportunity
                )
                for other_index, other in enumerate(rows)
            )
        )
    return result


def _source(condition_id: str) -> dict[str, Any]:
    verified = verify_attempt(condition_id)
    attempt_dir = RUN_DIR / "artifacts" / "attempts" / verified["attempt_id"]
    manifest = _json(attempt_dir / "manifest.json")
    final = manifest["checkpoints"]["final"]
    checkpoint = attempt_dir / final["path"]
    rebuilt = build_transfer_inventory(checkpoint)
    checkpoint_hash = inventory_content_sha256(rebuilt)
    if checkpoint_hash != verified["checkpoint_content_sha256"]:
        raise ValueError("TEAL source checkpoint hash differs from verified training evidence.")
    weight_path = attempt_dir / "diagnostics" / "weight_statistics.json"
    return {
        "condition_id": condition_id,
        "attempt_id": verified["attempt_id"],
        "checkpoint": checkpoint,
        "checkpoint_content_sha256": checkpoint_hash,
        "source_final_validation_loss": verified["final_validation_loss"],
        "topology": manifest["topology"],
        "weight_statistics": {
            "path": weight_path.relative_to(RUN_DIR).as_posix(),
            "sha256": _sha256(weight_path),
            "recomputed": False,
        },
    }


def _protocol_sha256(
    config: Mapping[str, Any],
    source: Mapping[str, Any],
    train_metadata: Mapping[str, Any],
    validation_metadata: Mapping[str, Any],
) -> str:
    teal = mapping(config, "teal_posthoc")
    value = {
        "code_sha256": _sha256(Path(__file__)),
        "condition_id": source["condition_id"],
        "checkpoint_content_sha256": source["checkpoint_content_sha256"],
        "sites": list(SITES),
        "target_sparsities": list(teal["target_sparsities"]),
        "calibration_blocks": int(teal["calibration_blocks"]),
        "calibration_selection": teal["calibration_selection"],
        "evaluation_batch_size": int(teal["evaluation_batch_size"]),
        "train_cache_sha256": train_metadata["tokens_sha256"],
        "validation_cache_sha256": validation_metadata["tokens_sha256"],
        "threshold_rule": "smallest empirical absolute-value order statistic; abs(x) <= threshold becomes zero",
    }
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _thresholds_for_target(calibration: Mapping[str, Any], target: float) -> dict[str, float]:
    thresholds = {}
    for name, statistics in calibration["statistics"].items():
        matches = [row for row in statistics["targets"] if math.isclose(float(row["target_sparsity"]), target)]
        if len(matches) != 1:
            raise ValueError(f"Calibration target {target} is missing for {name}.")
        thresholds[name] = float(matches[0]["threshold"])
    return thresholds


def _load_progress(path: Path, protocol: str) -> dict[float, dict[str, Any]]:
    if not path.exists():
        return {}
    result = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = json.loads(line)
        if row.get("protocol_sha256") != protocol:
            raise ValueError("Existing TEAL progress belongs to a different protocol.")
        target = float(row["target_sparsity"])
        if target in result:
            raise ValueError(f"Duplicate TEAL progress row at line {line_number}.")
        result[target] = row
    return result


def _append_progress(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _first_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            try:
                return _first_tensor(item)
            except TypeError:
                pass
    raise TypeError("Hook value did not contain a tensor.")


def _valid_threshold(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"Invalid threshold for {name}: {value}")
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _serializable_source(source: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.relative_to(RUN_DIR).as_posix() if isinstance(value, Path) else value
        for key, value in source.items()
    }
