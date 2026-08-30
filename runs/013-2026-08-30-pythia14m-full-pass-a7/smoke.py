"""Non-evidence memory and numerical probes for the Run 013 A7 endpoints."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import math
from typing import Any, Iterable

from _reuse_run004 import load_run004_module
from run_config import (
    EXPECTED_ACTIVE_SITES,
    RUN_DIR,
    load_config,
    mapping,
    require_cuda,
    site_gates,
    write_json,
)


ENDPOINT_CONDITION_IDS = (
    "a7-z-post-mixed-kappa-0",
    "a7-z-post-mixed-kappa-0p5",
)
_BASE = load_run004_module("_run013_frozen_run004_smoke", "smoke.py")


def run_smoke(
    *,
    batch_sizes: Iterable[int] = (2, 4),
    sequence_length: int = 2048,
    boundaries: int = 2,
    accumulation_steps: int = 1,
    target_gpu: str = "NVIDIA A100-SXM4-80GB",
    target_memory_bytes: int = 80 * 1024**3,
) -> dict[str, Any]:
    import torch

    config = load_config()
    sizes = tuple(int(value) for value in batch_sizes)
    if not sizes or any(value <= 0 for value in sizes):
        raise ValueError("Smoke batch sizes must be positive.")
    if sequence_length <= 1 or sequence_length > int(mapping(config, "data")["sequence_length"]):
        raise ValueError("Smoke sequence length must be in [2, 2048].")
    if boundaries <= 0 or accumulation_steps <= 0:
        raise ValueError("Smoke boundary and accumulation counts must be positive.")
    if not target_gpu or target_memory_bytes <= 0:
        raise ValueError("Target GPU identity and memory must be declared.")

    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the prelaunch smoke.")
    flash_probe = getattr(torch.backends.cuda, "is_flash_attention_available", None)
    flash_available = bool(flash_probe is not None and flash_probe())
    result: dict[str, Any] = {
        "schema_version": 1,
        "kind": "non_evidence_prelaunch_smoke",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "device": {
            "name": torch.cuda.get_device_name(0),
            "total_memory_bytes": int(torch.cuda.get_device_properties(0).total_memory),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "flash_attention_available": flash_available,
        },
        "target": {
            "gpu": target_gpu,
            "memory_bytes": int(target_memory_bytes),
            "micro_batch_size": int(mapping(config, "training")["micro_batch_size"]),
            "sequence_length": int(mapping(config, "data")["sequence_length"]),
            "gradient_accumulation_steps": int(
                mapping(config, "training")["gradient_accumulation_steps"]
            ),
        },
        "probe": {
            "condition_ids": list(ENDPOINT_CONDITION_IDS),
            "batch_sizes": list(sizes),
            "sequence_length": int(sequence_length),
            "boundaries_per_sample": int(boundaries),
            "accumulation_steps": int(accumulation_steps),
            "note": (
                "Synthetic repeated batches exercise both kappa endpoints through the exact "
                "mixed A7-Z-POST graph, FP16 task-only AdamW boundary, optimizer state, and "
                "accumulation path without creating scientific evidence."
            ),
        },
        "samples": [],
    }
    if not flash_available:
        result.update(
            status="blocked_local_environment",
            blocker=(
                "This CUDA build has no flash-attention kernel; its timing and memory are not "
                "representative."
            ),
            projection={"status": "unavailable", "reason": "flash_attention_unavailable"},
            exact_target={"status": "not_sampled", "all_conditions_fit": False},
        )
        return _publish(result)

    require_cuda(torch)
    for batch_size in sizes:
        for condition_id in ENDPOINT_CONDITION_IDS:
            try:
                sample = _BASE._sample(
                    config=config,
                    condition_id=condition_id,
                    batch_size=batch_size,
                    sequence_length=sequence_length,
                    boundaries=boundaries,
                    accumulation_steps=accumulation_steps,
                    torch=torch,
                    device=device,
                )
            except torch.cuda.OutOfMemoryError as error:
                sample = {
                    "condition_id": condition_id,
                    "batch_size": batch_size,
                    "sequence_length": sequence_length,
                    "gradient_accumulation_steps": accumulation_steps,
                    "status": "cuda_out_of_memory",
                    "error": str(error),
                }
                gc.collect()
                torch.cuda.empty_cache()
            result["samples"].append(sample)
    result["projection"] = _project_target(result["samples"], result["target"])
    result["exact_target"] = _exact_target(
        result["samples"], result["target"], sequence_length, accumulation_steps
    )
    result["status"] = "completed"
    return _publish(result)


def _publish(result: dict[str, Any]) -> dict[str, Any]:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    output = RUN_DIR / "prelaunch" / f"smoke-{stamp}.json"
    result["output"] = output.relative_to(RUN_DIR).as_posix()
    write_json(output, result)
    return result


def _project_target(samples: list[dict[str, Any]], target: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        "method": "affine peak-allocation projection by batch size, separately by kappa endpoint",
        "target_headroom_fraction": 0.9,
        "conditions": {},
    }
    target_batch = int(target["micro_batch_size"])
    limit = 0.9 * float(target["memory_bytes"])
    for condition_id in ENDPOINT_CONDITION_IDS:
        pairs = sorted(
            (float(row["batch_size"]), float(row["peak_memory_allocated_bytes"]))
            for row in samples
            if row.get("condition_id") == condition_id and row.get("status") == "completed"
        )
        if len(pairs) < 2 or len({x for x, _ in pairs}) < 2:
            output["conditions"][condition_id] = {"status": "insufficient_samples"}
            continue
        count = float(len(pairs))
        mean_x = sum(x for x, _ in pairs) / count
        mean_y = sum(y for _, y in pairs) / count
        denominator = sum((x - mean_x) ** 2 for x, _ in pairs)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in pairs) / denominator
        intercept = mean_y - slope * mean_x
        projected = max(0.0, intercept + slope * target_batch)
        output["conditions"][condition_id] = {
            "status": "projected",
            "intercept_bytes": int(intercept),
            "bytes_per_sequence": int(slope),
            "target_peak_allocated_bytes": int(projected),
            "target_peak_allocated_gib": projected / 1024**3,
            "target_90pct_limit_gib": limit / 1024**3,
            "fits_with_10pct_headroom": bool(math.isfinite(projected) and projected <= limit),
            "caveat": "Projection does not replace the exact target-GPU preflight.",
        }
    return output


def _exact_target(
    samples: list[dict[str, Any]],
    target: dict[str, Any],
    sequence_length: int,
    accumulation_steps: int,
) -> dict[str, Any]:
    if (
        sequence_length != int(target["sequence_length"])
        or accumulation_steps != int(target["gradient_accumulation_steps"])
    ):
        return {"status": "not_sampled", "all_conditions_fit": False}
    rows = [
        row
        for row in samples
        if row.get("condition_id") in ENDPOINT_CONDITION_IDS
        and int(row.get("batch_size", -1)) == int(target["micro_batch_size"])
    ]
    by_condition = {str(row["condition_id"]): row for row in rows}
    if set(by_condition) != set(ENDPOINT_CONDITION_IDS):
        return {
            "status": "incomplete",
            "sampled_conditions": sorted(by_condition),
            "all_conditions_fit": False,
        }
    memory_limit = 0.9 * float(target["memory_bytes"])
    conditions = {}
    for condition_id, row in by_condition.items():
        health = row.get("boundary_health", [])
        last = row.get("last_boundary", {})
        topology = row.get("topology", {})
        expected_threshold = 0.0 if condition_id.endswith("kappa-0") else 0.5
        healthy = (
            row.get("status") == "completed"
            and bool(health)
            and all(
                not bool(item.get("optimizer_step_skipped"))
                and not bool(item.get("gradient_overflow"))
                for item in health
            )
            and math.isfinite(float(last.get("task_loss", math.nan)))
            and math.isfinite(float(last.get("adamw_gradient_norm_post_clip", math.nan)))
            and float(last["adamw_gradient_norm_post_clip"]) <= 1.0 + 1e-5
            and "pressure_loss" not in last
            and topology.get("topology_id") == "A7-Z-POST"
            and topology.get("active_sites") == list(EXPECTED_ACTIVE_SITES)
            and topology.get("site_gate") is None
            and topology.get("site_gates") == site_gates(expected_threshold)
        )
        reserved = float(row.get("peak_memory_reserved_bytes", math.inf))
        fits = bool(healthy and reserved <= memory_limit)
        conditions[condition_id] = {
            "status": row.get("status"),
            "peak_memory_reserved_gib": (
                None if not math.isfinite(reserved) else reserved / 1024**3
            ),
            "finite_nonoverflowing_boundaries": healthy,
            "fits_with_10pct_headroom": fits,
            "boundary_seconds": row.get("boundary_seconds", []),
        }
    return {
        "status": "sampled",
        "memory_limit_gib": memory_limit / 1024**3,
        "conditions": conditions,
        "all_conditions_fit": all(row["fits_with_10pct_headroom"] for row in conditions.values()),
    }
