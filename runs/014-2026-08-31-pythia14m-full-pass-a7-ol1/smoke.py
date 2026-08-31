"""Non-evidence probes for the exact Run 014 mixed A7 seven-site OL1 boundary."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import math
from typing import Any, Iterable

from _reuse_run004 import load_run004_module
from run_config import (
    EXPECTED_ACTIVE_SITES,
    EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256,
    EXPECTED_PRESSURE_CAPTURE_TENSOR_COUNT,
    RUN_DIR,
    load_config,
    mapping,
    require_cuda,
    site_gates,
    write_json,
)
from run014_capture import SevenSitePressureCapture


ENDPOINT_CONDITION_IDS = (
    "a7-ol1-kappa-0",
    "a7-ol1-kappa-0p5",
)
_BASE = load_run004_module("_run014_frozen_run004_smoke", "smoke.py")
_BASE.ActivationCapture = SevenSitePressureCapture


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
        raise ValueError("Target GPU name and memory must be provided.")

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
                "Repeated synthetic batches exercise both kappa endpoints through the exact "
                "mixed A7 graph, seven-site FP16 OL1 accumulation, task-only AdamW, and OL1 "
                "correction without creating scientific evidence."
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
            exact_target={"status": "unavailable", "all_conditions_fit": False},
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


def _exact_target(
    samples: list[dict[str, Any]],
    target: dict[str, Any],
    sequence_length: int,
    accumulation_steps: int,
) -> dict[str, Any]:
    if (
        int(target["sequence_length"]) != int(sequence_length)
        or int(target["gradient_accumulation_steps"]) != int(accumulation_steps)
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
        finite_fields = (
            "task_loss",
            "pressure_loss",
            "adamw_gradient_norm_post_clip",
            "pressure_to_task_ratio_raw",
            "pressure_to_task_ratio_final",
            "trust_scale",
        )
        healthy = (
            row.get("status") == "completed"
            and bool(health)
            and all(
                not bool(item.get("optimizer_step_skipped"))
                and not bool(item.get("gradient_overflow"))
                for item in health
            )
            and all(math.isfinite(float(last.get(key, math.nan))) for key in finite_fields)
            and float(last["adamw_gradient_norm_post_clip"]) <= 1.0 + 1e-5
            and last.get("ol1_correction_applied") is True
            and float(last["pressure_weight"]) == 1.0
            and float(last["pressure_to_task_ratio_final"]) <= 1.0 + 1e-9
            and int(last.get("pressure_capture_tensor_count", -1))
            == EXPECTED_PRESSURE_CAPTURE_TENSOR_COUNT
            and last.get("pressure_capture_names_sha256")
            == EXPECTED_PRESSURE_CAPTURE_NAMES_SHA256
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
            "last_boundary": last,
        }
    return {
        "status": "sampled",
        "memory_limit_gib": memory_limit / 1024**3,
        "conditions": conditions,
        "all_conditions_fit": all(row["fits_with_10pct_headroom"] for row in conditions.values()),
    }
