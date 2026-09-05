#!/usr/bin/env python3
"""Recover only Run-024's final condition after post-timing serialization failed.

The first five condition JSONs are treated as immutable inputs: their hashes are
verified before and after the recovery.  Only A7-OL1 kappa=0.5 is executed, with
its original condition index, and the six-condition cohort is then serialized.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time


RUN_NAME = "024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels"
TARGET_ID = "a7-ol1-kappa-0p5"
PREEXISTING_IDS = (
    "a0-gelu",
    "a1h-relu",
    "a4-ol1-kappa-0",
    "a4-ol1-kappa-0p5",
    "a7-ol1-kappa-0",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, str]:
    records: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        digest, filename = line.split(maxsplit=1)
        filename = filename.lstrip("* ")
        if filename in records:
            raise ValueError(f"Duplicate manifest entry: {filename}")
        records[filename] = digest
    expected = {f"{condition_id}.json" for condition_id in PREEXISTING_IDS}
    if set(records) != expected:
        raise ValueError(f"Preexisting manifest mismatch: {set(records)} != {expected}")
    return records


def verify_preexisting(output_dir: Path, expected: dict[str, str]) -> dict[str, str]:
    actual = {filename: sha256_file(output_dir / filename) for filename in expected}
    if actual != expected:
        raise ValueError(f"Preexisting condition results changed: {actual} != {expected}")
    return actual


def parse_utc(path: Path) -> datetime:
    value = path.read_text(encoding="utf-8").strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError(f"Timestamp lacks timezone: {path}")
    return parsed.astimezone(timezone.utc)


def iso_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_benchmark(repo_root: Path):
    run_dir = repo_root / "runs" / RUN_NAME
    sys.path.insert(0, str(run_dir))
    sys.path.insert(1, str(repo_root / "src"))
    sys.argv.append("--include-batch32")
    try:
        path = run_dir / "03_benchmark.py"
        spec = importlib.util.spec_from_file_location("_run024_recovery_benchmark", path)
        if spec is None or spec.loader is None:
            raise ImportError(path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.argv.remove("--include-batch32")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--original-control-dir", type=Path, required=True)
    parser.add_argument("--preexisting-manifest", type=Path, required=True)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir.resolve()
    original_control = args.original_control_dir.resolve()
    manifest_path = args.preexisting_manifest.resolve()
    expected_hashes = load_manifest(manifest_path)
    verify_preexisting(output_dir, expected_hashes)
    target_path = output_dir / f"{TARGET_ID}.json"
    if target_path.exists():
        raise FileExistsError(f"Recovery refuses to overwrite target: {target_path}")

    benchmark = load_benchmark(repo_root)
    base = benchmark._impl
    config = benchmark.load_resolved_config()
    if config["measurement"]["full_model_batches"] != [1, 32]:
        raise ValueError("Recovery must retain the accepted B1+B32 batch decision.")
    if not base.torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    if tuple(base.torch.cuda.get_device_capability(0)) != tuple(config["runtime"]["compute_capability"]):
        raise RuntimeError("Recovery requires Hopper compute capability 9.0.")
    if base.torch.__version__.split("+")[0] != config["runtime"]["pythia_torch"]:
        raise RuntimeError(f"PyTorch mismatch: {base.torch.__version__}")
    if base.transformers.__version__ != config["runtime"]["pythia_transformers"]:
        raise RuntimeError(f"Transformers mismatch: {base.transformers.__version__}")
    if base.np.__version__ != config["runtime"]["pythia_numpy"]:
        raise RuntimeError(f"NumPy mismatch: {base.np.__version__}")

    base.load_sparse_ops(args.upstream_dir.resolve())
    tokens = base.np.memmap(base.repo_path(config["validation"]["tokens"]), mode="r", dtype=base.np.int32)
    conditions = base.conditions_for_phase(config, "sentinel")
    if [condition["id"] for condition in conditions] != [*PREEXISTING_IDS, TARGET_ID]:
        raise ValueError("Condition order changed from the confirmed six-condition sentinel cohort.")

    recovery_started = datetime.now(timezone.utc)
    recovery_clock = time.perf_counter()
    base.torch.cuda.reset_peak_memory_stats()
    recovered = base.run_condition(
        conditions[5],
        5,
        config=config,
        tokens=tokens,
        output_dir=output_dir,
        progress_path=output_dir / "progress.json",
    )
    recovery_elapsed = time.perf_counter() - recovery_clock
    recovery_finished = datetime.now(timezone.utc)
    if recovered["condition"]["id"] != TARGET_ID or not target_path.is_file():
        raise RuntimeError("Final condition did not serialize as expected.")
    post_hashes = verify_preexisting(output_dir, expected_hashes)

    results = [json.loads((output_dir / f"{condition['id']}.json").read_text(encoding="utf-8")) for condition in conditions]
    attention_records = [
        record
        for result in results
        for record in (result["attention_compositions"] or [])
    ]
    original_started = parse_utc(original_control / "started-utc.txt")
    original_finished = parse_utc(original_control / "finished-utc.txt")
    original_elapsed = (original_finished - original_started).total_seconds()
    cohort = {
        "schema_version": 1,
        "run": config["name"],
        "phase": "sentinel",
        "condition_ids": [condition["id"] for condition in conditions],
        "condition_files": [f"{condition['id']}.json" for condition in conditions],
        "completed_conditions": len(results),
        "elapsed_seconds": original_elapsed + recovery_elapsed,
        "environment": {
            "python": platform.python_version(),
            "torch": base.torch.__version__,
            "transformers": base.transformers.__version__,
            "numpy": base.np.__version__,
            "cuda_runtime": base.torch.version.cuda,
            "compute_capability": list(base.torch.cuda.get_device_capability(0)),
            "gpu": base.gpu_snapshot(),
            "upstream_directory": str(args.upstream_dir.resolve()),
            "upstream_commit": config["upstream"]["commit"],
            "patch_sha256": config["upstream"]["patch_sha256"],
        },
        "attention_promotion": {
            "tested": bool(attention_records),
            "all_layers_break_even": bool(attention_records) and all(record["break_even"] for record in attention_records),
            "any_layer_break_even": any(record["break_even"] for record in attention_records),
            "stop_if_not_break_even": True,
            "custom_fused_causal_kernel_authorized": False,
        },
        "execution_segments": [
            {
                "id": "007-community-h100-nvl-canonical-eager-20260905-0039",
                "started_utc": iso_utc(original_started),
                "finished_utc": iso_utc(original_finished),
                "elapsed_seconds": original_elapsed,
                "retained_condition_ids": list(PREEXISTING_IDS),
                "outcome": "post-timing serialization failure on final condition",
            },
            {
                "id": "008-community-h100-nvl-cross-device-attention-coverage-recovery",
                "started_utc": iso_utc(recovery_started),
                "finished_utc": iso_utc(recovery_finished),
                "elapsed_seconds": recovery_elapsed,
                "executed_condition_ids": [TARGET_ID],
                "outcome": "completed",
            },
        ],
        "partial_result_reuse": {
            "byte_preserved": True,
            "preexisting_condition_ids": list(PREEXISTING_IDS),
            "sha256_before": expected_hashes,
            "sha256_after": post_hashes,
            "same_pod_and_physical_gpu": True,
            "target_condition_reexecuted_from_start": True,
            "measurements_spliced_within_condition": False,
        },
    }
    base.write_json(output_dir / "cohort.json", cohort)
    base.update_progress(
        output_dir / "progress.json",
        "benchmark_complete",
        phase="sentinel",
        completed_conditions=len(results),
        total_conditions=len(conditions),
        phase_etc_seconds=0,
        recovery=True,
    )
    print(f"PASS: recovered {TARGET_ID} and serialized six-condition cohort -> {output_dir}")


if __name__ == "__main__":
    main()
