#!/usr/bin/env python3
"""Create compact review tables from a verified Run-024 attempt."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
from pathlib import Path


RAW_AUDIT_FILES = (
    "verification.json",
    "cohort.json",
    "remote-preflight.json",
    "static-preflight.json",
    "static-preflight-official.json",
    "upstream-positive-control.csv",
    "upstream-model-revision.txt",
    "upstream-model-sha256.txt",
    "applied-sakana-pythia410.patch.sha256",
    "benchmark-harness-sha256.txt",
    "batch-calibration.json",
    "batch-decision.json",
    "pythia-pip-freeze.txt",
    "upstream-pip-freeze.txt",
    "nvidia-smi-q.txt",
)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_bytes(source: Path, destination: Path) -> None:
    destination.write_bytes(source.read_bytes())


def resolved(path: Path) -> Path:
    value = path.resolve()
    if os.name == "nt" and not str(value).startswith("\\\\?\\"):
        return Path("\\\\?\\" + str(value))
    return value


def pearson(xs: list[float], ys: list[float]) -> float:
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    denominator = math.sqrt(sum((x - x_mean) ** 2 for x in xs) * sum((y - y_mean) ** 2 for y in ys))
    return numerator / denominator


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--attempt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    attempt = resolved(args.attempt_dir)
    output = resolved(args.output_dir)
    raw = output / "raw"
    output.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    verification = json.loads((attempt / "verification.json").read_text(encoding="utf-8"))
    if verification.get("passed") is not True or verification.get("phase") != "sentinel":
        raise ValueError("Input is not a passed Run-024 sentinel artifact.")
    batches = ["1"] + (["32"] if verification["batch_decision"]["include_batch32"] else [])
    summary_rows = []
    primitive_rows = []
    attention_rows = []
    for verified in verification["conditions"]:
        condition_id = verified["condition_id"]
        result_path = attempt / f"{condition_id}.json"
        result = json.loads(result_path.read_text(encoding="utf-8"))
        primitives = result["linear_primitives"]
        primitive_speedups = [
            row["timings"]["paired_speedup_vs_dense"]["pack_plus_kernel_plus_bias"]["median"]
            for row in primitives
        ]
        kernel_speedups = [
            row["timings"]["paired_speedup_vs_dense"]["kernel_only_plus_bias"]["median"]
            for row in primitives
        ]
        attention = result["attention_compositions"] or []
        attention_speedups = [row["composition_paired_speedup"] for row in attention]
        row = {
            "condition_id": condition_id,
            "source_loss": verified["source_loss"],
            "dense_bf16_loss": verified["dense_bf16_loss"],
            "sparse_bf16_loss": verified["sparse_bf16_loss"],
            "sparse_dense_abs_loss_delta": abs(verified["sparse_bf16_loss"] - verified["dense_bf16_loss"]),
            "R_model": verified["R_model"],
            "R_covered_linear": verified["R_covered"],
            "R_covered_linear_plus_attention": verified["R_covered_with_separate_attention"],
            "linear_primitive_count": len(primitives),
            "linear_break_even_count": sum(value > 1 for value in primitive_speedups),
            "linear_pack_kernel_speedup_median": statistics.median(primitive_speedups),
            "linear_pack_kernel_speedup_max": max(primitive_speedups),
            "linear_kernel_only_speedup_max": max(kernel_speedups),
            "attention_layer_count": len(attention),
            "attention_break_even_count": sum(value > 1 for value in attention_speedups),
            "attention_speedup_median": statistics.median(attention_speedups) if attention else None,
            "attention_speedup_max": max(attention_speedups) if attention else None,
            "peak_allocated_bytes": result["memory"]["peak_allocated_bytes"],
            "peak_reserved_bytes": result["memory"]["peak_reserved_bytes"],
        }
        for batch in batches:
            timing = result["full_model"][batch]
            paired = timing["paired_speedup_vs_native"]["sparse_linear"]
            row.update(
                {
                    f"b{batch}_native_median_ms": timing["native_dense"]["median_ms"],
                    f"b{batch}_sparse_median_ms": timing["sparse_linear"]["median_ms"],
                    f"b{batch}_paired_speedup": paired["median"],
                    f"b{batch}_speedup_p10": paired["p10"],
                    f"b{batch}_speedup_p90": paired["p90"],
                    f"b{batch}_robust_break_even": paired["median"] > 1 and paired["p10"] > 1,
                }
            )
        summary_rows.append(row)

        for primitive in primitives:
            timings = primitive["timings"]
            primitive_rows.append(
                {
                    "condition_id": condition_id,
                    "layer": primitive["layer"],
                    "operation": primitive["operation"],
                    "site": primitive["site"],
                    **primitive["shape"],
                    "exact_zero_mass": primitive["exact_zero_count"] / primitive["elements"],
                    "dense_median_ms": timings["dense_bf16"]["median_ms"],
                    "pack_median_ms": timings["pack_only"]["median_ms"],
                    "kernel_median_ms": timings["kernel_only_plus_bias"]["median_ms"],
                    "pack_kernel_median_ms": timings["pack_plus_kernel_plus_bias"]["median_ms"],
                    "kernel_only_speedup": timings["paired_speedup_vs_dense"]["kernel_only_plus_bias"]["median"],
                    "pack_kernel_speedup": timings["paired_speedup_vs_dense"]["pack_plus_kernel_plus_bias"]["median"],
                    "sparse_vs_fp32_relative_l2": primitive["errors"]["sparse_vs_fp32"]["relative_l2"],
                }
            )
        for attention_row in attention:
            composition = attention_row["composition_timings"]
            attention_rows.append(
                {
                    "condition_id": condition_id,
                    "layer": attention_row["layer"],
                    "q_exact_zero_mass": attention_row["qk_q_left_component"]["exact_zero_count"] / attention_row["qk_q_left_component"]["elements"],
                    "v_exact_zero_mass": attention_row["pv_vt_left_component"]["exact_zero_count"] / attention_row["pv_vt_left_component"]["elements"],
                    "composition_dense_median_ms": composition["dense_bf16"]["median_ms"],
                    "composition_sparse_median_ms": composition["sparse_q_and_v"]["median_ms"],
                    "composition_paired_speedup": attention_row["composition_paired_speedup"],
                    "break_even": attention_row["break_even"],
                }
            )
        copy_bytes(result_path, raw / result_path.name)

    write_csv(output / "sentinel-summary.csv", summary_rows)
    write_csv(output / "linear-primitive-summary.csv", primitive_rows)
    write_csv(output / "attention-summary.csv", attention_rows)
    associations = {
        "schema_version": 1,
        "n_conditions": len(summary_rows),
        "descriptive_only": True,
        "warning": "Six selected, topology-confounded sentinels are not an inferential regression sample.",
        "pearson": {
            "R_model_vs_b1_speedup": pearson([row["R_model"] for row in summary_rows], [row["b1_paired_speedup"] for row in summary_rows]),
            "R_covered_linear_vs_b1_speedup": pearson([row["R_covered_linear"] for row in summary_rows], [row["b1_paired_speedup"] for row in summary_rows]),
        },
    }
    (output / "association-summary.json").write_text(json.dumps(associations, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for name in RAW_AUDIT_FILES:
        copy_bytes(attempt / name, raw / name)
    copied = sorted(path for path in raw.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt")
    with (raw / "SHA256SUMS.txt").open("w", encoding="utf-8", newline="\n") as handle:
        for path in copied:
            handle.write(f"{sha256(path)}  {path.name}\n")
    print(f"Wrote {len(summary_rows)} conditions, {len(primitive_rows)} primitives, and {len(attention_rows)} attention rows to {output}")


if __name__ == "__main__":
    main()

