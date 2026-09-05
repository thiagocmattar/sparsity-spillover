#!/usr/bin/env python3
"""Reduce the matched Run-023/Run-024 sparse-kernel sentinel evidence."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import statistics


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
RUNS = {
    "70M": REPO_ROOT / "runs" / "023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels" / "results",
    "410M": REPO_ROOT / "runs" / "024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels" / "results",
}
CONDITIONS = (
    "a0-gelu",
    "a1h-relu",
    "a4-ol1-kappa-0",
    "a4-ol1-kappa-0p5",
    "a7-ol1-kappa-0",
    "a7-ol1-kappa-0p5",
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(path)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(row: dict[str, str], key: str) -> float | None:
    value = row[key]
    return None if value == "" else float(value)


def number_any(row: dict[str, str], *keys: str) -> float | None:
    for key in keys:
        if key in row:
            return number(row, key)
    raise KeyError(keys)


def median(rows: list[dict[str, str]], key: str) -> float:
    return statistics.median(float(row[key]) for row in rows)


def main() -> None:
    condition_rows: list[dict] = []
    operation_rows: list[dict] = []
    attention_rows: list[dict] = []
    by_scale: dict[str, dict[str, dict]] = {}
    identities: dict[str, str] = {}
    positive_controls: dict[str, float] = {}

    for scale, results_dir in RUNS.items():
        summaries = {row["condition_id"]: row for row in read_csv(results_dir / "sentinel-summary.csv")}
        primitives = read_csv(results_dir / "linear-primitive-summary.csv")
        attentions = read_csv(results_dir / "attention-summary.csv")
        verification = json.loads((results_dir / "raw" / "verification.json").read_text(encoding="utf-8"))
        cohort = json.loads((results_dir / "raw" / "cohort.json").read_text(encoding="utf-8"))
        identities[scale] = cohort["environment"]["gpu"]["identity"]
        positive_controls[scale] = verification["upstream_positive_control"]["twell_speedup"]
        by_scale[scale] = {}

        for condition_id in CONDITIONS:
            summary = summaries[condition_id]
            raw = json.loads((results_dir / "raw" / f"{condition_id}.json").read_text(encoding="utf-8"))
            extended = raw["kernel_covered_opportunity_with_separate_attention"]
            record = {
                "model_size": scale,
                "condition_id": condition_id,
                "source_loss": number(summary, "source_loss"),
                "R_model": number(summary, "R_model"),
                "R_covered_linear": number(summary, "R_covered_linear"),
                "R_covered_linear_plus_attention": number(summary, "R_covered_linear_plus_attention"),
                "attention_coverage_comparable_to_canonical_R_model": (
                    None if extended is None else extended.get("comparable_to_canonical_R_model", True)
                ),
                "b1_adapter_dense_speedup": raw["full_model"]["1"]["paired_speedup_vs_native"]["adapter_dense"]["median"],
                "b1_sparse_speedup": number(summary, "b1_paired_speedup"),
                "b32_adapter_dense_speedup": raw["full_model"]["32"]["paired_speedup_vs_native"]["adapter_dense"]["median"],
                "b32_sparse_speedup": number(summary, "b32_paired_speedup"),
                "linear_primitive_count": int(summary["linear_primitive_count"]),
                "linear_break_even_count": int(summary["linear_break_even_count"]),
                "linear_pack_kernel_speedup_median": number(summary, "linear_pack_kernel_speedup_median"),
                "linear_pack_kernel_speedup_max": number(summary, "linear_pack_kernel_speedup_max"),
                "linear_kernel_only_speedup_max": number(summary, "linear_kernel_only_speedup_max"),
                "attention_layer_count": int(summary["attention_layer_count"]),
                "attention_break_even_count": int(summary["attention_break_even_count"]),
                "attention_speedup_median": number_any(
                    summary,
                    "attention_speedup_median",
                    "attention_composition_speedup_median",
                ),
                "attention_speedup_max": number_any(
                    summary,
                    "attention_speedup_max",
                    "attention_composition_speedup_max",
                ),
            }
            condition_rows.append(record)
            by_scale[scale][condition_id] = record

        for condition_id in CONDITIONS:
            operations = sorted({row["operation"] for row in primitives if row["condition_id"] == condition_id})
            for operation in operations:
                group = [
                    row for row in primitives
                    if row["condition_id"] == condition_id and row["operation"] == operation
                ]
                operation_rows.append(
                    {
                        "model_size": scale,
                        "condition_id": condition_id,
                        "operation": operation,
                        "site": group[0]["site"],
                        "layers": len(group),
                        "M": int(group[0]["M"]),
                        "K": int(group[0]["K"]),
                        "N": int(group[0]["N"]),
                        "exact_zero_mass_median": median(group, "exact_zero_mass"),
                        "dense_median_ms_across_layers": median(group, "dense_median_ms"),
                        "kernel_median_ms_across_layers": median(group, "kernel_median_ms"),
                        "pack_kernel_median_ms_across_layers": median(group, "pack_kernel_median_ms"),
                        "kernel_only_speedup_median": median(group, "kernel_only_speedup"),
                        "pack_kernel_speedup_median": median(group, "pack_kernel_speedup"),
                    }
                )

        for condition_id in ("a7-ol1-kappa-0", "a7-ol1-kappa-0p5"):
            group = [row for row in attentions if row["condition_id"] == condition_id]
            attention_rows.append(
                {
                    "model_size": scale,
                    "condition_id": condition_id,
                    "layers": len(group),
                    "q_exact_zero_mass_median": median(group, "q_exact_zero_mass"),
                    "v_exact_zero_mass_median": median(group, "v_exact_zero_mass"),
                    "dense_composition_median_ms_across_layers": median(group, "composition_dense_median_ms"),
                    "sparse_composition_median_ms_across_layers": median(group, "composition_sparse_median_ms"),
                    "composition_speedup_median": median(group, "composition_paired_speedup"),
                    "composition_speedup_max": max(float(row["composition_paired_speedup"]) for row in group),
                    "break_even_count": sum(row["break_even"] == "True" for row in group),
                }
            )

    matched_rows = []
    for condition_id in CONDITIONS:
        old = by_scale["70M"][condition_id]
        new = by_scale["410M"][condition_id]
        matched_rows.append(
            {
                "condition_id": condition_id,
                "speedup_70m_b1": old["b1_sparse_speedup"],
                "speedup_410m_b1": new["b1_sparse_speedup"],
                "ratio_410m_over_70m_b1": new["b1_sparse_speedup"] / old["b1_sparse_speedup"],
                "speedup_70m_b32": old["b32_sparse_speedup"],
                "speedup_410m_b32": new["b32_sparse_speedup"],
                "ratio_410m_over_70m_b32": new["b32_sparse_speedup"] / old["b32_sparse_speedup"],
            }
        )

    write_csv(ANALYSIS_DIR / "condition-comparison.csv", condition_rows)
    write_csv(ANALYSIS_DIR / "matched-speedup-change.csv", matched_rows)
    write_csv(ANALYSIS_DIR / "operation-comparison.csv", operation_rows)
    write_csv(ANALYSIS_DIR / "attention-comparison.csv", attention_rows)
    summary = {
        "schema_version": 1,
        "same_physical_gpu": identities["70M"] == identities["410M"],
        "gpu_identities": identities,
        "official_positive_control_speedup": positive_controls,
        "full_model_break_even_count": {
            scale: {
                "batch_1": sum(by_scale[scale][condition]["b1_sparse_speedup"] > 1 for condition in CONDITIONS),
                "batch_32": sum(by_scale[scale][condition]["b32_sparse_speedup"] > 1 for condition in CONDITIONS),
            }
            for scale in RUNS
        },
        "410m_improvement_count_over_70m": {
            "batch_1": sum(row["speedup_410m_b1"] > row["speedup_70m_b1"] for row in matched_rows),
            "batch_32": sum(row["speedup_410m_b32"] > row["speedup_70m_b32"] for row in matched_rows),
        },
        "primitive_break_even_count": {
            scale: sum(row["linear_break_even_count"] for row in condition_rows if row["model_size"] == scale)
            for scale in RUNS
        },
        "primitive_count": {
            scale: sum(row["linear_primitive_count"] for row in condition_rows if row["model_size"] == scale)
            for scale in RUNS
        },
        "attention_break_even_count": {
            scale: sum(row["attention_break_even_count"] for row in condition_rows if row["model_size"] == scale)
            for scale in RUNS
        },
        "attention_layer_count": {
            scale: sum(row["attention_layer_count"] for row in condition_rows if row["model_size"] == scale)
            for scale in RUNS
        },
        "descriptive_only": True,
        "warning": "Selected endpoints differ in activation distributions and model quality; size is not independently randomized.",
    }
    (ANALYSIS_DIR / "comparison-summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("PASS: wrote matched 70M/410M sparse-kernel reductions")


if __name__ == "__main__":
    main()
