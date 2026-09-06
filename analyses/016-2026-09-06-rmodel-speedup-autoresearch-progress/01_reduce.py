#!/usr/bin/env python3
"""Reduce Run 025 replication and search-trajectory evidence for Analysis 016."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE_ANALYSIS = ROOT / "analyses/015-2026-09-06-pythia-agentic-kernel-search"
RUN = ROOT / "runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
ARTIFACTS = RUN / "retrieved/rtxpro4500-002/artifacts"

RMODEL_OUTPUT = HERE / "rmodel-speedup.csv"
REGRESSION_OUTPUT = HERE / "rmodel-regressions.csv"
PROGRESS_OUTPUT = HERE / "candidate-progress.csv"
PROVENANCE_OUTPUT = HERE / "source-provenance.json"

SIZE_ORDER = {"14M": 0, "70M": 1, "410M": 2}
GPU_ORDER = {
    "NVIDIA RTX PRO 4500 Blackwell": 0,
    "NVIDIA H100 NVL": 1,
}

RMODEL_FIELDS = [
    "hardware_attempt",
    "gpu",
    "condition_id",
    "model_size",
    "family",
    "kappa",
    "partition",
    "implementation",
    "canonical_validation_loss",
    "measured_R_model_fraction",
    "measured_R_model_percent",
    "block_zero_product_count",
    "model_product_count",
    "n_fresh_processes",
    "n_complete_validation",
    "n_correct_processes",
    "qualified",
    "process_median_speedup",
    "process_min_speedup",
    "process_max_speedup",
    "paired_timing_samples_per_process",
    "timing_input_clusters_per_process",
    "aggregation_unit",
    "process_sources",
]

REGRESSION_FIELDS = [
    "hardware_attempt",
    "gpu",
    "model_size",
    "stratum",
    "n_checkpoints",
    "n_correctness_failures_excluded",
    "intercept",
    "slope_per_R_model_fraction",
    "slope_per_10_percentage_points",
    "pearson_r",
    "R_squared",
    "spearman_rho",
    "R_model_min",
    "R_model_max",
    "method",
]

PROGRESS_FIELDS = [
    "model_size",
    "candidate_index",
    "candidate_id",
    "policy_label",
    "status",
    "a4_high_speedup",
    "a7_high_speedup",
    "endpoint_geomean_speedup",
    "a4_samples",
    "a7_samples",
    "quality_scope",
    "source_a4",
    "source_a7",
    "note",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(ROOT.resolve()).as_posix()


class Evidence:
    def __init__(self) -> None:
        self.sources: dict[str, dict[str, Any]] = {}

    def record(self, path: Path) -> None:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(resolved)
        key = relative(resolved)
        self.sources[key] = {
            "bytes": resolved.stat().st_size,
            "sha256": sha256(resolved),
        }

    def json(self, path: Path) -> dict[str, Any]:
        self.record(path)
        with path.open(encoding="utf-8") as stream:
            return json.load(stream)

    def csv(self, path: Path) -> list[dict[str, str]]:
        self.record(path)
        with path.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream))


def write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def reduce_rmodel(evidence: Evidence) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    source_rows = evidence.csv(SOURCE_ANALYSIS / "replication-summary.csv")
    if len(source_rows) != 54:
        raise ValueError(f"Expected 54 checkpoint summaries, found {len(source_rows)}")
    groups = {(row["gpu"], row["model_size"]) for row in source_rows}
    if groups != {(gpu, size) for gpu in GPU_ORDER for size in SIZE_ORDER}:
        raise ValueError(f"Unexpected GPU/model-size groups: {groups}")
    for row in source_rows:
        if int(row["n_fresh_processes"]) not in {2, 3}:
            raise ValueError(f"Unexpected process count in {row['condition_id']}")
        if int(row["n_complete_validation"]) != int(row["n_fresh_processes"]):
            raise ValueError(f"Incomplete validation in {row['condition_id']}")
        numerator = int(row["block_zero_product_count"])
        denominator = int(row["model_product_count"])
        reported = float(row["measured_R_model_fraction"])
        if not math.isclose(numerator / denominator, reported, rel_tol=0.0, abs_tol=5e-16):
            raise ValueError(f"Integer-count R_model mismatch in {row['condition_id']}")

    condition_order = {"A0": 0, "A1-H": 1, "A4-OL1": 2, "A7-OL1": 3}
    source_rows.sort(
        key=lambda row: (
            SIZE_ORDER[row["model_size"]],
            GPU_ORDER[row["gpu"]],
            condition_order[row["family"]],
            float(row["kappa"] or -1.0),
        )
    )
    rows = [{field: row[field] for field in RMODEL_FIELDS} for row in source_rows]

    source_fits = evidence.csv(SOURCE_ANALYSIS / "replication-regressions.csv")
    primary = [row for row in source_fits if row["stratum"] == "primary-qualified"]
    if len(primary) != 6:
        raise ValueError(f"Expected six primary qualified regressions, found {len(primary)}")
    primary.sort(key=lambda row: (SIZE_ORDER[row["model_size"]], GPU_ORDER[row["gpu"]]))
    fits = [{field: row[field] for field in REGRESSION_FIELDS} for row in primary]
    write_csv(RMODEL_OUTPUT, RMODEL_FIELDS, rows)
    write_csv(REGRESSION_OUTPUT, REGRESSION_FIELDS, fits)
    return rows, fits


def artifact_path(name: str) -> Path:
    return ARTIFACTS / name


def standard_endpoint(
    evidence: Evidence,
    attempt: str,
    condition: str,
    *,
    timing_key: str = "candidate",
    quality_key: str = "candidate",
    require_full_validation: bool = False,
) -> tuple[float, int, str]:
    folder = artifact_path(attempt)
    manifest = evidence.json(folder / "manifest.json")
    timing = evidence.json(folder / "timing.json")
    quality = evidence.json(folder / "development-quality.json")
    if manifest["arguments"]["condition"] != condition:
        raise ValueError(f"Condition mismatch in {attempt}")
    if not quality["pass"][quality_key]:
        raise ValueError(f"Development quality failed in scored artifact {attempt}")
    if "matched" in timing:
        summary = timing["matched"]["eager"]["summary"][timing_key]
    else:
        summary = timing["summary"][timing_key]
    if require_full_validation:
        full = evidence.json(folder / "full-validation.json")
        if full["blocks"] != 338 or not full["pass"][quality_key]:
            raise ValueError(f"Complete validation failed in scored artifact {attempt}")
    return float(summary["paired_geomean_speedup"]), int(summary["samples"]), relative(folder)


def progress_row(
    evidence: Evidence,
    *,
    model_size: str,
    candidate_index: int,
    policy_label: str,
    a4_attempt: str,
    a7_attempt: str,
    status: str,
    note: str,
    timing_key: str = "candidate",
    quality_key: str = "candidate",
    require_full_validation: bool = False,
    quality_scope: str,
) -> dict[str, Any]:
    a4, a4_samples, source_a4 = standard_endpoint(
        evidence,
        a4_attempt,
        f"{model_size.lower()}/a4-0p5",
        timing_key=timing_key,
        quality_key=quality_key,
        require_full_validation=require_full_validation,
    )
    a7, a7_samples, source_a7 = standard_endpoint(
        evidence,
        a7_attempt,
        f"{model_size.lower()}/a7-0p5",
        timing_key=timing_key,
        quality_key=quality_key,
        require_full_validation=require_full_validation,
    )
    return {
        "model_size": model_size,
        "candidate_index": candidate_index,
        "candidate_id": "P0" if candidate_index == 0 else f"K{candidate_index:03d}",
        "policy_label": policy_label,
        "status": status,
        "a4_high_speedup": f"{a4:.16g}",
        "a7_high_speedup": f"{a7:.16g}",
        "endpoint_geomean_speedup": f"{math.sqrt(a4 * a7):.16g}",
        "a4_samples": a4_samples,
        "a7_samples": a7_samples,
        "quality_scope": quality_scope,
        "source_a4": source_a4,
        "source_a7": source_a7,
        "note": note,
    }


def status_row(
    evidence: Evidence,
    *,
    model_size: str,
    candidate_index: int,
    policy_label: str,
    status: str,
    source_names: list[str],
    note: str,
) -> dict[str, Any]:
    sources: list[str] = []
    for name in source_names:
        path = artifact_path(name)
        status_json = evidence.json(path / "status.json")
        if status == "gate_failure" and status_json["stage"] != "failed_or_incomplete":
            raise ValueError(f"Expected gate failure in {name}")
        sources.append(relative(path))
    return {
        "model_size": model_size,
        "candidate_index": candidate_index,
        "candidate_id": "P0" if candidate_index == 0 else f"K{candidate_index:03d}",
        "policy_label": policy_label,
        "status": status,
        "a4_high_speedup": "",
        "a7_high_speedup": "",
        "endpoint_geomean_speedup": "",
        "a4_samples": "",
        "a7_samples": "",
        "quality_scope": "no comparable two-endpoint score",
        "source_a4": sources[0] if sources else "",
        "source_a7": sources[1] if len(sources) > 1 else "",
        "note": note,
    }


def reduce_progress(evidence: Evidence) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    rows.extend(
        [
            progress_row(
                evidence,
                model_size="14M",
                candidate_index=0,
                policy_label="Sakana-derived P0",
                a4_attempt="p0-14m-a4-0p5-active-rtxpro4500-001",
                a7_attempt="p0-14m-a7-0p5-active-rtxpro4500-001",
                status="baseline",
                note="Initial exact signed Pythia adaptation.",
                quality_scope="16-input fixed development gate",
            ),
            progress_row(
                evidence,
                model_size="14M",
                candidate_index=1,
                policy_label="fused register compaction",
                a4_attempt="confirm-14m-a4-0p5-k001-rtxpro4500-002",
                a7_attempt="confirm-14m-a7-0p5-k001-rtxpro4500-002",
                status="qualified",
                note="High-resolution K001 confirmation.",
                quality_scope="16-input fixed development gate",
            ),
            progress_row(
                evidence,
                model_size="14M",
                candidate_index=6,
                policy_label="shape portfolio",
                a4_attempt="portfolio-14m-a4-0p5-active-rtxpro4500-001",
                a7_attempt="portfolio-14m-a7-0p5-active-rtxpro4500-001",
                status="qualified_not_selected",
                note="K006 was correct but slower than K001 on both endpoints.",
                timing_key="k006",
                quality_key="k006",
                quality_scope="16-input fixed development gate",
            ),
            progress_row(
                evidence,
                model_size="14M",
                candidate_index=12,
                policy_label="suffix-5 layer policy",
                a4_attempt="final-14m-a4-0p5-k012-rtxpro4500-002",
                a7_attempt="final-14m-a7-0p5-k012-rtxpro4500-002",
                status="rejected_full_matrix",
                note="Both plotted endpoints passed, but A0, A4 kappa=0, and A7 kappa=0 failed complete validation.",
                require_full_validation=True,
                quality_scope="both endpoints pass complete 338-block validation; policy rejected on 3/6 development conditions",
            ),
            progress_row(
                evidence,
                model_size="14M",
                candidate_index=13,
                policy_label="suffix-3 robust policy",
                a4_attempt="k013final-14m-a4-0p5-development-rtxpro4500-002",
                a7_attempt="k013final-14m-a7-0p5-development-rtxpro4500-002",
                status="frozen_final",
                note="Frozen architecture policy after complete-validation repair.",
                require_full_validation=True,
                quality_scope="complete 338-block validation",
            ),
        ]
    )

    k012_full = []
    for folder in sorted(ARTIFACTS.glob("final-14m-*-k012-rtxpro4500-002")):
        full = evidence.json(folder / "full-validation.json")
        k012_full.append(bool(full["pass"]["candidate"]))
    if len(k012_full) != 6 or sum(k012_full) != 3:
        raise ValueError(f"Expected K012 to pass 3/6 complete validations, got {sum(k012_full)}/{len(k012_full)}")

    rows.extend(
        [
            status_row(
                evidence,
                model_size="70M",
                candidate_index=0,
                policy_label="Sakana-derived P0",
                status="gate_failure",
                source_names=["p0-70m-a7-0p5-active-rtxpro4500-002"],
                note="A7 high endpoint failed the fixed correctness gate before timing.",
            ),
            status_row(
                evidence,
                model_size="70M",
                candidate_index=1,
                policy_label="fused register compaction",
                status="gate_failure",
                source_names=["confirm-70m-a7-0p5-k001-rtxpro4500-002"],
                note="A7 high endpoint failed the fixed correctness gate before timing.",
            ),
            status_row(
                evidence,
                model_size="70M",
                candidate_index=3,
                policy_label="compensated accumulation",
                status="gate_failure",
                source_names=["confirm-70m-a7-0p5-k003-rtxpro4500-002"],
                note="A7 high endpoint failed the fixed correctness gate before timing.",
            ),
            progress_row(
                evidence,
                model_size="70M",
                candidate_index=4,
                policy_label="exact tile skip",
                a4_attempt="k004-70m-a4-0p5-active-rtxpro4500-002",
                a7_attempt="k004-70m-a7-0p5-active-rtxpro4500-002",
                status="qualified",
                note="First 70M candidate with two correctness-qualified endpoint timings.",
                quality_scope="16-input fixed development gate",
            ),
            status_row(
                evidence,
                model_size="70M",
                candidate_index=6,
                policy_label="shape portfolio",
                status="gate_failure",
                source_names=["portfolio-70m-a7-0p5-active-rtxpro4500-002"],
                note="A7 high endpoint failed the fixed correctness gate before timing.",
            ),
            status_row(
                evidence,
                model_size="70M",
                candidate_index=7,
                policy_label="tile-geometry search",
                status="single_endpoint_search",
                source_names=[],
                note="K007 was intentionally evaluated on A4 high only, so no common two-endpoint score exists.",
            ),
            status_row(
                evidence,
                model_size="70M",
                candidate_index=8,
                policy_label="layer-mask search",
                status="single_endpoint_search",
                source_names=[],
                note="K008 was intentionally evaluated on A7 high only, so no common two-endpoint score exists.",
            ),
            progress_row(
                evidence,
                model_size="70M",
                candidate_index=9,
                policy_label="suffix-2 layer policy",
                a4_attempt="final-70m-a4-0p5-k009-rtxpro4500-002",
                a7_attempt="final-70m-a7-0p5-k009-rtxpro4500-002",
                status="qualified",
                note="Frozen K009 policy; both high endpoints passed complete validation.",
                require_full_validation=True,
                quality_scope="complete 338-block validation",
            ),
            progress_row(
                evidence,
                model_size="70M",
                candidate_index=14,
                policy_label="topology-mask winners",
                a4_attempt="k014full-70m-a4-0p5-suffix3-rtxpro4500-002",
                a7_attempt="k014full-70m-a7-0p5-suffix2-rtxpro4500-002",
                status="selected_stage",
                note="K014's separately selected A4 and A7 topology winners.",
                require_full_validation=True,
                quality_scope="complete 338-block validation",
            ),
            progress_row(
                evidence,
                model_size="70M",
                candidate_index=15,
                policy_label="A7 h/z isolation update",
                a4_attempt="k014full-70m-a4-0p5-suffix3-rtxpro4500-002",
                a7_attempt="k015full-70m-a7-0p5-suffix3-hz-rtxpro4500-002",
                status="selected_stage",
                note="Incumbent after K015: retain K014 A4 and replace A7 with K015 suffix3-hz.",
                require_full_validation=True,
                quality_scope="complete 338-block validation",
            ),
            progress_row(
                evidence,
                model_size="70M",
                candidate_index=16,
                policy_label="topology-dispatched final",
                a4_attempt="k016final-70m-a4-0p5-development-rtxpro4500-002",
                a7_attempt="k016final-70m-a7-0p5-development-rtxpro4500-002",
                status="frozen_final",
                note="Frozen final policy rerun at 80 samples per endpoint.",
                require_full_validation=True,
                quality_scope="complete 338-block validation",
            ),
        ]
    )

    rows.extend(
        [
            status_row(
                evidence,
                model_size="410M",
                candidate_index=0,
                policy_label="Sakana-derived P0",
                status="gate_failure",
                source_names=[
                    "p0-410m-a4-0p5-active-rtxpro4500-002",
                    "p0-410m-a7-0p5-active-rtxpro4500-002",
                ],
                note="Both high endpoints failed the fixed correctness gate before timing.",
            ),
            status_row(
                evidence,
                model_size="410M",
                candidate_index=1,
                policy_label="fused register compaction",
                status="gate_failure",
                source_names=[
                    "k001-410m-a4-0p5-active-rtxpro4500-002",
                    "k001-410m-a7-0p5-active-rtxpro4500-002",
                ],
                note="Both high endpoints failed the fixed correctness gate before timing.",
            ),
            progress_row(
                evidence,
                model_size="410M",
                candidate_index=4,
                policy_label="exact tile skip",
                a4_attempt="k004-410m-a4-0p5-active-rtxpro4500-002",
                a7_attempt="k004-410m-a7-0p5-active-rtxpro4500-002",
                status="qualified",
                note="First 410M candidate with two correctness-qualified endpoint timings.",
                quality_scope="16-input fixed development gate",
            ),
            progress_row(
                evidence,
                model_size="410M",
                candidate_index=10,
                policy_label="flagged-z layer policy",
                a4_attempt="final-410m-a4-0p5-k010-rtxpro4500-002",
                a7_attempt="final-410m-a7-0p5-k010-rtxpro4500-002",
                status="frozen_final",
                note="Frozen final policy after site/layer isolation.",
                require_full_validation=True,
                quality_scope="complete 338-block validation",
            ),
        ]
    )

    rows.sort(key=lambda row: (SIZE_ORDER[row["model_size"]], int(row["candidate_index"])))
    write_csv(PROGRESS_OUTPUT, PROGRESS_FIELDS, rows)
    return rows


def main() -> None:
    evidence = Evidence()
    rmodel_rows, fits = reduce_rmodel(evidence)
    progress = reduce_progress(evidence)
    provenance = {
        "schema_version": 1,
        "analysis": "016-2026-09-06-rmodel-speedup-autoresearch-progress",
        "source_run": relative(RUN),
        "rmodel_rows": len(rmodel_rows),
        "primary_regressions": len(fits),
        "progress_rows": len(progress),
        "sources": dict(sorted(evidence.sources.items())),
    }
    with PROVENANCE_OUTPUT.open("w", encoding="utf-8") as stream:
        json.dump(provenance, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(
        f"Wrote {len(rmodel_rows)} R_model rows, {len(fits)} fits, "
        f"and {len(progress)} candidate rows from {len(evidence.sources)} source files."
    )


if __name__ == "__main__":
    main()
