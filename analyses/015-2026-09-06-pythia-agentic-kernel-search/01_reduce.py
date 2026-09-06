#!/usr/bin/env python3
"""Reduce Run 025's frozen full-model measurements into auditable tables."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "runs" / "025-2026-09-05-pythia-agentic-sparse-kernel-search"
RETRIEVED = RUN / "retrieved" / "rtxpro4500-002"
ARTIFACTS = RETRIEVED / "artifacts"

EXPECTED_CONDITIONS = {
    f"{size}/{condition}"
    for size in ("14m", "70m", "410m")
    for condition in (
        "a0",
        "a1h",
        "a4-0",
        "a4-0p01",
        "a4-0p05",
        "a4-0p1",
        "a4-0p5",
        "a7-0",
        "a7-0p01",
        "a7-0p05",
        "a7-0p1",
        "a7-0p5",
    )
}

FINAL_GLOBS = (
    "k013final-14m-*",
    "k016final-70m-*",
    "final-410m-*-k010-*",
    "k010final-410m-*",
)

FAMILY_ORDER = {"A0": 0, "A1-H": 1, "A4-OL1": 2, "A7-OL1": 3}
SIZE_ORDER = {"14M": 0, "70M": 1, "410M": 2}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extended(path: Path) -> Path:
    """Use the Windows extended-path prefix for deeply nested extracted files."""
    absolute = str(path.resolve())
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        return Path("\\\\?\\" + absolute)
    return Path(absolute)


def relative(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def read_timing_samples(path: Path) -> tuple[list[dict], dict[int, float]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    paired: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (int(row["repeat"]), int(row["input_index"]))
        mode = row["mode"]
        if mode in paired[key]:
            raise ValueError(f"Duplicate timing mode {mode} for {key}: {path}")
        paired[key][mode] = float(row["host_ms"])
    if any(set(values) != {"native", "candidate"} for values in paired.values()):
        raise ValueError(f"Incomplete paired timing rows: {path}")
    pair_rows = []
    logs_by_input: dict[int, list[float]] = defaultdict(list)
    for (repeat, input_index), values in sorted(paired.items()):
        ratio = values["native"] / values["candidate"]
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"Invalid speedup ratio: {path}")
        pair_rows.append(
            {
                "repeat": repeat,
                "input_index": input_index,
                "native_host_ms": values["native"],
                "candidate_host_ms": values["candidate"],
                "speedup": ratio,
            }
        )
        logs_by_input[input_index].append(math.log(ratio))
    input_log_means = {
        input_index: statistics.fmean(values)
        for input_index, values in logs_by_input.items()
    }
    return pair_rows, input_log_means


def cluster_ci(input_log_means: dict[int, float], seed_text: str) -> tuple[float, float]:
    values = np.asarray(list(input_log_means.values()), dtype=np.float64)
    if values.size < 2:
        raise ValueError("At least two timing-input clusters are required")
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20_000, values.size), replace=True).mean(axis=1)
    low, high = np.exp(np.quantile(draws, [0.025, 0.975]))
    return float(low), float(high)


def extract_quality(directory: Path) -> tuple[dict, str]:
    full = directory / "full-validation.json"
    if full.is_file():
        quality = load_json(full)
        for field, expected in (
            ("blocks", 338),
            ("documents", 500),
            ("input_tokens", 692_224),
            ("excluded_tail_tokens", 1_444),
        ):
            if quality[field] != expected:
                raise ValueError(f"Invalid complete-validation {field}: {full}")
        if not quality["complete"]:
            raise ValueError(f"Validation not marked complete: {full}")
        return quality, "complete-validation"
    development = directory / "development-quality.json"
    if not development.is_file():
        raise FileNotFoundError(f"No quality artifact in {directory}")
    return load_json(development), "development-16"


def load_measurement(directory: Path) -> dict:
    manifest_path = directory / "manifest.json"
    timing_path = directory / "timing.json"
    sample_path = directory / "timing-samples.jsonl"
    manifest = load_json(manifest_path)
    timing = load_json(timing_path)["matched"]["eager"]["summary"]
    quality, quality_scope = extract_quality(directory)
    pairs, input_log_means = read_timing_samples(sample_path)

    recomputed = math.exp(statistics.fmean(math.log(row["speedup"]) for row in pairs))
    recorded = float(timing["candidate"]["paired_geomean_speedup"])
    if not math.isclose(recomputed, recorded, rel_tol=0, abs_tol=1e-12):
        raise ValueError(f"Speedup reduction mismatch: {directory}")
    if len(pairs) != int(timing["candidate"]["samples"]):
        raise ValueError(f"Timing sample-count mismatch: {directory}")
    if not math.isclose(
        statistics.median(row["candidate_host_ms"] for row in pairs),
        float(timing["candidate"]["median_host_ms"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Candidate median mismatch: {directory}")
    if not math.isclose(
        statistics.median(row["native_host_ms"] for row in pairs),
        float(timing["native"]["median_host_ms"]),
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise ValueError(f"Native median mismatch: {directory}")

    checkpoint = manifest["checkpoint"]
    canonical = checkpoint["canonical_logical_products"]
    measured = canonical["measured"]
    coverage = canonical["coverage"]
    if not math.isclose(
        measured["block_zero_product_count"] / measured["model_product_count"],
        measured["R_model"],
        rel_tol=0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"Canonical R_model integer-count mismatch: {directory}")
    if coverage["sequences"] != 338 or not coverage["complete_block_coverage"]:
        raise ValueError(f"Canonical logical-product coverage is incomplete: {directory}")

    loss = quality["loss"]
    loss_delta = quality["loss_delta"]
    passed = quality["pass"]
    if isinstance(loss, dict):
        candidate_loss = float(loss["candidate"])
        native_loss = float(loss["native"])
        candidate_delta = float(loss_delta["candidate"])
        candidate_pass = bool(passed["candidate"])
    else:
        raise ValueError(f"Unexpected quality schema: {directory}")

    failed_gates = [row for row in quality.get("gates", {}).get("candidate", []) if not row["pass"]]
    low, high = cluster_ci(input_log_means, directory.name)
    partition = manifest.get("partition") or checkpoint.get("partition") or "development"
    size = str(checkpoint["size"]).upper().replace("PYTHIA-", "")
    family = checkpoint["family"]
    kappa = checkpoint.get("kappa")

    record = {
        "model_size": size,
        "condition_id": manifest["arguments"]["condition"],
        "family": family,
        "kappa": "" if kappa is None else float(kappa),
        "partition": partition,
        "implementation": manifest["arguments"]["implementation"],
        "quality_scope": quality_scope,
        "correctness_pass": candidate_pass,
        "failed_validation_blocks": len(failed_gates),
        "max_abs_logit_error": max((float(row["max_abs"]) for row in failed_gates), default=0.0),
        "max_relative_l2": max((float(row["relative_l2"]) for row in failed_gates), default=0.0),
        "canonical_validation_loss": float(coverage["loss"]),
        "run025_native_validation_loss": native_loss,
        "run025_candidate_validation_loss": candidate_loss,
        "run025_validation_loss_delta": candidate_delta,
        "R_model_fraction": float(measured["R_model"]),
        "R_model_percent": 100.0 * float(measured["R_model"]),
        "block_zero_product_count": int(measured["block_zero_product_count"]),
        "model_product_count": int(measured["model_product_count"]),
        "candidate_median_host_ms": float(timing["candidate"]["median_host_ms"]),
        "native_median_host_ms": float(timing["native"]["median_host_ms"]),
        "paired_geomean_speedup": recorded,
        "speedup_input_cluster_ci95_low": low,
        "speedup_input_cluster_ci95_high": high,
        "paired_timing_samples": len(pairs),
        "timing_input_clusters": len(input_log_means),
        "timing_passes": int(manifest["arguments"]["passes"]),
        "timing_data": "first 16 fixed seed-2500 training-cache blocks",
        "attention_qk_pv_execution": manifest.get("attention", "not recorded"),
        "selected_linear_sites": ";".join(manifest.get("selected_linear_sites", [])),
        "source": relative(directory),
        "manifest_sha256": sha256(manifest_path),
        "timing_sha256": sha256(timing_path),
        "timing_samples_sha256": sha256(sample_path),
        "quality_sha256": sha256(directory / ("full-validation.json" if quality_scope == "complete-validation" else "development-quality.json")),
        "_input_log_means": input_log_means,
        "_input_manifest_sha256": manifest["input_manifest"]["sha256"],
        "_development_sha256": manifest["development"]["sha256"],
    }
    return record


def collect_final() -> list[dict]:
    directories: dict[str, Path] = {}
    for pattern in FINAL_GLOBS:
        for directory in ARTIFACTS.glob(pattern):
            measurement = load_measurement(directory)
            condition = measurement["condition_id"]
            if condition in directories:
                raise ValueError(f"Duplicate final condition {condition}")
            directories[condition] = directory
    if set(directories) != EXPECTED_CONDITIONS:
        missing = sorted(EXPECTED_CONDITIONS - set(directories))
        extra = sorted(set(directories) - EXPECTED_CONDITIONS)
        raise ValueError(f"Final matrix mismatch; missing={missing}, extra={extra}")
    rows = [load_measurement(directories[condition]) for condition in sorted(directories)]
    rows.sort(key=lambda row: (
        SIZE_ORDER[row["model_size"]],
        FAMILY_ORDER[row["family"]],
        -1.0 if row["kappa"] == "" else float(row["kappa"]),
    ))
    return rows


def rank_average(values: list[float]) -> np.ndarray:
    values_array = np.asarray(values, dtype=np.float64)
    order = np.argsort(values_array, kind="mergesort")
    ranks = np.empty(len(values_array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values_array[order[end]] == values_array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def fit_rows(rows: list[dict], size: str, stratum: str) -> dict:
    x = np.asarray([float(row["R_model_fraction"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["paired_geomean_speedup"]) for row in rows], dtype=np.float64)
    if len(rows) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError(f"Invalid regression group {size}/{stratum}")
    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * x
    total = float(np.square(y - y.mean()).sum())
    residual = float(np.square(y - prediction).sum())
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rank_average(x.tolist()), rank_average(y.tolist()))[0, 1])
    loo_slopes = []
    loo_r2 = []
    if len(rows) >= 4:
        for omitted in range(len(rows)):
            keep = np.arange(len(rows)) != omitted
            sub_slope, sub_intercept = np.polyfit(x[keep], y[keep], 1)
            sub_prediction = sub_intercept + sub_slope * x[keep]
            sub_total = float(np.square(y[keep] - y[keep].mean()).sum())
            sub_residual = float(np.square(y[keep] - sub_prediction).sum())
            loo_slopes.append(float(sub_slope))
            loo_r2.append(1.0 - sub_residual / sub_total)
    return {
        "model_size": size,
        "stratum": stratum,
        "n_checkpoints": len(rows),
        "n_correctness_failures_excluded": 0,
        "intercept": float(intercept),
        "slope_per_R_model_fraction": float(slope),
        "slope_per_10_percentage_points": float(slope * 0.1),
        "pearson_r": pearson,
        "R_squared": 1.0 - residual / total,
        "spearman_rho": spearman,
        "R_model_min": float(x.min()),
        "R_model_max": float(x.max()),
        "loo_slope_min": min(loo_slopes) if loo_slopes else "",
        "loo_slope_max": max(loo_slopes) if loo_slopes else "",
        "loo_R_squared_min": min(loo_r2) if loo_r2 else "",
        "loo_R_squared_max": max(loo_r2) if loo_r2 else "",
        "method": "descriptive unweighted OLS with intercept; one checkpoint per row",
    }


def collect_regressions(final_rows: list[dict]) -> list[dict]:
    fits = []
    for size in SIZE_ORDER:
        size_rows = [row for row in final_rows if row["model_size"] == size]
        qualified = [row for row in size_rows if row["correctness_pass"]]
        groups = {
            "all-final-including-failures": size_rows,
            "qualified-final": qualified,
            "development-qualified": [row for row in qualified if row["partition"] == "development"],
            "transfer-qualified": [row for row in qualified if row["partition"] != "development"],
            "A4-qualified": [row for row in qualified if row["family"] == "A4-OL1"],
            "A7-qualified": [row for row in qualified if row["family"] == "A7-OL1"],
        }
        for stratum, rows in groups.items():
            if len(rows) < 3 or len({row["R_model_fraction"] for row in rows}) < 2:
                continue
            fit = fit_rows(rows, size, stratum)
            if stratum == "qualified-final":
                fit["n_correctness_failures_excluded"] = len(size_rows) - len(qualified)
            fits.append(fit)
    return fits


def comparable_pair(
    group: str,
    condition: str,
    baseline_name: str,
    optimized_name: str,
    evidence_scope: str,
    mechanism: str,
) -> dict:
    baseline = load_measurement(ARTIFACTS / baseline_name)
    optimized = load_measurement(ARTIFACTS / optimized_name)
    if baseline["condition_id"] != optimized["condition_id"] or baseline["condition_id"] != condition:
        raise ValueError(f"Condition mismatch in {group}/{condition}")
    if baseline["R_model_fraction"] != optimized["R_model_fraction"]:
        raise ValueError(f"R_model changed in {group}/{condition}")
    for key in ("_input_manifest_sha256", "_development_sha256"):
        if baseline[key] != optimized[key]:
            raise ValueError(f"Input identity changed in {group}/{condition}: {key}")
    common_inputs = sorted(set(baseline["_input_log_means"]) & set(optimized["_input_log_means"]))
    if len(common_inputs) != 16:
        raise ValueError(f"Expected 16 common timing inputs in {group}/{condition}")
    deltas = np.asarray(
        [optimized["_input_log_means"][index] - baseline["_input_log_means"][index] for index in common_inputs],
        dtype=np.float64,
    )
    seed = int(hashlib.sha256(f"{group}/{condition}".encode()).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    draws = np.exp(rng.choice(deltas, size=(20_000, len(deltas)), replace=True).mean(axis=1))
    low, high = np.quantile(draws, [0.025, 0.975])
    ratio = optimized["paired_geomean_speedup"] / baseline["paired_geomean_speedup"]
    return {
        "comparison_group": group,
        "model_size": optimized["model_size"],
        "condition_id": condition,
        "family": optimized["family"],
        "kappa": optimized["kappa"],
        "R_model_fraction": optimized["R_model_fraction"],
        "R_model_percent": optimized["R_model_percent"],
        "baseline_implementation": baseline["implementation"],
        "optimized_implementation": optimized["implementation"],
        "baseline_speedup": baseline["paired_geomean_speedup"],
        "optimized_speedup": optimized["paired_geomean_speedup"],
        "speedup_delta": optimized["paired_geomean_speedup"] - baseline["paired_geomean_speedup"],
        "speedup_ratio_optimized_over_baseline": ratio,
        "speedup_ratio_input_cluster_ci95_low": float(low),
        "speedup_ratio_input_cluster_ci95_high": float(high),
        "baseline_correctness_pass": baseline["correctness_pass"],
        "optimized_correctness_pass": optimized["correctness_pass"],
        "both_correct": baseline["correctness_pass"] and optimized["correctness_pass"],
        "evidence_scope": evidence_scope,
        "mechanism": mechanism,
        "baseline_source": baseline["source"],
        "optimized_source": optimized["source"],
    }


def collect_optimization_pairs() -> list[dict]:
    pairs = []
    early_14m = {
        "14m/a0": ("p0-14m-a0-all-rtxpro4500-001", "k001-14m-a0-all-rtxpro4500-001"),
        "14m/a1h": ("p0-14m-a1h-h-rtxpro4500-001", "k001-14m-a1h-h-rtxpro4500-001"),
        "14m/a4-0p5": ("p0-14m-a4-0p5-active-rtxpro4500-001", "k001-14m-a4-0p5-active-rtxpro4500-001"),
        "14m/a7-0p5": ("p0-14m-a7-0p5-active-rtxpro4500-001", "k001-14m-a7-0p5-active-rtxpro4500-001"),
    }
    for condition, names in early_14m.items():
        pairs.append(comparable_pair(
            "14M exploratory P0-to-K001", condition, *names,
            "16 development blocks; three passes; development gates only",
            "replace minimal Sakana-derived P0 packing with fused signed-exact warp compaction",
        ))

    final_14m = {
        "14m/a0": ("final-14m-a0-k012-rtxpro4500-002", "k013final-14m-a0-development-rtxpro4500-002"),
        "14m/a1h": ("final-14m-a1h-k012-rtxpro4500-002", "k013final-14m-a1h-development-rtxpro4500-002"),
        "14m/a4-0": ("final-14m-a4-0-k012-rtxpro4500-002", "k013final-14m-a4-0-development-rtxpro4500-002"),
        "14m/a4-0p5": ("final-14m-a4-0p5-k012-rtxpro4500-002", "k013final-14m-a4-0p5-development-rtxpro4500-002"),
        "14m/a7-0": ("final-14m-a7-0-k012-rtxpro4500-002", "k013final-14m-a7-0-development-rtxpro4500-002"),
        "14m/a7-0p5": ("final-14m-a7-0p5-k012-rtxpro4500-002", "k013final-14m-a7-0p5-development-rtxpro4500-002"),
    }
    for condition, names in final_14m.items():
        pairs.append(comparable_pair(
            "14M final-policy repair K012-to-K013", condition, *names,
            "same 16 timing blocks and complete 338-block validation",
            "replace the faster but partially invalid predecessor with frozen suffix-three policy",
        ))

    k013_by_condition = {
        load_json(path / "manifest.json")["arguments"]["condition"]: path.name
        for path in ARTIFACTS.glob("k013final-14m-*-development-*")
    }
    if set(k013_by_condition) != set(final_14m):
        raise ValueError("Unexpected K013 development condition inventory")

    k016_by_condition = {
        load_json(path / "manifest.json")["arguments"]["condition"]: path.name
        for path in ARTIFACTS.glob("k016final-70m-*")
    }
    k009_by_condition = {
        load_json(path / "manifest.json")["arguments"]["condition"]: path.name
        for path in ARTIFACTS.glob("final-70m-*-k009-*")
    }
    if set(k016_by_condition) != set(k009_by_condition) or len(k016_by_condition) != 12:
        raise ValueError("70M K009/K016 condition inventory mismatch")
    for condition in sorted(k016_by_condition):
        pairs.append(comparable_pair(
            "70M complete-validation K009-to-K016",
            condition,
            k009_by_condition[condition],
            k016_by_condition[condition],
            "same 16 timing blocks; both implementations evaluated over complete validation",
            "topology-specific native fallback plus layer/site-specific sparse dispatch",
        ))

    final_410m = {
        "410m/a0": ("k004-410m-a0-all-rtxpro4500-002", "k010-410m-a0-rtxpro4500-002", "native fallback for A0"),
        "410m/a4-0p5": ("k004flagz-410m-a4-0p5-rtxpro4500-002", "k010-410m-a4-0p5-rtxpro4500-002", "restrict flagged-z sparse projection to layers 2 through 20"),
        "410m/a7-0p5": ("k004flagz-410m-a7-0p5-rtxpro4500-002", "k010-410m-a7-0p5-rtxpro4500-002", "restrict flagged-z sparse projection to layers 2 through 20"),
    }
    for condition, (baseline, optimized, mechanism) in final_410m.items():
        pairs.append(comparable_pair(
            "410M development K004-to-K010", condition, baseline, optimized,
            "16 development blocks; five passes; development gates only",
            mechanism,
        ))
    return pairs


def verify_retrieval() -> dict:
    postpackage = load_json(RETRIEVED / "run025-rtxpro4500-002-postpackage.json")
    archive = RETRIEVED / "run025-rtxpro4500-002-evidence.tar.gz"
    archive_hash = sha256(archive)
    if archive.stat().st_size != postpackage["archive"]["bytes"] or archive_hash != postpackage["archive"]["sha256"]:
        raise ValueError("Local archive differs from stable remote post-package identity")

    inventory_path = ARTIFACTS / "rtxpro4500-002-evidence-inventory.json"
    inventory = load_json(inventory_path)
    mismatches = []
    checked = 0
    for item in inventory["files"]:
        path = extended(RETRIEVED / Path(item["path"]))
        if not path.is_file():
            mismatches.append({"path": item["path"], "reason": "missing"})
            continue
        checked += 1
        actual_bytes = path.stat().st_size
        actual_hash = sha256(path)
        if actual_bytes != item["bytes"] or actual_hash != item["sha256"]:
            mismatches.append({
                "path": item["path"],
                "reason": "hash-or-size",
                "inventory_bytes": item["bytes"],
                "archive_bytes": actual_bytes,
                "inventory_sha256": item["sha256"],
                "archive_sha256": actual_hash,
            })
    expected_log = "artifacts/phase13-package-evidence-rtxpro4500-002.launch.log"
    if mismatches != [{
        "path": expected_log,
        "reason": "hash-or-size",
        "inventory_bytes": 0,
        "archive_bytes": 46,
        "inventory_sha256": hashlib.sha256(b"").hexdigest(),
        "archive_sha256": sha256(extended(RETRIEVED / Path(expected_log))),
    }]:
        raise ValueError(f"Unexpected retrieval mismatch set: {mismatches}")
    return {
        "passed": True,
        "archive": {
            "path": relative(archive),
            "bytes": archive.stat().st_size,
            "sha256": archive_hash,
            "remote_postpackage_identity": relative(RETRIEVED / "run025-rtxpro4500-002-postpackage.json"),
        },
        "inventory": {
            "path": relative(inventory_path),
            "listed_files": inventory["file_count"],
            "checked_files": checked,
            "missing_files": 0,
            "matching_files": checked - len(mismatches),
        },
        "known_controller_log_race": mismatches[0],
        "interpretation": (
            "All 4,100 stable inventoried files match. The only mismatch is the Phase 13 "
            "launch log: it was empty when self-inventoried and contained its 46-byte terminal "
            "message when the immutable archive was written. The archive itself matches the "
            "stable remote post-package SHA-256."
        ),
    }


def public_row(row: dict) -> dict:
    return {key: value for key, value in row.items() if not key.startswith("_")}


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt_kappa(value: object) -> str:
    return "-" if value == "" else f"{float(value):g}"


def make_tables(final_rows: list[dict], fits: list[dict], optimization: list[dict]) -> str:
    lines = [
        "# Run 025 frozen-kernel results",
        "",
        "All speedups are paired geometric means of native/candidate host-time ratios on the first 16 fixed seed-2500 training-cache blocks. The 95% intervals resample input identities, not fresh processes. Canonical loss and `R_model` come from the source checkpoint's complete 338-block logical pass; Run 025 separately repeated complete candidate/native loss and elementwise-logit checks.",
        "",
    ]
    for size in SIZE_ORDER:
        lines.extend([
            f"## Pythia-{size}",
            "",
            "| Family | kappa | partition | canonical loss | R_model | speedup (95% input-cluster CI) | validation delta | status |",
            "| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |",
        ])
        for row in final_rows:
            if row["model_size"] != size:
                continue
            status = "pass" if row["correctness_pass"] else f"FAIL ({row['failed_validation_blocks']} blocks)"
            lines.append(
                f"| {row['family']} | {fmt_kappa(row['kappa'])} | {row['partition']} | "
                f"{row['canonical_validation_loss']:.6f} | {row['R_model_percent']:.4f}% | "
                f"{row['paired_geomean_speedup']:.4f}x "
                f"[{row['speedup_input_cluster_ci95_low']:.4f}, {row['speedup_input_cluster_ci95_high']:.4f}] | "
                f"{row['run025_validation_loss_delta']:+.6f} | {status} |"
            )
        lines.append("")

    lines.extend([
        "## Descriptive associations",
        "",
        "| Size | stratum | n | excluded failures | slope per +10 pp R_model | Pearson r | R2 | Spearman rho |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for fit in fits:
        lines.append(
            f"| {fit['model_size']} | {fit['stratum']} | {fit['n_checkpoints']} | "
            f"{fit['n_correctness_failures_excluded']} | {fit['slope_per_10_percentage_points']:+.4f}x | "
            f"{fit['pearson_r']:+.4f} | {fit['R_squared']:.4f} | {fit['spearman_rho']:+.4f} |"
        )
    lines.extend([
        "",
        "These fits are descriptive across one-seed trained checkpoints. They are not scaling laws; repeated timings are not training replicates. The figure uses only the `qualified-final` fits.",
        "",
        "## Same-checkpoint search transitions",
        "",
        "| Group | condition | R_model | baseline | optimized | ratio | both correct |",
        "| --- | --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in optimization:
        lines.append(
            f"| {row['comparison_group']} | {row['condition_id']} | {row['R_model_percent']:.4f}% | "
            f"{row['baseline_speedup']:.4f}x | {row['optimized_speedup']:.4f}x | "
            f"{row['speedup_ratio_optimized_over_baseline']:.4f}x | "
            f"{'yes' if row['both_correct'] else 'no'} |"
        )
    lines.extend([
        "",
        "The 14M P0-to-K001 and 410M K004-to-K010 comparisons are development evidence. The 14M K012-to-K013 and 70M K009-to-K016 comparisons include complete validation. A dense fallback can improve deployed latency but is not a sparse-kernel win for that condition.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    verification = verify_retrieval()
    final_rows = collect_final()
    fits = collect_regressions(final_rows)
    optimization = collect_optimization_pairs()
    public_final = [public_row(row) for row in final_rows]

    write_csv(HERE / "final-results.csv", public_final)
    write_csv(HERE / "regressions.csv", fits)
    write_csv(HERE / "same-rmodel-optimization.csv", optimization)
    (HERE / "source-verification.json").write_text(
        json.dumps(verification, indent=2) + "\n", encoding="utf-8"
    )
    (HERE / "reduction.json").write_text(
        json.dumps({"final": public_final, "regressions": fits, "optimization": optimization}, indent=2) + "\n",
        encoding="utf-8",
    )
    (HERE / "tables.md").write_text(make_tables(public_final, fits, optimization), encoding="utf-8")
    print(
        f"PASS: {len(public_final)} final conditions; "
        f"{sum(row['correctness_pass'] for row in public_final)} qualified; "
        f"{len(optimization)} same-R_model transitions"
    )


if __name__ == "__main__":
    main()
