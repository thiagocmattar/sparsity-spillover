#!/usr/bin/env python3
"""Reduce Run 025 fresh-process, component, and H100-transfer evidence.

The original Analysis 015 reduction owns the first Blackwell process for every
checkpoint.  This reducer has a narrower responsibility: it treats independent
Python processes as the replication unit, separates eager and CUDA-graph modes,
and compares the frozen policies across Blackwell and Hopper without pooling
timing samples as if they were training replicates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "runs" / "025-2026-09-05-pythia-agentic-sparse-kernel-search"

RTX_ATTEMPT = "rtxpro4500-003"
H100_ATTEMPT = "h100nvl-002"

RTX_EXTRACTION = RUN / "retrieved" / RTX_ATTEMPT / "evidence-verified"
H100_EXTRACTION = RUN / "retrieved" / H100_ATTEMPT / "evidence-verified"
RTX_ARTIFACTS_REL = Path("artifacts")
H100_ARTIFACTS_REL = Path("runs") / RUN.name / "artifacts"

RTX_ARCHIVE = RUN / "retrieved" / RTX_ATTEMPT / "run025-rtxpro4500-003-evidence.tar.gz"
H100_ARCHIVE = RUN / "retrieved" / H100_ATTEMPT / "run025-h100nvl-002-evidence.tar"
RTX_CLOSEOUT = RUN / "autoresearch" / "launch-control" / RTX_ATTEMPT / "closeout.json"
H100_CLOSEOUT = RUN / "autoresearch" / "launch-control" / H100_ATTEMPT / "closeout.json"

EXPECTED_ARCHIVES = {
    RTX_ATTEMPT: {
        "bytes": 6_334_093,
        "sha256": "74783671c3db7c795de20f19482d4db583868b0f26d781be2358a82cef839cbf",
        "inventoried_files": 5_486,
        "process_directories": 128,
    },
    H100_ATTEMPT: {
        "bytes": 15_267_840,
        "sha256": "0a38a9f425eb8bfffa287682ca77acdbc7814200e9b99e8202fa55a49a8462b9",
        "inventoried_files": 1_088,
        "process_directories": 92,
    },
}

CONDITION_SUFFIXES = (
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
EXPECTED_RTX_CONDITIONS = {
    f"{size}/{condition}"
    for size in ("14m", "70m", "410m")
    for condition in CONDITION_SUFFIXES
}
H100_SENTINEL_SUFFIXES = ("a0", "a1h", "a4-0", "a4-0p5", "a7-0", "a7-0p5")
EXPECTED_H100_CONDITIONS = {
    f"{size}/{condition}"
    for size in ("14m", "70m", "410m")
    for condition in H100_SENTINEL_SUFFIXES
}

SIZE_ORDER = {"14M": 0, "70M": 1, "410M": 2}
FAMILY_ORDER = {"A0": 0, "A1-H": 1, "A4-OL1": 2, "A7-OL1": 3}
HARDWARE_ORDER = {RTX_ATTEMPT: 0, H100_ATTEMPT: 1}


def extended(path: Path) -> Path:
    """Return a Windows extended path for deeply nested extracted evidence."""
    absolute = str(path.resolve())
    if os.name == "nt" and not absolute.startswith("\\\\?\\"):
        return Path("\\\\?\\" + absolute)
    return Path(absolute)


EXTENDED_ROOT = extended(ROOT)
RTX_ARTIFACTS = extended(RTX_EXTRACTION) / RTX_ARTIFACTS_REL
H100_ARTIFACTS = extended(H100_EXTRACTION) / H100_ARTIFACTS_REL


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_relative(path: Path) -> str:
    absolute = str(path)
    root = str(EXTENDED_ROOT)
    if os.name == "nt":
        absolute_cmp = absolute.lower()
        root_cmp = root.lower().rstrip("\\") + "\\"
        if not absolute_cmp.startswith(root_cmp):
            raise ValueError(f"Evidence path is outside repository: {path}")
        result = absolute[len(root_cmp) :]
    else:
        result = str(path.relative_to(EXTENDED_ROOT))
    return result.replace("\\", "/")


def rank_average(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and array[order[end]] == array[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    return ranks


def process_cluster_ci(input_log_means: dict[int, float], seed_text: str) -> tuple[float, float]:
    values = np.asarray(list(input_log_means.values()), dtype=np.float64)
    if values.size != 16:
        raise ValueError(f"Expected 16 timing-input clusters, got {values.size}")
    seed = int(hashlib.sha256(seed_text.encode("utf-8")).hexdigest()[:16], 16)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(20_000, values.size), replace=True).mean(axis=1)
    low, high = np.exp(np.quantile(draws, [0.025, 0.975]))
    return float(low), float(high)


def read_mode_timing_samples(
    path: Path, reference_mode: str, candidate_mode: str
) -> tuple[list[dict], dict[int, float]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    selected = [row for row in rows if row["mode"] in {reference_mode, candidate_mode}]
    paired: dict[tuple[int, int], dict[str, float]] = defaultdict(dict)
    for row in selected:
        key = (int(row["repeat"]), int(row["input_index"]))
        mode = row["mode"]
        if mode in paired[key]:
            raise ValueError(f"Duplicate {mode} timing for {key}: {path}")
        paired[key][mode] = float(row["host_ms"])
    expected_modes = {reference_mode, candidate_mode}
    if not paired or any(set(values) != expected_modes for values in paired.values()):
        raise ValueError(f"Incomplete {expected_modes} timing pairs: {path}")

    pair_rows: list[dict] = []
    logs_by_input: dict[int, list[float]] = defaultdict(list)
    for (repeat, input_index), values in sorted(paired.items()):
        ratio = values[reference_mode] / values[candidate_mode]
        if not math.isfinite(ratio) or ratio <= 0:
            raise ValueError(f"Invalid paired speedup in {path}")
        pair_rows.append(
            {
                "repeat": repeat,
                "input_index": input_index,
                "reference_host_ms": values[reference_mode],
                "candidate_host_ms": values[candidate_mode],
                "speedup": ratio,
            }
        )
        logs_by_input[input_index].append(math.log(ratio))
    input_log_means = {
        input_index: statistics.fmean(values)
        for input_index, values in logs_by_input.items()
    }
    return pair_rows, input_log_means


def parse_process(directory: Path, attempt: str) -> tuple[str, int | None, str]:
    name = directory.name
    winner_match = re.match(r"(?:fresh|transfer-winner)-r([123])-", name)
    if winner_match:
        return "winner", int(winner_match.group(1)), "full"
    if name.startswith("transfer-p0-"):
        return "p0", 1, "full"
    if name.startswith("component-ffn-"):
        return "component", None, "ffn"
    if name.startswith("component-attnproj-"):
        return "component", None, "attention_projection"
    raise ValueError(f"Unexpected {attempt} process directory: {name}")


def extract_quality(directory: Path, mode: str) -> dict:
    full_path = directory / "full-validation.json"
    development_path = directory / "development-quality.json"
    if full_path.is_file():
        quality = load_json(full_path)
        for field, expected in (
            ("blocks", 338),
            ("documents", 500),
            ("input_tokens", 692_224),
            ("excluded_tail_tokens", 1_444),
        ):
            if quality[field] != expected:
                raise ValueError(f"Invalid complete-validation {field}: {full_path}")
        if not quality["complete"]:
            raise ValueError(f"Complete validation not marked complete: {full_path}")
        scope = "complete-validation"
        quality_path = full_path
    elif development_path.is_file():
        quality = load_json(development_path)
        if int(quality["blocks"]) != 16:
            raise ValueError(f"Expected 16 development blocks: {development_path}")
        scope = "development-16"
        quality_path = development_path
    else:
        raise FileNotFoundError(f"No quality artifact in {directory}")

    reference_key = "native" if mode == "eager" else "native_graph"
    candidate_key = "candidate" if mode == "eager" else "candidate_graph"
    pass_map = quality.get("pass", {})
    loss_map = quality.get("loss", {})
    delta_map = quality.get("loss_delta", {})
    gates_map = quality.get("gates", {})
    failed_gates = [row for row in gates_map.get(candidate_key, []) if not row["pass"]]
    return {
        "scope": scope,
        "path": quality_path,
        "reference_key": reference_key,
        "candidate_key": candidate_key,
        "reference_pass": pass_map.get(reference_key),
        "candidate_pass": pass_map.get(candidate_key),
        "mode_pass": bool(pass_map.get(reference_key)) and bool(pass_map.get(candidate_key)),
        "reference_loss": loss_map.get(reference_key),
        "candidate_loss": loss_map.get(candidate_key),
        "candidate_loss_delta": delta_map.get(candidate_key),
        "failed_candidate_gates": len(failed_gates),
        "max_abs_logit_error": max((float(row["max_abs"]) for row in failed_gates), default=0.0),
        "max_relative_l2": max((float(row["relative_l2"]) for row in failed_gates), default=0.0),
    }


def load_mode_record(directory: Path, attempt: str, mode: str) -> dict:
    manifest_path = directory / "manifest.json"
    status_path = directory / "status.json"
    timing_path = directory / "timing.json"
    sample_path = directory / "timing-samples.jsonl"
    manifest = load_json(manifest_path)
    status = load_json(status_path)
    role, process_repeat, component = parse_process(directory, attempt)
    quality = extract_quality(directory, mode)

    checkpoint = manifest["checkpoint"]
    canonical = checkpoint["canonical_logical_products"]
    measured = canonical["measured"]
    coverage = canonical["coverage"]
    integer_fraction = measured["block_zero_product_count"] / measured["model_product_count"]
    if not math.isclose(integer_fraction, measured["R_model"], rel_tol=0, abs_tol=1e-15):
        raise ValueError(f"Canonical R_model integer-count mismatch: {directory}")
    if coverage["sequences"] != 338 or not coverage["complete_block_coverage"]:
        raise ValueError(f"Incomplete canonical logical-product coverage: {directory}")

    group_name = "eager" if mode == "eager" else "cuda_graph"
    reference_key = quality["reference_key"]
    candidate_key = quality["candidate_key"]
    timing_available = False
    speedup: float | str = ""
    reference_median: float | str = ""
    candidate_median: float | str = ""
    timing_samples: int | str = ""
    timing_inputs: int | str = ""
    ci_low: float | str = ""
    ci_high: float | str = ""
    timing_sha: str | str = ""
    sample_sha: str | str = ""
    if timing_path.is_file():
        timing_document = load_json(timing_path)
        timing_group = timing_document.get("matched", {}).get(group_name)
        if timing_group is not None:
            summary = timing_group["summary"]
            pairs, input_log_means = read_mode_timing_samples(
                sample_path, reference_key, candidate_key
            )
            recomputed = math.exp(statistics.fmean(math.log(row["speedup"]) for row in pairs))
            recorded = float(summary[candidate_key]["paired_geomean_speedup"])
            if not math.isclose(recomputed, recorded, rel_tol=0, abs_tol=1e-12):
                raise ValueError(f"Timing reduction mismatch: {directory}/{mode}")
            if len(pairs) != int(summary[candidate_key]["samples"]):
                raise ValueError(f"Timing sample-count mismatch: {directory}/{mode}")
            reference_values = [row["reference_host_ms"] for row in pairs]
            candidate_values = [row["candidate_host_ms"] for row in pairs]
            reference_median = float(summary[reference_key]["median_host_ms"])
            candidate_median = float(summary[candidate_key]["median_host_ms"])
            if not math.isclose(statistics.median(reference_values), reference_median, abs_tol=1e-12):
                raise ValueError(f"Reference median mismatch: {directory}/{mode}")
            if not math.isclose(statistics.median(candidate_values), candidate_median, abs_tol=1e-12):
                raise ValueError(f"Candidate median mismatch: {directory}/{mode}")
            if len(pairs) != 80 or len(input_log_means) != 16:
                raise ValueError(f"Expected 80 paired timings over 16 inputs: {directory}/{mode}")
            ci_low, ci_high = process_cluster_ci(input_log_means, f"{attempt}/{directory.name}/{mode}")
            timing_available = True
            speedup = recorded
            timing_samples = len(pairs)
            timing_inputs = len(input_log_means)
            timing_sha = sha256(timing_path)
            sample_sha = sha256(sample_path)

    size = str(checkpoint["size"]).upper().replace("PYTHIA-", "")
    condition = manifest["arguments"]["condition"]
    kappa = checkpoint.get("kappa")
    qualified = (
        timing_available
        and quality["scope"] == "complete-validation"
        and quality["mode_pass"]
    )
    return {
        "hardware_attempt": attempt,
        "gpu": manifest["gpu"],
        "process_id": directory.name,
        "role": role,
        "process_repeat": "" if process_repeat is None else process_repeat,
        "component": component,
        "execution_mode": mode,
        "condition_id": condition,
        "model_size": size,
        "family": checkpoint["family"],
        "kappa": "" if kappa is None else float(kappa),
        "partition": manifest.get("partition") or checkpoint.get("partition") or "development",
        "implementation": manifest["arguments"]["implementation"],
        "selected_linear_sites": ";".join(manifest.get("selected_linear_sites", [])),
        "attention_qk_pv_execution": manifest.get("attention", "not recorded"),
        "process_stage": status["stage"],
        "process_error": status.get("error", ""),
        "quality_scope": quality["scope"],
        "reference_correctness_pass": quality["reference_pass"],
        "candidate_correctness_pass": quality["candidate_pass"],
        "mode_correctness_pass": quality["mode_pass"],
        "failed_candidate_validation_blocks": quality["failed_candidate_gates"],
        "max_abs_logit_error": quality["max_abs_logit_error"],
        "max_relative_l2": quality["max_relative_l2"],
        "qualified_complete_timing": qualified,
        "canonical_validation_loss": float(coverage["loss"]),
        "measured_R_model_fraction": float(measured["R_model"]),
        "measured_R_model_percent": 100.0 * float(measured["R_model"]),
        "block_zero_product_count": int(measured["block_zero_product_count"]),
        "model_product_count": int(measured["model_product_count"]),
        "run025_reference_validation_loss": quality["reference_loss"],
        "run025_candidate_validation_loss": quality["candidate_loss"],
        "run025_candidate_validation_loss_delta": quality["candidate_loss_delta"],
        "timing_available": timing_available,
        "paired_geomean_speedup": speedup,
        "reference_median_host_ms": reference_median,
        "candidate_median_host_ms": candidate_median,
        "speedup_input_cluster_ci95_low": ci_low,
        "speedup_input_cluster_ci95_high": ci_high,
        "paired_timing_samples": timing_samples,
        "timing_input_clusters": timing_inputs,
        "input_manifest_sha256": manifest["input_manifest"]["sha256"],
        "development_input_sha256": manifest["development"]["sha256"],
        "manifest_sha256": sha256(manifest_path),
        "timing_sha256": timing_sha,
        "timing_samples_sha256": sample_sha,
        "quality_sha256": sha256(quality["path"]),
        "source": source_relative(directory),
    }


def collect_attempt(artifacts: Path, attempt: str) -> tuple[list[dict], int]:
    prefixes = ("fresh-", "component-") if attempt == RTX_ATTEMPT else (
        "transfer-p0-",
        "transfer-winner-",
        "component-",
    )
    directories = sorted(
        [path for path in artifacts.iterdir() if path.is_dir() and path.name.startswith(prefixes)],
        key=lambda path: path.name,
    )
    expected_count = EXPECTED_ARCHIVES[attempt]["process_directories"]
    if len(directories) != expected_count:
        raise ValueError(f"{attempt} process inventory {len(directories)} != {expected_count}")

    records: list[dict] = []
    for directory in directories:
        manifest = load_json(directory / "manifest.json")
        execution = manifest["arguments"]["execution"]
        if not execution or any(mode not in {"eager", "graph"} for mode in execution):
            raise ValueError(f"Unexpected execution modes in {directory}: {execution}")
        for declared_mode in execution:
            mode = "cuda_graph" if declared_mode == "graph" else "eager"
            records.append(load_mode_record(directory, attempt, mode))
    return records, len(directories)


def canonical_identity(record: dict) -> tuple:
    return (
        record["condition_id"],
        record["canonical_validation_loss"],
        record["measured_R_model_fraction"],
        record["block_zero_product_count"],
        record["model_product_count"],
    )


def verify_cross_realization_identity(records: list[dict]) -> None:
    by_condition: dict[str, set[tuple]] = defaultdict(set)
    for record in records:
        by_condition[record["condition_id"]].add(canonical_identity(record))
    mismatches = {condition: values for condition, values in by_condition.items() if len(values) != 1}
    if mismatches:
        raise ValueError(f"Canonical checkpoint identity changed across realizations: {mismatches}")


def primary_winner_records(records: list[dict]) -> list[dict]:
    selected = []
    for row in records:
        if row["role"] != "winner" or row["execution_mode"] != "eager":
            continue
        repeat = int(row["process_repeat"])
        if row["hardware_attempt"] == RTX_ATTEMPT or repeat in {2, 3}:
            selected.append(row)
    return selected


def aggregate_winners(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in primary_winner_records(records):
        grouped[(row["hardware_attempt"], row["condition_id"])].append(row)

    expected_groups = {
        (RTX_ATTEMPT, condition) for condition in EXPECTED_RTX_CONDITIONS
    } | {
        (H100_ATTEMPT, condition) for condition in EXPECTED_H100_CONDITIONS
    }
    if set(grouped) != expected_groups:
        raise ValueError("Primary winner condition inventory mismatch")

    summaries: list[dict] = []
    for (attempt, condition), rows in grouped.items():
        rows.sort(key=lambda row: int(row["process_repeat"]))
        expected_repeats = {1, 2, 3} if attempt == RTX_ATTEMPT else {2, 3}
        repeats = {int(row["process_repeat"]) for row in rows}
        if repeats != expected_repeats:
            raise ValueError(f"Primary process repeats mismatch for {attempt}/{condition}: {repeats}")
        if not all(row["timing_available"] for row in rows):
            raise ValueError(f"Missing primary timing for {attempt}/{condition}")
        metadata_keys = (
            "model_size",
            "family",
            "kappa",
            "partition",
            "implementation",
            "canonical_validation_loss",
            "measured_R_model_fraction",
            "block_zero_product_count",
            "model_product_count",
            "input_manifest_sha256",
            "development_input_sha256",
        )
        for key in metadata_keys:
            if len({row[key] for row in rows}) != 1:
                raise ValueError(f"Process metadata mismatch {attempt}/{condition}/{key}")

        speedups = [float(row["paired_geomean_speedup"]) for row in rows]
        mean_speedup = statistics.fmean(speedups)
        process_sd = statistics.pstdev(speedups)
        base = rows[0]
        summaries.append(
            {
                "hardware_attempt": attempt,
                "gpu": base["gpu"],
                "condition_id": condition,
                "model_size": base["model_size"],
                "family": base["family"],
                "kappa": base["kappa"],
                "partition": base["partition"],
                "implementation": base["implementation"],
                "canonical_validation_loss": base["canonical_validation_loss"],
                "measured_R_model_fraction": base["measured_R_model_fraction"],
                "measured_R_model_percent": base["measured_R_model_percent"],
                "block_zero_product_count": base["block_zero_product_count"],
                "model_product_count": base["model_product_count"],
                "primary_process_repeats": ";".join(str(value) for value in sorted(repeats)),
                "n_fresh_processes": len(rows),
                "n_complete_validation": sum(row["quality_scope"] == "complete-validation" for row in rows),
                "n_correct_processes": sum(bool(row["mode_correctness_pass"]) for row in rows),
                "qualified": all(bool(row["qualified_complete_timing"]) for row in rows),
                "process_median_speedup": statistics.median(speedups),
                "process_mean_speedup": mean_speedup,
                "process_min_speedup": min(speedups),
                "process_max_speedup": max(speedups),
                "process_population_sd": process_sd,
                "process_cv": process_sd / mean_speedup,
                "median_native_host_ms": statistics.median(
                    float(row["reference_median_host_ms"]) for row in rows
                ),
                "median_candidate_host_ms": statistics.median(
                    float(row["candidate_median_host_ms"]) for row in rows
                ),
                "paired_timing_samples_per_process": 80,
                "timing_input_clusters_per_process": 16,
                "aggregation_unit": "median of independent-process paired geometric speedups",
                "process_policy": (
                    "RTX eager-only repeats 1-3"
                    if attempt == RTX_ATTEMPT
                    else "H100 eager-only repeats 2-3; repeat 1 reserved for graph control"
                ),
                "process_sources": ";".join(row["source"] for row in rows),
            }
        )
    summaries.sort(
        key=lambda row: (
            HARDWARE_ORDER[row["hardware_attempt"]],
            SIZE_ORDER[row["model_size"]],
            FAMILY_ORDER[row["family"]],
            -1.0 if row["kappa"] == "" else float(row["kappa"]),
        )
    )
    return summaries


def fit_group(rows: list[dict], attempt: str, size: str, stratum: str, failures: int) -> dict:
    x = np.asarray([float(row["measured_R_model_fraction"]) for row in rows], dtype=np.float64)
    y = np.asarray([float(row["process_median_speedup"]) for row in rows], dtype=np.float64)
    if len(rows) < 3 or np.ptp(x) == 0 or np.ptp(y) == 0:
        raise ValueError(f"Invalid regression group {attempt}/{size}/{stratum}")
    slope, intercept = np.polyfit(x, y, 1)
    prediction = intercept + slope * x
    total = float(np.square(y - y.mean()).sum())
    residual = float(np.square(y - prediction).sum())
    pearson = float(np.corrcoef(x, y)[0, 1])
    spearman = float(np.corrcoef(rank_average(x.tolist()), rank_average(y.tolist()))[0, 1])
    loo_slopes: list[float] = []
    loo_r2: list[float] = []
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
        "hardware_attempt": attempt,
        "gpu": rows[0]["gpu"],
        "model_size": size,
        "stratum": stratum,
        "n_checkpoints": len(rows),
        "n_correctness_failures_excluded": failures,
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
        "method": "descriptive unweighted OLS with intercept; one checkpoint median per row",
    }


def collect_regressions(summaries: list[dict]) -> list[dict]:
    fits: list[dict] = []
    for attempt in HARDWARE_ORDER:
        for size in SIZE_ORDER:
            all_rows = [
                row
                for row in summaries
                if row["hardware_attempt"] == attempt and row["model_size"] == size
            ]
            if not all_rows:
                continue
            qualified = [row for row in all_rows if row["qualified"]]
            fits.append(
                fit_group(
                    qualified,
                    attempt,
                    size,
                    "primary-qualified",
                    len(all_rows) - len(qualified),
                )
            )
            if attempt == RTX_ATTEMPT:
                sentinels = [
                    row for row in qualified if row["condition_id"] in EXPECTED_H100_CONDITIONS
                ]
                fits.append(fit_group(sentinels, attempt, size, "H100-matched-sentinels", 0))
                for family in ("A4-OL1", "A7-OL1"):
                    family_rows = [row for row in qualified if row["family"] == family]
                    if len(family_rows) >= 3:
                        fits.append(fit_group(family_rows, attempt, size, f"{family}-qualified", 0))
    return fits


def collect_hardware_transfer(summaries: list[dict]) -> list[dict]:
    by_key = {(row["hardware_attempt"], row["condition_id"]): row for row in summaries}
    rows: list[dict] = []
    for condition in sorted(
        EXPECTED_H100_CONDITIONS,
        key=lambda value: (
            ("14m", "70m", "410m").index(value.split("/")[0]),
            H100_SENTINEL_SUFFIXES.index(value.split("/")[1]),
        ),
    ):
        rtx = by_key[(RTX_ATTEMPT, condition)]
        h100 = by_key[(H100_ATTEMPT, condition)]
        for key in (
            "implementation",
            "canonical_validation_loss",
            "measured_R_model_fraction",
            "block_zero_product_count",
            "model_product_count",
        ):
            if rtx[key] != h100[key]:
                raise ValueError(f"Hardware transfer changed {key}: {condition}")
        rows.append(
            {
                "condition_id": condition,
                "model_size": rtx["model_size"],
                "family": rtx["family"],
                "kappa": rtx["kappa"],
                "implementation": rtx["implementation"],
                "canonical_validation_loss": rtx["canonical_validation_loss"],
                "measured_R_model_fraction": rtx["measured_R_model_fraction"],
                "measured_R_model_percent": rtx["measured_R_model_percent"],
                "rtx_process_median_speedup": rtx["process_median_speedup"],
                "rtx_process_min_speedup": rtx["process_min_speedup"],
                "rtx_process_max_speedup": rtx["process_max_speedup"],
                "h100_process_median_speedup": h100["process_median_speedup"],
                "h100_process_min_speedup": h100["process_min_speedup"],
                "h100_process_max_speedup": h100["process_max_speedup"],
                "h100_minus_rtx_speedup": h100["process_median_speedup"] - rtx["process_median_speedup"],
                "h100_over_rtx_speedup_ratio": h100["process_median_speedup"] / rtx["process_median_speedup"],
                "both_qualified": bool(rtx["qualified"] and h100["qualified"]),
                "interpretation": "same frozen policy/checkpoint/workload; independent hardware processes",
            }
        )
    return rows


def collect_components(records: list[dict], summaries: list[dict]) -> list[dict]:
    winners = {(row["hardware_attempt"], row["condition_id"]): row for row in summaries}
    rows: list[dict] = []
    for record in records:
        if record["role"] != "component" or record["execution_mode"] != "eager":
            continue
        winner = winners[(record["hardware_attempt"], record["condition_id"])]
        rows.append(
            {
                "hardware_attempt": record["hardware_attempt"],
                "gpu": record["gpu"],
                "condition_id": record["condition_id"],
                "model_size": record["model_size"],
                "family": record["family"],
                "kappa": record["kappa"],
                "implementation": record["implementation"],
                "component": record["component"],
                "selected_linear_sites": record["selected_linear_sites"],
                "measured_R_model_fraction": record["measured_R_model_fraction"],
                "measured_R_model_percent": record["measured_R_model_percent"],
                "component_speedup": record["paired_geomean_speedup"],
                "component_speedup_input_cluster_ci95_low": record["speedup_input_cluster_ci95_low"],
                "component_speedup_input_cluster_ci95_high": record["speedup_input_cluster_ci95_high"],
                "component_correctness_pass": record["qualified_complete_timing"],
                "full_policy_process_median_speedup": winner["process_median_speedup"],
                "full_policy_process_min_speedup": winner["process_min_speedup"],
                "full_policy_process_max_speedup": winner["process_max_speedup"],
                "component_minus_native": float(record["paired_geomean_speedup"]) - 1.0,
                "full_policy_minus_native": winner["process_median_speedup"] - 1.0,
                "interpretation": "full-model timing with only the named component eligible for sparse execution; effects are not additive",
                "source": record["source"],
            }
        )
    rows.sort(
        key=lambda row: (
            HARDWARE_ORDER[row["hardware_attempt"]],
            SIZE_ORDER[row["model_size"]],
            FAMILY_ORDER[row["family"]],
            -1.0 if row["kappa"] == "" else float(row["kappa"]),
            row["component"],
        )
    )
    if len(rows) != 40:
        raise ValueError(f"Expected 40 component measurements, got {len(rows)}")
    if not all(row["component_correctness_pass"] for row in rows):
        raise ValueError("A component measurement failed complete correctness")
    return rows


def collect_graph_controls(records: list[dict]) -> list[dict]:
    rows = [
        {key: value for key, value in record.items()}
        for record in records
        if record["hardware_attempt"] == H100_ATTEMPT
        and record["execution_mode"] == "cuda_graph"
    ]
    rows.sort(key=lambda row: (row["role"], row["model_size"], row["condition_id"]))
    if len(rows) != 36:
        raise ValueError(f"Expected 36 declared H100 graph controls, got {len(rows)}")
    return rows


def collect_h100_same_rmodel(records: list[dict]) -> list[dict]:
    eager = [
        row
        for row in records
        if row["hardware_attempt"] == H100_ATTEMPT and row["execution_mode"] == "eager"
    ]
    p0 = {row["condition_id"]: row for row in eager if row["role"] == "p0"}
    winner_r1 = {
        row["condition_id"]: row
        for row in eager
        if row["role"] == "winner" and int(row["process_repeat"]) == 1
    }
    if set(p0) != EXPECTED_H100_CONDITIONS or set(winner_r1) != EXPECTED_H100_CONDITIONS:
        raise ValueError("H100 P0/winner-r1 inventory mismatch")
    rows: list[dict] = []
    for condition in sorted(EXPECTED_H100_CONDITIONS):
        baseline = p0[condition]
        optimized = winner_r1[condition]
        for key in (
            "canonical_validation_loss",
            "measured_R_model_fraction",
            "block_zero_product_count",
            "model_product_count",
            "input_manifest_sha256",
            "development_input_sha256",
        ):
            if baseline[key] != optimized[key]:
                raise ValueError(f"H100 same-R comparison changed {key}: {condition}")
        timing_pair = bool(baseline["timing_available"] and optimized["timing_available"])
        both_qualified = bool(
            timing_pair
            and baseline["qualified_complete_timing"]
            and optimized["qualified_complete_timing"]
        )
        baseline_speed = baseline["paired_geomean_speedup"]
        optimized_speed = optimized["paired_geomean_speedup"]
        rows.append(
            {
                "condition_id": condition,
                "model_size": baseline["model_size"],
                "family": baseline["family"],
                "kappa": baseline["kappa"],
                "measured_R_model_fraction": baseline["measured_R_model_fraction"],
                "measured_R_model_percent": baseline["measured_R_model_percent"],
                "baseline_implementation": baseline["implementation"],
                "optimized_implementation": optimized["implementation"],
                "baseline_timing_available": baseline["timing_available"],
                "optimized_timing_available": optimized["timing_available"],
                "timing_pair_available": timing_pair,
                "baseline_speedup": baseline_speed,
                "optimized_speedup": optimized_speed,
                "optimized_minus_baseline_speedup": (
                    float(optimized_speed) - float(baseline_speed) if timing_pair else ""
                ),
                "optimized_over_baseline_ratio": (
                    float(optimized_speed) / float(baseline_speed) if timing_pair else ""
                ),
                "baseline_qualified": baseline["qualified_complete_timing"],
                "optimized_qualified": optimized["qualified_complete_timing"],
                "both_qualified": both_qualified,
                "comparison_status": (
                    "both qualified"
                    if both_qualified
                    else "timed but at least one numerical gate failed"
                    if timing_pair
                    else "pre-timing numerical gate stopped at least one implementation"
                ),
                "scope": "one P0 and one graph-enabled winner process; eager mode only",
                "baseline_source": baseline["source"],
                "optimized_source": optimized["source"],
            }
        )
    return rows


def verify_retrievals(rtx_processes: int, h100_processes: int) -> dict:
    closeouts = {
        RTX_ATTEMPT: load_json(RTX_CLOSEOUT),
        H100_ATTEMPT: load_json(H100_CLOSEOUT),
    }
    archives = {RTX_ATTEMPT: RTX_ARCHIVE, H100_ATTEMPT: H100_ARCHIVE}
    process_counts = {RTX_ATTEMPT: rtx_processes, H100_ATTEMPT: h100_processes}
    result: dict[str, dict] = {}
    for attempt, expected in EXPECTED_ARCHIVES.items():
        archive = archives[attempt]
        actual_bytes = archive.stat().st_size
        actual_sha = sha256(archive)
        closeout = closeouts[attempt]
        if actual_bytes != expected["bytes"] or actual_sha != expected["sha256"]:
            raise ValueError(f"Archive identity mismatch for {attempt}")
        if closeout["archive_bytes"] != actual_bytes or closeout["archive_sha256"] != actual_sha:
            raise ValueError(f"Closeout/archive disagreement for {attempt}")
        if process_counts[attempt] != expected["process_directories"]:
            raise ValueError(f"Process count mismatch for {attempt}")
        result[attempt] = {
            "archive": source_relative(extended(archive)),
            "bytes": actual_bytes,
            "sha256": actual_sha,
            "inventoried_files": expected["inventoried_files"],
            "process_directories": process_counts[attempt],
            "closeout": source_relative(extended(RTX_CLOSEOUT if attempt == RTX_ATTEMPT else H100_CLOSEOUT)),
        }
    return result


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def display_kappa(value: object) -> str:
    return "-" if value == "" else f"{float(value):g}"


def make_tables(
    summaries: list[dict],
    fits: list[dict],
    transfers: list[dict],
    same_r: list[dict],
    components: list[dict],
    graphs: list[dict],
) -> str:
    lines = [
        "# Run 025 fresh-process and hardware-transfer results",
        "",
        "The replication unit is a fresh Python process. RTX summaries use the median of three eager-only process estimates. H100 summaries use the median of eager-only repeats 2 and 3; repeat 1 intentionally initialized and measured CUDA graphs and is retained only in the graph/sensitivity records. Each process estimate is itself a paired geometric mean over 80 native/candidate timings on 16 fixed input blocks.",
        "",
        "## Frozen winner measurements",
        "",
        "| GPU | size | family | kappa | loss | R_model | process speedup range; median | n | qualified |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | ---: | --- |",
    ]
    for row in summaries:
        lines.append(
            f"| {row['gpu']} | {row['model_size']} | {row['family']} | {display_kappa(row['kappa'])} | "
            f"{row['canonical_validation_loss']:.6f} | {row['measured_R_model_percent']:.4f}% | "
            f"[{row['process_min_speedup']:.4f}, {row['process_max_speedup']:.4f}]; "
            f"**{row['process_median_speedup']:.4f}x** | {row['n_fresh_processes']} | "
            f"{'yes' if row['qualified'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## Descriptive associations",
            "",
            "| GPU | size | stratum | n | excluded | slope / +10 pp R_model | R2 | Spearman rho | LOO slope range |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in fits:
        loo = (
            "-"
            if row["loo_slope_min"] == ""
            else f"[{float(row['loo_slope_min']) * 0.1:+.4f}, {float(row['loo_slope_max']) * 0.1:+.4f}]"
        )
        lines.append(
            f"| {row['gpu']} | {row['model_size']} | {row['stratum']} | {row['n_checkpoints']} | "
            f"{row['n_correctness_failures_excluded']} | {row['slope_per_10_percentage_points']:+.4f}x | "
            f"{row['R_squared']:.4f} | {row['spearman_rho']:+.4f} | {loo} |"
        )

    lines.extend(
        [
            "",
            "## Matched hardware transfer",
            "",
            "| size | family | kappa | R_model | RTX median | H100 median | H100 - RTX | both qualified |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in transfers:
        lines.append(
            f"| {row['model_size']} | {row['family']} | {display_kappa(row['kappa'])} | "
            f"{row['measured_R_model_percent']:.4f}% | {row['rtx_process_median_speedup']:.4f}x | "
            f"{row['h100_process_median_speedup']:.4f}x | {row['h100_minus_rtx_speedup']:+.4f}x | "
            f"{'yes' if row['both_qualified'] else 'no'} |"
        )

    lines.extend(
        [
            "",
            "## H100 same-R_model starting-point comparisons",
            "",
            "| size | family | kappa | R_model | P0 speedup | winner speedup | ratio | status |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in same_r:
        baseline = "-" if row["baseline_speedup"] == "" else f"{float(row['baseline_speedup']):.4f}x"
        optimized = "-" if row["optimized_speedup"] == "" else f"{float(row['optimized_speedup']):.4f}x"
        ratio = (
            "-"
            if row["optimized_over_baseline_ratio"] == ""
            else f"{float(row['optimized_over_baseline_ratio']):.4f}x"
        )
        lines.append(
            f"| {row['model_size']} | {row['family']} | {display_kappa(row['kappa'])} | "
            f"{row['measured_R_model_percent']:.4f}% | {baseline} | {optimized} | {ratio} | "
            f"{row['comparison_status']} |"
        )

    lines.extend(
        [
            "",
            "## Component probes",
            "",
            "| GPU | size | family | kappa | component | sites | component speedup | full-policy median |",
            "| --- | --- | --- | ---: | --- | --- | ---: | ---: |",
        ]
    )
    for row in components:
        lines.append(
            f"| {row['gpu']} | {row['model_size']} | {row['family']} | {display_kappa(row['kappa'])} | "
            f"{row['component']} | {row['selected_linear_sites']} | {float(row['component_speedup']):.4f}x | "
            f"{row['full_policy_process_median_speedup']:.4f}x |"
        )

    lines.extend(
        [
            "",
            "## H100 CUDA-graph audit",
            "",
            "| role | size | family | kappa | implementation | graph speedup | validation scope | graph correct | process stage |",
            "| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |",
        ]
    )
    for row in graphs:
        speed = "-" if row["paired_geomean_speedup"] == "" else f"{float(row['paired_geomean_speedup']):.4f}x"
        lines.append(
            f"| {row['role']} | {row['model_size']} | {row['family']} | {display_kappa(row['kappa'])} | "
            f"{row['implementation']} | {speed} | {row['quality_scope']} | "
            f"{'yes' if row['mode_correctness_pass'] else 'no'} | {row['process_stage']} |"
        )

    lines.extend(
        [
            "",
            "Timing repetitions are systems repetitions, not additional model seeds. Component effects are full-model timings with only the named linears eligible and are not additive. QK/PV attention remained dense SDPA; `attention_projection` refers only to the sparse output/projection linears. `R_model` remains the canonical logical-product opportunity of the checkpoint, not measured runtime savings.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    rtx_records, rtx_processes = collect_attempt(RTX_ARTIFACTS, RTX_ATTEMPT)
    h100_records, h100_processes = collect_attempt(H100_ARTIFACTS, H100_ATTEMPT)
    records = rtx_records + h100_records
    if len(rtx_records) != 128 or len(h100_records) != 128:
        raise ValueError(
            f"Expected 128 declared process-mode rows per attempt; got {len(rtx_records)} and {len(h100_records)}"
        )
    verify_cross_realization_identity(records)
    retrievals = verify_retrievals(rtx_processes, h100_processes)
    summaries = aggregate_winners(records)
    fits = collect_regressions(summaries)
    transfers = collect_hardware_transfer(summaries)
    components = collect_components(records, summaries)
    graphs = collect_graph_controls(records)
    same_r = collect_h100_same_rmodel(records)

    write_csv(HERE / "replication-processes.csv", records)
    write_csv(HERE / "replication-summary.csv", summaries)
    write_csv(HERE / "replication-regressions.csv", fits)
    write_csv(HERE / "hardware-transfer.csv", transfers)
    write_csv(HERE / "component-results.csv", components)
    write_csv(HERE / "cuda-graph-controls.csv", graphs)
    write_csv(HERE / "h100-same-rmodel.csv", same_r)
    (HERE / "replication-reduction.json").write_text(
        json.dumps(
            {
                "retrievals": retrievals,
                "inventory": {
                    "process_directories": {
                        RTX_ATTEMPT: rtx_processes,
                        H100_ATTEMPT: h100_processes,
                    },
                    "declared_process_mode_rows": {
                        RTX_ATTEMPT: len(rtx_records),
                        H100_ATTEMPT: len(h100_records),
                    },
                    "primary_winner_summaries": len(summaries),
                    "hardware_transfer_conditions": len(transfers),
                    "component_measurements": len(components),
                    "declared_graph_controls": len(graphs),
                    "timed_graph_controls": sum(row["timing_available"] for row in graphs),
                    "h100_same_rmodel_conditions": len(same_r),
                    "timed_h100_same_rmodel_pairs": sum(row["timing_pair_available"] for row in same_r),
                    "qualified_h100_same_rmodel_pairs": sum(row["both_qualified"] for row in same_r),
                },
                "primary_winner_summary": summaries,
                "regressions": fits,
                "hardware_transfer": transfers,
                "h100_same_rmodel": same_r,
                "components": components,
                "cuda_graph_controls": graphs,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (HERE / "replication-tables.md").write_text(
        make_tables(summaries, fits, transfers, same_r, components, graphs),
        encoding="utf-8",
    )
    print(
        "PASS: "
        f"{len(records)} declared process-mode rows; "
        f"{len(summaries)} primary winner summaries; "
        f"{len(transfers)} matched hardware transfers; "
        f"{len(components)} component probes"
    )


if __name__ == "__main__":
    main()
