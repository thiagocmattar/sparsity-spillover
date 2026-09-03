"""Build the selected-ladder frontiers and A0 gradient diagnostic."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import correlation
from typing import Any, Iterable


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent.parent
RUN_004 = REPO_ROOT / "runs" / "004-2026-08-29-pythia14m-full-pass-l1n"
RUN_014 = REPO_ROOT / "runs" / "014-2026-08-31-pythia14m-full-pass-a7-ol1"
RUN_015 = REPO_ROOT / "runs" / "015-2026-08-31-pythia14m-corrected-a4-ol1"
RUN_018 = REPO_ROOT / "runs" / "018-2026-09-01-pythia70m-selected-ladder-canonical-init"
RUN_019 = REPO_ROOT / "runs" / "019-2026-09-01-pythia410m-selected-ladder-canonical-init"
TEAL_14 = (
    REPO_ROOT
    / "analyses"
    / "005-2026-08-30-run004-controls-teal-posthoc"
    / "teal_frontier.json"
)
TEAL_70 = RUN_018 / "artifacts" / "teal" / "teal_frontiers.json"
TEAL_410 = RUN_019 / "artifacts" / "teal" / "teal_frontiers.json"
FIGURE_DATA = ANALYSIS_DIR / "figure_data.json"
TABLES = ANALYSIS_DIR / "tables.md"
FIGURE_410 = ANALYSIS_DIR / "figures" / "01-pythia410m-selected-ladder.pdf"
FIGURE_ALL = ANALYSIS_DIR / "figures" / "02-pythia14m-70m-410m-selected-ladder.pdf"
FIGURE_A0_GRADIENT = ANALYSIS_DIR / "figures" / "03-a0-gradient-norm-vs-tokens.pdf"

SCALES = ("14M", "70M", "410M")
FAMILIES = ("A4-OL1", "A7-OL1")
CONTROLS = ("A0", "A1-H")
KAPPAS = (0.0, 0.01, 0.05, 0.1, 0.5)
TARGETS = tuple(index / 10 for index in range(10))
SITES = ("a", "m", "h", "q_post", "k_post", "v", "z", "attention_output")
COVERAGE = {
    "sequences": 338,
    "input_tokens": 692_224,
    "excluded_tail_tokens": 1_444,
    "complete_block_coverage": True,
}
TRAINING_STEPS = 712
TRAINING_TOKENS = 1_493_172_224
TOKENS_PER_BOUNDARY = 2_097_152
A0_EVENT_PATHS = {
    "14M": RUN_004
    / "artifacts"
    / "attempts"
    / "001-20260829-221007-bb5288c8"
    / "events.jsonl",
    "70M": RUN_018
    / "artifacts"
    / "attempts"
    / "001-20260901-133016-4e43b254"
    / "events.jsonl",
    "410M": RUN_019
    / "artifacts"
    / "attempts"
    / "001-20260902-141527-bcb97fb1"
    / "events.jsonl",
}
A0_CONDITION_IDS = {"14M": "gelu-control", "70M": "a0-gelu", "410M": "a0-gelu"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"Numeric mismatch: {actual!r} != {expected!r}")


def _require_coverage(row: dict[str, Any], source: str) -> None:
    for key, expected in COVERAGE.items():
        if row.get(key) != expected:
            raise ValueError(f"Coverage mismatch for {source}/{key}")


def _event_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _a0_gradient_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for scale in SCALES:
        events_path = A0_EVENT_PATHS[scale]
        training = [
            row for row in _event_rows(events_path) if row.get("event") == "train"
        ]
        if len(training) != TRAINING_STEPS:
            raise ValueError(f"Incomplete A0 boundary history for {scale}")
        if [int(row["step"]) for row in training] != list(
            range(1, TRAINING_STEPS + 1)
        ):
            raise ValueError(f"Non-contiguous A0 boundary history for {scale}")

        clipped_count = 0
        task_loss_values = []
        pre_values = []
        post_values = []
        for event in training:
            step = int(event["step"])
            tokens = int(event["input_tokens_seen"])
            task_loss = float(event["task_loss"])
            pre = float(event["adamw_gradient_norm_pre_clip"])
            post = float(event["adamw_gradient_norm_post_clip"])
            threshold = float(event["adamw_gradient_clip_norm"])
            clipped = bool(event["adamw_gradient_was_clipped"])
            if event.get("condition_id") != A0_CONDITION_IDS[scale]:
                raise ValueError(f"A0 condition identity mismatch for {scale}")
            if tokens != step * TOKENS_PER_BOUNDARY:
                raise ValueError(f"A0 token coordinate mismatch for {scale}/step {step}")
            if not all(
                math.isfinite(value) and value > 0.0
                for value in (task_loss, pre, post)
            ):
                raise ValueError(f"Invalid A0 training metric for {scale}/step {step}")
            if threshold != 1.0 or event.get("adamw_gradient_clipping_enabled") is not True:
                raise ValueError(f"A0 clipping contract mismatch for {scale}/step {step}")
            if event.get("gradient_overflow") or event.get("optimizer_step_skipped"):
                raise ValueError(f"A0 overflow or skipped update for {scale}/step {step}")
            if clipped != (pre > threshold):
                raise ValueError(f"A0 clipping flag mismatch for {scale}/step {step}")
            if clipped:
                _close(post, threshold, tolerance=1e-5)
                clipped_count += 1
            else:
                _close(post, pre, tolerance=1e-5)
            task_loss_values.append(task_loss)
            pre_values.append(pre)
            post_values.append(post)
            rows.append(
                {
                    "scale": scale,
                    "condition_id": A0_CONDITION_IDS[scale],
                    "step": step,
                    "input_tokens_seen": tokens,
                    "task_loss": task_loss,
                    "gradient_norm_pre_clip": pre,
                    "gradient_norm_post_clip": post,
                    "clip_threshold": threshold,
                    "clipped": clipped,
                }
            )
        summaries.append(
            {
                "scale": scale,
                "boundaries": len(training),
                "clipped_boundaries": clipped_count,
                "clipped_fraction": clipped_count / len(training),
                "initial_task_loss": task_loss_values[0],
                "final_task_loss": task_loss_values[-1],
                "minimum_task_loss": min(task_loss_values),
                "maximum_task_loss": max(task_loss_values),
                "minimum_pre_clip_norm": min(pre_values),
                "maximum_pre_clip_norm": max(pre_values),
                "minimum_post_clip_norm": min(post_values),
                "maximum_post_clip_norm": max(post_values),
            }
        )
    return rows, summaries


def _matches_family(condition_id: str, scale: str, family: str) -> bool:
    if scale == "14M" and family == "A4-OL1":
        return condition_id.startswith("a4z-ol1-kappa-")
    prefix = "a4-ol1-kappa-" if family == "A4-OL1" else "a7-ol1-kappa-"
    return condition_id.startswith(prefix)


def _trained_rows(run_dir: Path, scale: str, family: str) -> list[dict[str, Any]]:
    rows = []
    for attempt in sorted((run_dir / "artifacts" / "attempts").iterdir()):
        if not attempt.is_dir() or not (attempt / "metrics.json").exists():
            continue
        metrics_path = attempt / "metrics.json"
        manifest_path = attempt / "manifest.json"
        logical_path = attempt / "diagnostics" / "logical_products.json"
        activation_path = attempt / "diagnostics" / "activation_statistics.json"
        events_path = attempt / "events.jsonl"
        metrics = _read_json(metrics_path)
        condition = metrics["condition"]
        condition_id = condition["id"]
        if not _matches_family(condition_id, scale, family):
            continue

        manifest = _read_json(manifest_path)
        logical = _read_json(logical_path)
        activation = _read_json(activation_path)
        training = [
            event for event in _event_rows(events_path) if event.get("event") == "train"
        ]
        if manifest.get("status") != "completed" or manifest["condition"]["id"] != condition_id:
            raise ValueError(f"Incomplete or mismatched manifest: {attempt.name}")
        if metrics["training"]["completed_steps"] != TRAINING_STEPS:
            raise ValueError(f"Incomplete training steps: {attempt.name}")
        if metrics["training"]["input_tokens"] != TRAINING_TOKENS:
            raise ValueError(f"Training-token mismatch: {attempt.name}")
        if len(training) != TRAINING_STEPS:
            raise ValueError(f"Incomplete boundary log: {attempt.name}")
        if any(event["gradient_overflow"] or event["optimizer_step_skipped"] for event in training):
            raise ValueError(f"Overflow or skipped update: {attempt.name}")
        if any(not math.isfinite(float(event["task_loss"])) for event in training):
            raise ValueError(f"Non-finite task loss: {attempt.name}")

        _require_coverage(metrics["validation"]["final"], f"{attempt.name}/final")
        _require_coverage(logical["coverage"], f"{attempt.name}/logical")
        measured = logical["measured"]
        zero_count = int(measured["block_zero_product_count"])
        product_count = int(measured["model_product_count"])
        r_model = float(measured["R_model"])
        _close(zero_count / product_count, r_model, tolerance=1e-16)
        if sum(
            int(value["zero_product_count"])
            for value in measured["per_operation"].values()
        ) != int(measured["block_zero_product_count"]):
            raise ValueError(f"Logical zero counts do not reconcile: {attempt.name}")

        pooled = {row["name"]: row for row in activation["pooled_by_site"]}
        if set(pooled) != set(SITES):
            raise ValueError(f"Per-site activation rows are incomplete: {attempt.name}")
        site_exact_zero = {}
        for site in SITES:
            site_row = pooled[site]
            count = int(site_row["exact_zero_count"])
            total = int(site_row["total"])
            fraction = float(site_row["exact_zero_fraction"])
            _close(count / total, fraction, tolerance=1e-15)
            site_exact_zero[site] = {
                "exact_zero_count": count,
                "total_count": total,
                "exact_zero_fraction": fraction,
            }

        rows.append(
            {
                "scale": scale,
                "family": family,
                "condition_id": condition_id,
                "attempt_id": attempt.name,
                "kappa": float(condition["gate_threshold"]),
                "validation_loss": float(logical["coverage"]["loss"]),
                "execution_validation_loss": float(metrics["validation"]["final"]["loss"]),
                "R_model": r_model,
                "R_model_max": float(
                    logical["architecture_maximum"]["R_model_max_fraction"]
                ),
                "logical_counts": {
                    "zero_product_count": zero_count,
                    "model_product_count": product_count,
                },
                "site_exact_zero": site_exact_zero,
                "conflict_steps": sum(bool(event["gradient_conflict"]) for event in training),
                "projection_steps": sum(bool(event["projection_applied"]) for event in training),
                "median_tokens_per_second": float(
                    metrics["training"]["median_tokens_per_second"]
                ),
                "source_files": {
                    _repo_path(path): _sha256(path)
                    for path in (
                        manifest_path,
                        metrics_path,
                        logical_path,
                        activation_path,
                        events_path,
                    )
                },
            }
        )

    rows.sort(key=lambda row: row["kappa"])
    if tuple(row["kappa"] for row in rows) != KAPPAS:
        raise ValueError(f"Incomplete {scale} {family} kappa grid")
    return rows


def _teal_rows(path: Path, scale: str) -> list[dict[str, Any]]:
    payload = _read_json(path)
    raw_points = payload["conditions"] if scale == "14M" else payload["points"]
    id_map = {
        "14M": {"gelu-control": "A0", "relu-control": "A1-H"},
        "70M": {"a0-gelu": "A0", "a1h-relu": "A1-H"},
        "410M": {"a0-gelu": "A0", "a1h-relu": "A1-H"},
    }[scale]
    rows = []
    for point in raw_points:
        condition_id = point["condition_id"]
        if condition_id not in id_map:
            continue
        _require_coverage(
            point["validation"], f"{scale}/{condition_id}/{point['target_sparsity']}"
        )
        logical = point["logical_products"]
        zero_count = int(logical["block_zero_product_count"])
        product_count = int(logical["model_product_count"])
        r_model = float(logical["R_model"])
        _close(zero_count / product_count, r_model, tolerance=1e-16)
        pooled = {
            row.get("site", row.get("name")): row
            for row in point["activations_by_site"]
        }
        if not {"a", "m", "h", "z"}.issubset(pooled) or not set(pooled).issubset(SITES):
            raise ValueError(
                f"Unexpected post-hoc site coverage: {scale}/{condition_id}/"
                f"{point['target_sparsity']}"
            )
        site_exact_zero = {}
        for site, site_row in pooled.items():
            count = int(site_row["exact_zero_count"])
            total = int(site_row["total"])
            fraction = float(site_row["exact_zero_fraction"])
            _close(count / total, fraction, tolerance=1e-15)
            site_exact_zero[site] = {
                "exact_zero_count": count,
                "total_count": total,
                "exact_zero_fraction": fraction,
            }
        rows.append(
            {
                "scale": scale,
                "control": id_map[condition_id],
                "condition_id": condition_id,
                "target_sparsity": float(point["target_sparsity"]),
                "validation_loss": float(point["validation"]["loss"]),
                "R_model": r_model,
                "logical_counts": {
                    "zero_product_count": zero_count,
                    "model_product_count": product_count,
                },
                "site_exact_zero": site_exact_zero,
            }
        )
    rows.sort(key=lambda row: (row["control"], row["target_sparsity"]))
    for control in CONTROLS:
        grid = tuple(
            row["target_sparsity"] for row in rows if row["control"] == control
        )
        if grid != TARGETS:
            raise ValueError(f"Incomplete {scale} {control} TEAL grid")
    return rows


def _ranks(values: Iterable[float]) -> list[float]:
    values = list(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = float(rank)
    return ranks


def _series_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first, last = rows[0], rows[-1]
    r_values = [row["R_model"] for row in rows]
    losses = [row["validation_loss"] for row in rows]
    return {
        "scale": first["scale"],
        "family": first["family"],
        "delta_R_model_percentage_points_kappa_0_to_0p5": 100.0
        * (last["R_model"] - first["R_model"]),
        "delta_validation_loss_kappa_0_to_0p5": last["validation_loss"]
        - first["validation_loss"],
        "pearson_R_model_vs_loss": correlation(r_values, losses),
        "spearman_R_model_vs_loss": correlation(_ranks(r_values), _ranks(losses)),
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = (
        left["validation_loss"] <= right["validation_loss"]
        and left["R_model"] >= right["R_model"]
    )
    strict = (
        left["validation_loss"] < right["validation_loss"]
        or left["R_model"] > right["R_model"]
    )
    return no_worse and strict


def _verify_run019_controls(teal: list[dict[str, Any]]) -> None:
    verification = _read_json(RUN_019 / "artifacts" / "verification.json")
    if verification.get("status") != "verified" or verification.get("condition_count") != 12:
        raise ValueError("Run 019 terminal verification is not valid and complete")
    verified = {row["condition"]["id"]: row for row in verification["conditions"]}
    id_map = {"A0": "a0-gelu", "A1-H": "a1h-relu"}
    for control, condition_id in id_map.items():
        zero = next(
            row
            for row in teal
            if row["scale"] == "410M"
            and row["control"] == control
            and row["target_sparsity"] == 0.0
        )
        _close(zero["validation_loss"], float(verified[condition_id]["final_validation_loss"]))
        _close(zero["R_model"], float(verified[condition_id]["R_model"]), tolerance=1e-16)


def build_figure_data() -> dict[str, Any]:
    a0_gradient_norms, a0_gradient_summaries = _a0_gradient_rows()
    series = []
    for run_dir, scale, family in (
        (RUN_015, "14M", "A4-OL1"),
        (RUN_014, "14M", "A7-OL1"),
        (RUN_018, "70M", "A4-OL1"),
        (RUN_018, "70M", "A7-OL1"),
        (RUN_019, "410M", "A4-OL1"),
        (RUN_019, "410M", "A7-OL1"),
    ):
        series.extend(_trained_rows(run_dir, scale, family))
    teal = (
        _teal_rows(TEAL_14, "14M")
        + _teal_rows(TEAL_70, "70M")
        + _teal_rows(TEAL_410, "410M")
    )
    _verify_run019_controls(teal)

    grouped = {
        (scale, family): [
            row
            for row in series
            if row["scale"] == scale and row["family"] == family
        ]
        for scale in SCALES
        for family in FAMILIES
    }
    summaries = [
        _series_summary(grouped[(scale, family)])
        for scale in SCALES
        for family in FAMILIES
    ]
    persistence = {}
    for scale in SCALES:
        a4 = grouped[(scale, "A4-OL1")]
        a7 = grouped[(scale, "A7-OL1")]
        persistence[scale] = {
            "A4_dominates_A7_at_kappa_0": _dominates(a4[0], a7[0]),
            "A7_dominates_A4_at_kappa_0": _dominates(a7[0], a4[0]),
            "A7_dominates_A4_at_kappa_0p5": _dominates(a7[-1], a4[-1]),
            "A7_kappa_0p1_delta_R_model_percentage_points_from_kappa_0": 100.0
            * (a7[3]["R_model"] - a7[0]["R_model"]),
            "A7_kappa_0p1_delta_validation_loss_from_kappa_0": a7[3][
                "validation_loss"
            ]
            - a7[0]["validation_loss"],
        }

    source_paths = (
        RUN_014 / "artifacts" / "verification.json",
        RUN_015 / "artifacts" / "verification.json",
        RUN_018 / "artifacts" / "verification.json",
        RUN_019 / "artifacts" / "verification.json",
        TEAL_14,
        TEAL_70,
        TEAL_410,
        *(A0_EVENT_PATHS[scale] for scale in SCALES),
    )
    return {
        "schema_version": 3,
        "status": "complete_verified_analysis",
        "question": (
            "How does the selected R_model versus validation-loss structure at "
            "Pythia-410M compare with Pythia-14M and 70M, and how do the A0 "
            "global task-gradient norms evolve before and after clipping?"
        ),
        "coverage": {"documents": 500, **COVERAGE, "seed_count_per_scale": 1},
        "endpoint_pairing": (
            "validation loss and R_model come from the same eager full-validation "
            "logical-product pass"
        ),
        "trained_endpoints": sorted(
            series, key=lambda row: (SCALES.index(row["scale"]), row["family"], row["kappa"])
        ),
        "teal_points": sorted(
            teal,
            key=lambda row: (
                SCALES.index(row["scale"]),
                row["control"],
                row["target_sparsity"],
            ),
        ),
        "a0_gradient_norms": a0_gradient_norms,
        "a0_gradient_summaries": a0_gradient_summaries,
        "series_summaries": summaries,
        "persistence_checks": persistence,
        "sources": {_repo_path(path): _sha256(path) for path in source_paths},
        "interpretation": {
            "R_model": "logical zero-product opportunity, not runtime speedup",
            "comparison": "descriptive one-seed scale persistence, not a scaling law",
            "lines": "dose-order guides, not fitted response curves",
            "teal": "evaluation-only clipping, not trained intervention endpoints",
            "training_trajectory": (
                "mean causal-language-model task loss and global full-model "
                "task-gradient L2 norms at each optimizer boundary; gradient norms "
                "are not normalized by parameter count"
            ),
        },
    }


def _percentage(fraction: float, digits: int = 3) -> str:
    value = 100.0 * fraction
    if value == 0:
        return f"{value:.{digits}f}"
    if value < 0.001:
        return "<0.001"
    return f"{value:.{digits}f}"


def _site_percentage(row: dict[str, Any], site: str) -> str:
    site_row = row["site_exact_zero"].get(site)
    return "n.m." if site_row is None else _percentage(site_row["exact_zero_fraction"])


def _result_row(
    label: str, dose: str, row: dict[str, Any], *, include_scale: bool = False
) -> str:
    prefix = f"| {row['scale']} | " if include_scale else "| "
    return (
        f"{prefix}{label} | {dose} | {row['validation_loss']:.6f} | "
        f"{_percentage(row['R_model'], 4)} | "
        f"{_site_percentage(row, 'h')} | {_site_percentage(row, 'm')} | "
        f"{_site_percentage(row, 'a')} | {_site_percentage(row, 'z')} | "
        f"{_site_percentage(row, 'q_post')} | {_site_percentage(row, 'k_post')} | "
        f"{_site_percentage(row, 'v')} |"
    )


def table_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Pythia selected-ladder results through 410M",
        "",
        "Complete numeric tables supporting Analysis 011. Percentages use count-first pooling over the complete validation workload.",
        "",
        "## Metric and coverage conventions",
        "",
        "| Item | Reporting contract |",
        "|---|---|",
        "| Validation | All 500 documents; 338 complete 2,048-token blocks; 692,224 input tokens; 1,444-token tail excluded and recorded |",
        "| Endpoint pairing | Loss and `R_model` come from the same eager logical-product pass |",
        "| Site mass | Exact-zero percentage: pooled `count(x == 0) / count(x)` across all layers and complete validation blocks |",
        "| q and k | Post-RoPE tensors (`q_post`, `k_post`), the actual QK operands |",
        "| `R_model` | Count-pooled zero-operand logical-product opportunity over the model denominator; not measured speedup |",
        "| TEAL | Evaluation-only uniform clipping at `a,m,h,z`; `p` is common target sparsity and delta loss is paired to the same scale/control at `p=0` |",
        "| n.m. | Not measured in the source artifact; no value is inferred from `R_model` |",
        "",
        "## Pythia-410M selected endpoints",
        "",
        "| Condition | Dose | Loss | R_model (%) | h zero (%) | m zero (%) | a zero (%) | z zero (%) | q zero (%) | k zero (%) | v zero (%) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for control in CONTROLS:
        row = next(
            item
            for item in data["teal_points"]
            if item["scale"] == "410M"
            and item["control"] == control
            and item["target_sparsity"] == 0.0
        )
        lines.append(_result_row(control, "p=0", row))
    for family in FAMILIES:
        for row in data["trained_endpoints"]:
            if row["scale"] == "410M" and row["family"] == family:
                lines.append(_result_row(family, f"kappa={row['kappa']:g}", row))

    lines.extend(
        [
            "",
            "## Three-scale trained endpoints",
            "",
            "| Scale | Family | kappa | Loss | R_model (%) | h zero (%) | m zero (%) | a zero (%) | z zero (%) | q zero (%) | k zero (%) | v zero (%) |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for family in FAMILIES:
        for kappa in KAPPAS:
            for scale in SCALES:
                row = next(
                    item
                    for item in data["trained_endpoints"]
                    if item["scale"] == scale
                    and item["family"] == family
                    and item["kappa"] == kappa
                )
                lines.append(
                    _result_row(family, f"{kappa:g}", row, include_scale=True)
                )

    lines.extend(
        [
            "",
            "## Endpoint-span summaries",
            "",
            "| Scale | Family | Delta R_model, kappa 0 to 0.5 (pp) | Delta loss | Pearson r | Spearman rho |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in data["series_summaries"]:
        lines.append(
            f"| {row['scale']} | {row['family']} | "
            f"{row['delta_R_model_percentage_points_kappa_0_to_0p5']:.4f} | "
            f"{row['delta_validation_loss_kappa_0_to_0p5']:+.6f} | "
            f"{row['pearson_R_model_vs_loss']:.4f} | "
            f"{row['spearman_R_model_vs_loss']:.4f} |"
        )

    for control in CONTROLS:
        lines.extend(
            [
                "",
                f"## {control} post-hoc TEAL frontier",
                "",
                "| p | Scale | Loss | Delta loss | R_model (%) | h zero (%) | m zero (%) | a zero (%) | z zero (%) | q zero (%) | k zero (%) | v zero (%) |",
                "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        selected = [row for row in data["teal_points"] if row["control"] == control]
        baseline = {
            scale: next(
                row["validation_loss"]
                for row in selected
                if row["scale"] == scale and row["target_sparsity"] == 0.0
            )
            for scale in SCALES
        }
        for target in TARGETS:
            for scale in SCALES:
                row = next(
                    item
                    for item in selected
                    if item["scale"] == scale and item["target_sparsity"] == target
                )
                lines.append(
                    f"| {target:.1f} | {scale} | {row['validation_loss']:.6f} | "
                    f"{row['validation_loss'] - baseline[scale]:+.6f} | "
                    f"{_percentage(row['R_model'], 4)} | "
                    f"{_site_percentage(row, 'h')} | {_site_percentage(row, 'm')} | "
                    f"{_site_percentage(row, 'a')} | {_site_percentage(row, 'z')} | "
                    f"{_site_percentage(row, 'q_post')} | {_site_percentage(row, 'k_post')} | "
                    f"{_site_percentage(row, 'v')} |"
                )

    lines.extend(
        [
            "",
            "## A0 training and global task-gradient clipping summary",
            "",
            "| Scale | Optimizer boundaries | Initial task loss | Final task loss | Minimum task loss | Clipped boundaries | Clipped (%) | Minimum pre-clip L2 | Maximum pre-clip L2 | Maximum post-clip L2 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in data["a0_gradient_summaries"]:
        lines.append(
            f"| {row['scale']} | {row['boundaries']} | "
            f"{row['initial_task_loss']:.6f} | {row['final_task_loss']:.6f} | "
            f"{row['minimum_task_loss']:.6f} | {row['clipped_boundaries']} | "
            f"{100.0 * row['clipped_fraction']:.1f} | "
            f"{row['minimum_pre_clip_norm']:.6f} | "
            f"{row['maximum_pre_clip_norm']:.6f} | "
            f"{row['maximum_post_clip_norm']:.6f} |"
        )

    lines.extend(
        [
            "",
            "Run 019 records all eight diagnostic sites for TEAL. Earlier 14M/70M TEAL artifacts record only the four clipping sites, so downstream q/k/v cells remain `n.m.` rather than being reconstructed.",
            "",
        ]
    )
    return "\n".join(lines)


def _annotate(
    axis: Any,
    row: dict[str, float],
    label: str,
    offset: tuple[float, float],
    color: str,
) -> None:
    axis.annotate(
        label,
        xy=(100.0 * row["R_model"], row["validation_loss"]),
        xytext=offset,
        textcoords="offset points",
        fontsize=7.7,
        fontweight="bold",
        color=color,
        ha="left",
        va="center",
        bbox={
            "boxstyle": "round,pad=0.11",
            "facecolor": "white",
            "edgecolor": "none",
            "alpha": 0.86,
        },
        zorder=12,
    )


def _style_axis(axis: Any) -> None:
    axis.grid(True, color="#D8D8D8", linewidth=0.65, alpha=0.72)
    axis.set_axisbelow(True)
    axis.tick_params(direction="out", length=3.5, width=0.8)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color("#555555")
    axis.spines["bottom"].set_color("#555555")


def _axis_exit_point(
    last_visible: dict[str, Any], first_offscale: dict[str, Any], y_max: float
) -> dict[str, float]:
    loss_span = first_offscale["validation_loss"] - last_visible["validation_loss"]
    if loss_span <= 0:
        raise ValueError("Off-scale frontier point does not cross the upper axis limit")
    fraction = (y_max - last_visible["validation_loss"]) / loss_span
    return {
        "R_model": last_visible["R_model"]
        + fraction * (first_offscale["R_model"] - last_visible["R_model"]),
        "validation_loss": y_max,
    }


def _visible_prefix(rows: list[dict[str, Any]], y_max: float) -> list[dict[str, Any]]:
    visible = [row for row in rows if row["validation_loss"] <= y_max]
    if rows[: len(visible)] != visible:
        raise ValueError("A trajectory leaves and re-enters the visible y range")
    return visible


def render_figure(
    data: dict[str, Any], scales: tuple[str, ...], output: Path, title: str
) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 10.3,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
            "pdf.compression": 9,
        }
    )
    trained_styles = {
        "A4-OL1": {
            "label": "A4-OL1 trained ladder",
            "color": "#6F4C9B",
            "marker": "h",
            "linestyle": "--",
        },
        "A7-OL1": {
            "label": "A7-OL1 trained ladder",
            "color": "#56B4E9",
            "marker": "p",
            "linestyle": "--",
        },
    }
    posthoc_styles = {
        "A0": {
            "label": "Post-hoc clipping on A0",
            "color": "#CC79A7",
            "marker": "X",
            "linestyle": ":",
        },
        "A1-H": {
            "label": "Post-hoc clipping on A1-H",
            "color": "#222222",
            "marker": "v",
            "linestyle": (0, (4.0, 1.5, 1.0, 1.5)),
        },
    }
    control_styles = {
        "A0": {"marker": "P", "color": "#777777"},
        "A1-H": {"marker": "^", "color": "#222222"},
    }
    scale_styles = {
        "14M": {
            "face": "series",
            "edge": "white",
            "edgewidth": 0.9,
            "linewidth": 1.75,
            "alpha": 0.78,
        },
        "70M": {
            "face": "white",
            "edge": "series",
            "edgewidth": 1.35,
            "linewidth": 2.05,
            "alpha": 0.88,
        },
        "410M": {
            "face": "series",
            "edge": "#151515",
            "edgewidth": 1.15,
            "linewidth": 2.4,
            "alpha": 0.98,
        },
    }
    y_max = 6.0
    figure, axis = plt.subplots(figsize=(14.5, 6.7))

    trained_groups = {}
    for scale_index, scale in enumerate(scales):
        for family in FAMILIES:
            rows = [
                row
                for row in data["trained_endpoints"]
                if row["scale"] == scale and row["family"] == family
            ]
            trained_groups[(scale, family)] = rows
            family_style = trained_styles[family]
            scale_style = scale_styles[scale]
            face = family_style["color"] if scale_style["face"] == "series" else "white"
            edge = family_style["color"] if scale_style["edge"] == "series" else scale_style["edge"]
            axis.plot(
                [100.0 * row["R_model"] for row in rows],
                [row["validation_loss"] for row in rows],
                color=family_style["color"],
                marker=family_style["marker"],
                linestyle=family_style["linestyle"],
                linewidth=scale_style["linewidth"],
                markersize=8.4,
                markerfacecolor=face,
                markeredgecolor=edge,
                markeredgewidth=scale_style["edgewidth"],
                alpha=scale_style["alpha"],
                clip_on=True,
                zorder=6 + scale_index,
            )

    teal_groups = {}
    control_endpoints = {}
    for scale_index, scale in enumerate(scales):
        for control in CONTROLS:
            rows = [
                row
                for row in data["teal_points"]
                if row["scale"] == scale and row["control"] == control
            ]
            teal_groups[(scale, control)] = rows
            family_style = posthoc_styles[control]
            scale_style = scale_styles[scale]
            face = family_style["color"] if scale_style["face"] == "series" else "white"
            edge = family_style["color"] if scale_style["edge"] == "series" else scale_style["edge"]
            axis.plot(
                [100.0 * row["R_model"] for row in rows],
                [row["validation_loss"] for row in rows],
                color=family_style["color"],
                linestyle=family_style["linestyle"],
                linewidth=scale_style["linewidth"],
                alpha=scale_style["alpha"],
                clip_on=True,
                zorder=3 + scale_index,
            )
            axis.scatter(
                [100.0 * row["R_model"] for row in rows[1:]],
                [row["validation_loss"] for row in rows[1:]],
                facecolor=face,
                edgecolor=edge,
                marker=family_style["marker"],
                s=54,
                linewidth=scale_style["edgewidth"],
                alpha=scale_style["alpha"],
                clip_on=True,
                zorder=7 + scale_index,
            )
            endpoint = rows[0]
            control_endpoints[(scale, control)] = endpoint
            endpoint_style = control_styles[control]
            endpoint_face = (
                endpoint_style["color"] if scale_style["face"] == "series" else "white"
            )
            endpoint_edge = (
                endpoint_style["color"] if scale_style["edge"] == "series" else scale_style["edge"]
            )
            axis.scatter(
                [100.0 * endpoint["R_model"]],
                [endpoint["validation_loss"]],
                facecolor=endpoint_face,
                edgecolor=endpoint_edge,
                marker=endpoint_style["marker"],
                s=73,
                linewidth=scale_style["edgewidth"],
                zorder=10 + scale_index,
            )

    if scales == ("410M",):
        _annotate(axis, control_endpoints[("410M", "A0")], "A0", (8, -13), "#777777")
        _annotate(axis, control_endpoints[("410M", "A1-H")], "A1-H", (8, 11), "#222222")

    posthoc_offsets = {
        ("14M", "A0"): (-46, -15),
        ("14M", "A1-H"): (7, -15),
        ("70M", "A0"): (-48, -15),
        ("70M", "A1-H"): (7, -15),
        ("410M", "A0"): (-55, -15),
        ("410M", "A1-H"): (7, -15),
    }
    for key, rows in teal_groups.items():
        scale, control = key
        visible = _visible_prefix(rows, y_max)
        if len(visible) < len(rows):
            dose = rows[len(visible)]["target_sparsity"]
            point = _axis_exit_point(visible[-1], rows[len(visible)], y_max)
        else:
            dose = rows[-1]["target_sparsity"]
            point = rows[-1]
        # Scale is already encoded by marker fill/edge.  Repeating it at every
        # axis-exit label makes the three-scale panel materially harder to read.
        label = rf"$p={dose:.1f}$"
        _annotate(axis, point, label, posthoc_offsets[key], posthoc_styles[control]["color"])

    trained_offsets = {
        ("14M", "A4-OL1"): (-42, 12),
        ("14M", "A7-OL1"): (-50, -14),
        ("70M", "A4-OL1"): (-58, 12),
        ("70M", "A7-OL1"): (-58, -14),
        ("410M", "A4-OL1"): (-64, 12),
        ("410M", "A7-OL1"): (-68, -14),
    }
    for key, rows in trained_groups.items():
        scale, family = key
        visible = _visible_prefix(rows, y_max)
        if len(visible) < len(rows):
            dose = rows[len(visible)]["kappa"]
            point = _axis_exit_point(visible[-1], rows[len(visible)], y_max)
        else:
            dose = rows[-1]["kappa"]
            point = rows[-1]
        label = rf"$\kappa={dose:g}$"
        _annotate(axis, point, label, trained_offsets[key], trained_styles[family]["color"])

    axis.set_xlim(-1.0, 84.0)
    axis.set_ylim(4.05, y_max)
    axis.set_xlabel(r"Measured $R_{\mathrm{model}}$ (%)")
    axis.set_ylabel("Paired validation loss (lower is better)")
    _style_axis(axis)

    handles = []
    for control in CONTROLS:
        style = posthoc_styles[control]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.2,
                markersize=6.5,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=style["label"],
            )
        )
    for family in FAMILIES:
        style = trained_styles[family]
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=2.25,
                markersize=7.0,
                markeredgecolor="white",
                markeredgewidth=0.8,
                label=style["label"],
            )
        )
    if len(scales) > 1:
        for scale in scales:
            style = scale_styles[scale]
            face = "#666666" if style["face"] == "series" else "white"
            edge = "#666666" if style["edge"] == "series" else style["edge"]
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#777777",
                    marker="o",
                    linestyle="none",
                    markerfacecolor=face,
                    markeredgecolor=edge,
                    markeredgewidth=style["edgewidth"],
                    markersize=6.5,
                    label={
                        "14M": "14M (filled, white edge)",
                        "70M": "70M (open)",
                        "410M": "410M (filled, dark edge)",
                    }[scale],
                )
            )

    figure.suptitle(title, x=0.5, y=0.979, fontsize=14.0, fontweight="bold")
    figure.text(
        0.5,
        0.94,
        "A0/A1-H curves use post-hoc TEAL clipping; A4-OL1 and A7-OL1 are trained ladders",
        ha="center",
        va="center",
        fontsize=9.4,
        color="#444444",
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.908),
        ncol=4,
        frameon=False,
        handlelength=2.5,
        columnspacing=1.05,
        labelspacing=0.72,
    )
    figure.text(
        0.5,
        0.022,
        "Points above loss 6 are retained and clipped by the axis limit; trajectories exiting the panel continue off-scale.\n"
        r"Lines connect dose/target order only. $R_{\mathrm{model}}$ is exact-zero logical-product opportunity, not measured speedup; one seed per scale.",
        ha="center",
        va="bottom",
        fontsize=7.9,
        color="#444444",
        linespacing=1.32,
    )
    figure.subplots_adjust(left=0.075, right=0.988, top=0.75, bottom=0.155)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="pdf",
        bbox_inches="tight",
        metadata={"Creator": "Analysis 011", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def render_a0_gradient_figure(data: dict[str, Any], output: Path) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FixedLocator, FuncFormatter, MultipleLocator

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.2,
            "axes.labelsize": 10.3,
            "legend.fontsize": 8.7,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "pdf.fonttype": 42,
            "pdf.compression": 9,
        }
    )
    loss_color = "#333333"
    before_color = "#CC79A7"
    after_color = "#56B4E9"
    threshold_color = "#555555"
    figure, axes = plt.subplots(
        2,
        3,
        figsize=(14.5, 8.2),
        sharex=True,
        sharey="row",
        gridspec_kw={"height_ratios": (1.0, 1.0)},
    )
    summaries = {row["scale"]: row for row in data["a0_gradient_summaries"]}

    for column, scale in enumerate(SCALES):
        rows = [row for row in data["a0_gradient_norms"] if row["scale"] == scale]
        tokens_billions = [row["input_tokens_seen"] / 1e9 for row in rows]
        loss_axis = axes[0, column]
        gradient_axis = axes[1, column]
        loss_axis.plot(
            tokens_billions,
            [row["task_loss"] for row in rows],
            color=loss_color,
            linewidth=1.15,
            alpha=0.94,
            zorder=5,
        )
        gradient_axis.plot(
            tokens_billions,
            [row["gradient_norm_pre_clip"] for row in rows],
            color=before_color,
            linewidth=1.05,
            alpha=0.86,
            zorder=5,
        )
        gradient_axis.plot(
            tokens_billions,
            [row["gradient_norm_post_clip"] for row in rows],
            color=after_color,
            linestyle="--",
            linewidth=1.15,
            alpha=0.98,
            zorder=6,
        )
        gradient_axis.axhline(
            1.0,
            color=threshold_color,
            linestyle=(0, (2.0, 2.0)),
            linewidth=1.0,
            alpha=0.82,
            zorder=3,
        )
        summary = summaries[scale]
        loss_axis.text(
            0.965,
            0.945,
            f"final loss {summary['final_task_loss']:.3f}",
            transform=loss_axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#D0D0D0",
                "linewidth": 0.6,
                "alpha": 0.92,
            },
            zorder=10,
        )
        gradient_axis.text(
            0.965,
            0.945,
            f"clipped {summary['clipped_boundaries']}/712 "
            f"({100.0 * summary['clipped_fraction']:.1f}%)",
            transform=gradient_axis.transAxes,
            ha="right",
            va="top",
            fontsize=8.0,
            color="#333333",
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": "white",
                "edgecolor": "#D0D0D0",
                "linewidth": 0.6,
                "alpha": 0.92,
            },
            zorder=10,
        )
        loss_axis.set_title(
            f"Pythia-{scale}", fontsize=10.7, fontweight="bold", pad=8
        )
        loss_axis.set_xlim(0.0, 1.52)
        loss_axis.set_ylim(3.7, 11.3)
        loss_axis.xaxis.set_major_locator(MultipleLocator(0.5))
        loss_axis.yaxis.set_major_locator(MultipleLocator(1.0))
        gradient_axis.set_yscale("log")
        gradient_axis.set_ylim(0.16, 32.0)
        gradient_axis.xaxis.set_major_locator(MultipleLocator(0.5))
        gradient_axis.yaxis.set_major_locator(
            FixedLocator([0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
        )
        gradient_axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _: f"{value:g}")
        )
        for axis in (loss_axis, gradient_axis):
            _style_axis(axis)
            axis.grid(
                True, which="major", color="#D8D8D8", linewidth=0.65, alpha=0.72
            )

    axes[0, 0].set_ylabel("Training task loss")
    axes[1, 0].set_ylabel(r"Global task-gradient $L_2$ norm (log scale)")
    axes[1, 1].set_xlabel("Training tokens seen (billions)")
    handles = [
        Line2D([0], [0], color=loss_color, linewidth=2.0, label="Training task loss"),
        Line2D([0], [0], color=before_color, linewidth=2.0, label="Before clipping"),
        Line2D(
            [0],
            [0],
            color=after_color,
            linestyle="--",
            linewidth=2.0,
            label="After clipping",
        ),
        Line2D(
            [0],
            [0],
            color=threshold_color,
            linestyle=(0, (2.0, 2.0)),
            linewidth=1.3,
            label="Global clip threshold = 1",
        ),
    ]
    figure.suptitle(
        "A0 optimization trajectories through one MiniPile pass",
        x=0.5,
        y=0.975,
        fontsize=14.0,
        fontweight="bold",
    )
    figure.text(
        0.5,
        0.925,
        "Training task loss and global full-model L2 gradient norm at every optimizer boundary",
        ha="center",
        va="center",
        fontsize=9.4,
        color="#444444",
    )
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.895),
        ncol=4,
        frameon=False,
        handlelength=2.7,
        columnspacing=1.5,
    )
    figure.text(
        0.5,
        0.018,
        "Raw boundary values; no smoothing. Each boundary contains 2,097,152 tokens. "
        "All 712 updates completed without overflow or skipping.\n"
        "The norm is not parameter-count normalized; identical global clipping at 1.0 therefore need not have identical effects across scales.",
        ha="center",
        va="bottom",
        fontsize=7.9,
        color="#444444",
        linespacing=1.32,
    )
    figure.subplots_adjust(
        left=0.073, right=0.988, top=0.82, bottom=0.125, wspace=0.08, hspace=0.12
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        format="pdf",
        bbox_inches="tight",
        metadata={"Creator": "Analysis 011", "CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def main() -> None:
    data = build_figure_data()
    _write_json(FIGURE_DATA, data)
    _write_text(TABLES, table_markdown(data))
    render_figure(
        data,
        ("410M",),
        FIGURE_410,
        "Pythia-410M: trained frontiers and post-hoc clipping controls",
    )
    render_figure(
        data,
        SCALES,
        FIGURE_ALL,
        "Pythia-14M, 70M, and 410M: trained and post-hoc frontiers",
    )
    render_a0_gradient_figure(data, FIGURE_A0_GRADIENT)
    print(
        f"wrote {len(data['trained_endpoints'])} trained endpoints, "
        f"{len(data['teal_points'])} TEAL points, "
        f"{len(data['a0_gradient_norms'])} A0 gradient rows, and 3 figures"
    )


if __name__ == "__main__":
    main()
