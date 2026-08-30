"""Reduce Run 004/009 boundary logs into gradient-interaction evidence."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
RUNS = {
    "naive_l1": REPO_ROOT / "runs/004-2026-08-29-pythia14m-full-pass-l1n",
    "ol1": REPO_ROOT / "runs/009-2026-08-30-pythia14m-full-pass-ol1",
}
METHODS = {"naive_l1": "l1_naive", "ol1": "orthogonal_l1"}
LAMBDAS = (0.05, 0.1, 0.5, 1.0)
EXPECTED_STEPS = 712


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _finite_float(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise RuntimeError(f"Non-finite {name}: {value!r}")
    return result


def _quantile(values: list[float], q: float) -> float:
    if not values or not 0.0 <= q <= 1.0:
        raise ValueError("A quantile requires finite values and q in [0, 1].")
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (position - lower) * (ordered[upper] - ordered[lower])


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values or not all(math.isfinite(value) for value in values):
        raise RuntimeError("Cannot summarize an empty or non-finite series.")
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "minimum": min(values),
        "q05": _quantile(values, 0.05),
        "q25": _quantile(values, 0.25),
        "median": _quantile(values, 0.50),
        "q75": _quantile(values, 0.75),
        "q95": _quantile(values, 0.95),
        "maximum": max(values),
    }


def _lambda_key(value: float) -> str:
    return f"{value:g}"


def _load_events(
    run_key: str,
) -> tuple[
    dict[float, list[dict[str, Any]]],
    dict[str, dict[str, str]],
    dict[str, str],
]:
    run_dir = RUNS[run_key]
    verification_path = run_dir / "artifacts/verification.json"
    verification = _json(verification_path)
    if verification.get("status") != "verified" or verification.get("evidence_label") != "valid":
        raise RuntimeError(f"Run is not verified valid evidence: {run_dir}")

    selected: dict[float, list[dict[str, Any]]] = {}
    sources: dict[str, dict[str, str]] = {
        "verification": {
            "path": str(verification_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _sha256(verification_path),
        }
    }
    for condition_row in verification["conditions"]:
        condition = condition_row["condition"]
        if condition["pressure_method"] != METHODS[run_key]:
            continue
        if (
            condition["is_control"]
            or condition["activation"] != "relu"
            or condition["pressure_sites"] != ["h"]
        ):
            raise RuntimeError(f"Unexpected pressure condition: {condition!r}")
        if run_key == "ol1" and not math.isclose(
            float(condition["step_budget"]), 1.0, abs_tol=1e-12
        ):
            raise RuntimeError(f"Unexpected OL1 trust budget: {condition!r}")
        weight = _finite_float(condition["pressure_weight"], "pressure weight")
        if weight in selected:
            raise RuntimeError(f"Duplicate lambda for {run_key}: {weight}")
        attempt_id = condition_row["attempt_id"]
        events_path = run_dir / "artifacts/attempts" / attempt_id / "events.jsonl"
        events = [
            json.loads(line)
            for line in events_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        train = [row for row in events if row.get("event") == "train"]
        if len(train) != EXPECTED_STEPS:
            raise RuntimeError(f"Expected {EXPECTED_STEPS} train events: {events_path}")
        if [int(row["step"]) for row in train] != list(range(1, EXPECTED_STEPS + 1)):
            raise RuntimeError(f"Non-contiguous train steps: {events_path}")
        for row in train:
            if row["condition_id"] != condition["id"]:
                raise RuntimeError(f"Condition mismatch in {events_path}")
            if not math.isclose(float(row["pressure_weight"]), weight, abs_tol=1e-12):
                raise RuntimeError(f"Pressure-weight mismatch in {events_path}")
            if row["optimizer_step_skipped"] or row["gradient_overflow"]:
                raise RuntimeError(f"Skipped or overflowing boundary in {events_path}")
        selected[weight] = train
        sources[condition["id"]] = {
            "path": str(events_path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _sha256(events_path),
        }

    if set(selected) != set(LAMBDAS):
        raise RuntimeError(f"Unexpected lambda grid for {run_key}: {sorted(selected)}")
    identity = {
        field: str(verification[field])
        for field in ("initial_parameter_sha256", "training_schedule_sha256")
    }
    return selected, sources, identity


def _raw_interaction(rows: list[dict[str, Any]], weight: float) -> dict[str, Any]:
    dots = [_finite_float(row["task_pressure_gradient_dot"], "raw gradient dot") for row in rows]
    cosines = [
        _finite_float(row["task_pressure_gradient_cosine"], "raw gradient cosine")
        for row in rows
    ]
    task_norms = [_finite_float(row["task_gradient_norm"], "task gradient norm") for row in rows]
    if any(norm <= 0.0 for norm in task_norms):
        raise RuntimeError("Task-gradient norms must be positive.")

    combined_task_dots = [
        task_norm * task_norm + weight * dot
        for task_norm, dot in zip(task_norms, dots, strict=True)
    ]
    negative_interference = [
        -weight * dot / (task_norm * task_norm)
        for task_norm, dot in zip(task_norms, dots, strict=True)
        if dot < 0.0
    ]
    return {
        "gradient_dot": _summary(dots),
        "gradient_cosine": _summary(cosines),
        "negative_dot_count": sum(dot < 0.0 for dot in dots),
        "negative_dot_fraction": sum(dot < 0.0 for dot in dots) / len(dots),
        "first_boundary_cosine": cosines[0],
        "task_combined_gradient_dot": _summary(combined_task_dots),
        "task_combined_negative_dot_count": sum(value < 0.0 for value in combined_task_dots),
        "negative_component_task_alignment_offset_fraction": _summary(negative_interference),
    }


def _ol1_interaction(rows: list[dict[str, Any]]) -> dict[str, Any]:
    before_dots = [_finite_float(row["task_pressure_dot_before"], "direction dot") for row in rows]
    before_cosines = [
        _finite_float(row["task_pressure_cosine_before"], "direction cosine") for row in rows
    ]
    projected = [bool(row["projection_applied"]) for row in rows]
    if projected != [dot < 0.0 for dot in before_dots]:
        raise RuntimeError("OL1 projection decisions do not match direction-dot signs.")

    projected_after_cosines = [
        abs(_finite_float(row["task_pressure_cosine_after"], "post-projection cosine"))
        for row in rows
        if row["projection_applied"]
    ]
    trust_scales = [_finite_float(row["trust_scale"], "trust scale") for row in rows]
    raw_ratios = [
        _finite_float(row["pressure_to_task_ratio_raw"], "raw pressure/task ratio")
        for row in rows
    ]
    final_ratios = [
        _finite_float(row["pressure_to_task_ratio_final"], "final pressure/task ratio")
        for row in rows
    ]
    if any(ratio > 1.0 + 1e-9 for ratio in final_ratios):
        raise RuntimeError("OL1 final ratio exceeds the declared trust budget.")
    return {
        "task_pressure_dot_before": _summary(before_dots),
        "task_pressure_cosine_before": _summary(before_cosines),
        "negative_direction_dot_count": sum(dot < 0.0 for dot in before_dots),
        "projection_count": sum(projected),
        "projected_post_cosine_absolute": _summary(projected_after_cosines),
        "trust_scale": _summary(trust_scales),
        "trust_cap_count": sum(scale < 1.0 for scale in trust_scales),
        "pressure_to_task_ratio_raw": _summary(raw_ratios),
        "pressure_to_task_ratio_final": _summary(final_ratios),
    }


def _markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Gradient-interaction tables",
        "",
        "Each condition contains 712 complete optimizer boundaries. Conflict fractions",
        "count boundaries before dividing; quantiles are over boundary-level scalar",
        "diagnostics. The task and pressure gradients in the first two tables are",
        "separate, unweighted gradients computed before naive L1 combines and globally",
        "clips them.",
        "",
        "## Naive-L1 raw task-pressure interaction",
        "",
        "| lambda | negative dot (n/712) | negative dot (%) | cosine q05 | "
        "cosine q25 | median cosine | cosine q75 | cosine q95 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for weight in LAMBDAS:
        row = result["raw_interaction"]["naive_l1"][_lambda_key(weight)]
        cosine = row["gradient_cosine"]
        lines.append(
            f"| {weight:g} | {row['negative_dot_count']}/712 | "
            f"{100.0 * row['negative_dot_fraction']:.2f} | {cosine['q05']:+.6f} | "
            f"{cosine['q25']:+.6f} | {cosine['median']:+.6f} | "
            f"{cosine['q75']:+.6f} | {cosine['q95']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "## Naive-L1 combined raw-gradient alignment",
            "",
            "For a boundary with a negative task-pressure dot, the alignment offset is",
            "`-lambda * <g_task,g_pressure> / ||g_task||^2`. Thus",
            "`<g_task,g_task + lambda*g_pressure> = ||g_task||^2 * (1 - offset)`.",
            "The offset describes the raw pre-clip direction; it is not a realized loss",
            "change or an AdamW-step measurement.",
            "",
            "| lambda | combined raw dot < 0 (n/712) | negative-pressure "
            "boundaries | median alignment offset (%) | q95 offset (%) | maximum offset (%) |",
            "| ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for weight in LAMBDAS:
        row = result["raw_interaction"]["naive_l1"][_lambda_key(weight)]
        offset = row["negative_component_task_alignment_offset_fraction"]
        lines.append(
            f"| {weight:g} | {row['task_combined_negative_dot_count']}/712 | "
            f"{offset['count']} | {100.0 * offset['median']:.3f} | "
            f"{100.0 * offset['q95']:.3f} | {100.0 * offset['maximum']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## OL1 AdamW-relative interaction and trust cap",
            "",
            "The adaptive quantities compare `d_task = m_hat/D` with",
            "`d_pressure = g_pressure/D`, where `D = sqrt(v_hat) + adam_eps` uses",
            "task-only AdamW state. A negative dot triggers projection. The residual",
            "post-projection cosine is reported only for projected boundaries.",
            "",
            "| lambda | negative adaptive dot / projected (n/712) | cosine-before "
            "q05 | median cosine before | cosine-before q95 | max abs projected "
            "cosine after | trust cap active (n/712) | median raw ratio |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for weight in LAMBDAS:
        row = result["ol1_adaptive_interaction"][_lambda_key(weight)]
        cosine = row["task_pressure_cosine_before"]
        residual = row["projected_post_cosine_absolute"]
        lines.append(
            f"| {weight:g} | {row['negative_direction_dot_count']}/712 | "
            f"{cosine['q05']:+.6f} | {cosine['median']:+.6f} | {cosine['q95']:+.6f} | "
            f"{residual['maximum']:.3e} | {row['trust_cap_count']}/712 | "
            f"{row['pressure_to_task_ratio_raw']['median']:.6f} |"
        )

    lines.extend(
        [
            "",
            "All eight conditions begin from the same initialization and have",
            f"first-boundary raw cosines within {result['first_boundary_raw_cosine_span']:.3e}.",
            "Because lambda does not weight the recorded component gradients, later",
            "differences in conflict describe lambda-dependent training trajectories,",
            "not an algebraic change in angle from multiplying by lambda.",
            "",
            "The OL1 projection constrains alignment with the task-only adaptive",
            "direction. It does not guarantee non-increase of the current task loss or",
            "validation loss, and the trust budget bounds rather than eliminates",
            "hyperparameter sensitivity.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> tuple[Path, Path]:
    events: dict[str, dict[float, list[dict[str, Any]]]] = {}
    sources: dict[str, Any] = {}
    identities: dict[str, dict[str, str]] = {}
    for run_key in RUNS:
        events[run_key], sources[run_key], identities[run_key] = _load_events(run_key)
    if identities["naive_l1"] != identities["ol1"]:
        raise RuntimeError(f"Run identities are not matched: {identities!r}")

    raw = {
        run_key: {
            _lambda_key(weight): _raw_interaction(events[run_key][weight], weight)
            for weight in LAMBDAS
        }
        for run_key in RUNS
    }
    adaptive = {
        _lambda_key(weight): _ol1_interaction(events["ol1"][weight])
        for weight in LAMBDAS
    }
    first_cosines = [
        raw[run_key][_lambda_key(weight)]["first_boundary_cosine"]
        for run_key in RUNS
        for weight in LAMBDAS
    ]

    result = {
        "schema_version": 1,
        "evidence_status": "complete_verified_matched_cohorts",
        "question": (
            "How do raw task-pressure interactions motivate OL1, and what does "
            "OL1 guarantee operationally?"
        ),
        "coverage": {
            "methods": ["l1_naive", "orthogonal_l1"],
            "lambda_grid": list(LAMBDAS),
            "optimizer_boundaries_per_condition": EXPECTED_STEPS,
            "total_naive_l1_boundaries": EXPECTED_STEPS * len(LAMBDAS),
            "total_ol1_boundaries": EXPECTED_STEPS * len(LAMBDAS),
            "matched_identity": identities["naive_l1"],
        },
        "raw_interaction": raw,
        "ol1_adaptive_interaction": adaptive,
        "first_boundary_raw_cosine_span": max(first_cosines) - min(first_cosines),
        "sources": sources,
        "interpretation": {
            "raw_conflict": (
                "A negative raw dot means the isolated pressure descent component "
                "has a positive first-order task-loss contribution."
            ),
            "combined_gradient": (
                "The complete naive-L1 raw gradient remained task-aligned at every "
                "recorded boundary."
            ),
            "ol1": (
                "OL1 removes only the component that opposes the task-only AdamW "
                "adaptive direction and caps the correction-to-task direction ratio."
            ),
        },
        "nonclaims": [
            "raw component conflict does not show that the complete naive-L1 "
            "AdamW step increases task loss",
            "AdamW-relative non-opposition does not guarantee non-increase of "
            "training or validation loss",
            "the projection is a minimum-change feasible direction, not a globally "
            "largest loss-safe pressure step",
            "the trust budget bounds pressure magnitude but does not make OL1 hyperparameter-free",
            "one seed and one Pythia-14M scale do not establish behavior at larger "
            "scales or under other recipes",
        ],
    }
    artifact_path = ANALYSIS_DIR / "gradient_interaction.json"
    table_path = ANALYSIS_DIR / "gradient_tables.md"
    _write_text_atomic(artifact_path, json.dumps(result, indent=2, sort_keys=True) + "\n")
    _write_text_atomic(table_path, _markdown(result))
    print(artifact_path)
    print(table_path)
    return artifact_path, table_path


if __name__ == "__main__":
    main()
