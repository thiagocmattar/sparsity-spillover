"""Reduce the matched Run 004 naive-L1 and Run 009 OL1 endpoints."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
RUN004_DIR = REPO_ROOT / "runs/004-2026-08-29-pythia14m-full-pass-l1n"
RUN009_DIR = REPO_ROOT / "runs/009-2026-08-30-pythia14m-full-pass-ol1"
VERIFICATIONS = {
    "run004": RUN004_DIR / "artifacts/verification.json",
    "run009": RUN009_DIR / "artifacts/verification.json",
}
LAMBDAS = (0.05, 0.1, 0.5, 1.0)
SITES = ("h", "m", "q_post", "k_post", "v", "attention_output")
EXPECTED_COVERAGE = {
    "sequences": 338,
    "input_tokens": 692_224,
    "source_tokens": 693_668,
    "excluded_tail_tokens": 1_444,
    "complete_block_coverage": True,
}


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_text_atomic(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _load_verified(path: Path) -> dict[str, Any]:
    value = _json(path)
    if value.get("status") != "verified" or value.get("evidence_label") != "valid":
        raise RuntimeError(f"Source is not verified valid evidence: {path}")
    return value


def _require_close(actual: float, expected: float, name: str) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
        raise RuntimeError(f"{name} differs: {actual!r} != {expected!r}")


def _validate_coverage(value: dict[str, Any], name: str) -> None:
    for field, expected in EXPECTED_COVERAGE.items():
        if value.get(field) != expected:
            raise RuntimeError(f"{name}.{field} differs: {value.get(field)!r} != {expected!r}")


def _endpoint(
    run_name: str,
    run_dir: Path,
    verification_row: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, str]]]:
    attempt_id = verification_row["attempt_id"]
    base = run_dir / "artifacts/attempts" / attempt_id
    paths = {
        "metrics": base / "metrics.json",
        "activation_statistics": base / "diagnostics/activation_statistics.json",
        "logical_products": base / "diagnostics/logical_products.json",
    }
    metrics = _json(paths["metrics"])
    activation = _json(paths["activation_statistics"])
    logical = _json(paths["logical_products"])
    condition = verification_row["condition"]

    if metrics["condition"] != condition:
        raise RuntimeError(f"Condition differs between verification and metrics: {attempt_id}")
    if int(metrics["training"]["completed_steps"]) != 712:
        raise RuntimeError(f"Expected 712 completed boundaries: {attempt_id}")
    if int(metrics["training"]["input_tokens"]) != 1_493_172_224:
        raise RuntimeError(f"Training-token coverage differs: {attempt_id}")

    final_validation = metrics["validation"]["final"]
    _validate_coverage(final_validation, f"{attempt_id}.validation.final")
    _require_close(
        float(final_validation["loss"]),
        float(verification_row["final_validation_loss"]),
        f"{attempt_id}.final_validation_loss",
    )
    _validate_coverage(logical["coverage"], f"{attempt_id}.logical.coverage")

    pooled = {row["name"]: row for row in activation["pooled_by_site"]}
    if set(pooled) != set(SITES):
        raise RuntimeError(f"Activation sites differ for {attempt_id}: {sorted(pooled)}")
    site_exact: dict[str, float] = {}
    site_near: dict[str, float] = {}
    for site in SITES:
        row = pooled[site]
        total = int(row["total"])
        if total <= 0 or int(row["finite"]) != total or int(row["nonfinite"]) != 0:
            raise RuntimeError(f"Invalid activation coverage at {attempt_id}:{site}")
        exact = int(row["exact_zero_count"]) / total
        near = int(row["threshold_hits"]["0.001"]) / total
        _require_close(exact, float(row["exact_zero_fraction"]), f"{attempt_id}:{site}.exact")
        _require_close(
            near,
            float(row["threshold_fractions"]["0.001"]),
            f"{attempt_id}:{site}.near_1e-3",
        )
        site_exact[site] = exact
        site_near[site] = near

    measured = logical["measured"]
    r_model = int(measured["block_zero_product_count"]) / int(
        measured["model_product_count"]
    )
    _require_close(r_model, float(measured["R_model"]), f"{attempt_id}.R_model")

    maximum = logical["architecture_maximum"]
    expected_topology = "A0" if condition["id"] == "gelu-control" else "A1-H"
    expected_active_sites = [] if expected_topology == "A0" else ["h"]
    if (
        maximum["topology_id"] != expected_topology
        or maximum["active_sites"] != expected_active_sites
    ):
        raise RuntimeError(f"Unexpected architecture ceiling identity: {attempt_id}")
    r_model_max = int(maximum["reachable_product_count"]) / int(
        maximum["model_product_count"]
    )
    _require_close(
        r_model_max,
        float(maximum["R_model_max_fraction"]),
        f"{attempt_id}.R_model_max",
    )

    pressure_weight = float(condition["pressure_weight"])
    endpoint = {
        "run": run_name,
        "attempt_id": attempt_id,
        "condition_id": condition["id"],
        "activation": condition["activation"],
        "pressure_method": condition["pressure_method"],
        "lambda": None if condition["is_control"] else pressure_weight,
        "topology": expected_topology,
        "site_exact_zero_fraction": site_exact,
        "site_near_zero_fraction_1e-3": site_near,
        "R_model": r_model,
        "R_model_max": r_model_max,
        "final_validation_loss": float(final_validation["loss"]),
    }
    sources = {
        name: {
            "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": _sha256(path),
        }
        for name, path in paths.items()
    }
    return endpoint, sources


def _by_method_and_lambda(
    endpoints: list[dict[str, Any]], method: str
) -> dict[float, dict[str, Any]]:
    selected = {
        float(row["lambda"]): row
        for row in endpoints
        if row["pressure_method"] == method and row["lambda"] is not None
    }
    if set(selected) != set(LAMBDAS):
        raise RuntimeError(f"Unexpected lambda grid for {method}: {sorted(selected)}")
    return selected


def _percent_text(fraction: float) -> str:
    percent = 100.0 * fraction
    if percent == 0.0:
        return "0"
    if percent < 0.0001:
        return f"{percent:.3e}"
    return f"{percent:.6f}"


def _condition_text(row: dict[str, Any]) -> str:
    if row["condition_id"] == "gelu-control":
        return "GeLU control"
    if row["condition_id"] == "relu-control":
        return "ReLU control"
    if row["pressure_method"] == "l1_naive":
        return "Naive L1"
    if row["pressure_method"] == "orthogonal_l1":
        return "OL1"
    raise RuntimeError(f"Unknown condition: {row['condition_id']}")


def _lambda_text(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value:.2f}"


def _table_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Analysis 003 tables",
        "",
        "All percentages are count-first reductions from the reloaded final checkpoints.",
        "Activation statistics pool all six layers and all 338 complete validation",
        "blocks (692,224 input tokens); the 1,444-token tail is excluded. Run 009",
        "reuses Run 004's controls rather than rerunning them.",
        "",
        "## Exact-zero mass, measured R_model, and final validation loss",
        "",
        "| Run / condition | lambda | `h` zero (%) | `m` zero (%) | `q_post` zero (%) | `k_post` zero (%) | `v` zero (%) | attention output zero (%) | `R_model` (%) | final validation loss |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in result["conditions"]:
        exact = row["site_exact_zero_fraction"]
        lines.append(
            "| "
            + f"{row['run'].replace('run', 'Run ')} {_condition_text(row)} | "
            + f"{_lambda_text(row['lambda'])} | "
            + " | ".join(_percent_text(exact[site]) for site in SITES)
            + f" | {_percent_text(row['R_model'])} | {row['final_validation_loss']:.6f} |"
        )

    lines.extend(
        [
            "",
            "## Near-zero mass at |x| <= 1e-3",
            "",
            "| Run / condition | lambda | `h` (%) | `m` (%) | `q_post` (%) | `k_post` (%) | `v` (%) | attention output (%) |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["conditions"]:
        near = row["site_near_zero_fraction_1e-3"]
        lines.append(
            "| "
            + f"{row['run'].replace('run', 'Run ')} {_condition_text(row)} | "
            + f"{_lambda_text(row['lambda'])} | "
            + " | ".join(f"{100.0 * near[site]:.6f}" for site in SITES)
            + " |"
        )

    lines.extend(
        [
            "",
            "## Matched OL1 minus naive-L1 endpoint changes",
            "",
            "Positive zero-mass and `R_model` values mean OL1 is higher; negative loss",
            "values mean OL1 has lower validation loss.",
            "",
            "| lambda | `h` exact-zero change (pp) | `h` near-zero change (pp) | `R_model` change (pp) | final validation-loss change |",
            "| ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in result["matched_rows"]:
        delta = row["delta_ol1_minus_naive_l1"]
        lines.append(
            f"| {row['lambda']:.2f} | "
            f"{100.0 * delta['site_exact_zero_fraction']['h']:+.6f} | "
            f"{100.0 * delta['site_near_zero_fraction_1e-3']['h']:+.6f} | "
            f"{100.0 * delta['R_model']:+.6f} | "
            f"{delta['final_validation_loss']:+.6f} |"
        )

    lines.extend(
        [
            "",
            "`attention_output` is the output of `attention.dense` (`W_o`) before",
            "residual addition; it is not the pre-`W_o` context site `z`. `R_model`",
            "is an exact-zero logical-product opportunity, not measured speedup.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> tuple[Path, Path]:
    evidence = {name: _load_verified(path) for name, path in VERIFICATIONS.items()}
    for field in ("initial_parameter_sha256", "training_schedule_sha256"):
        if evidence["run004"][field] != evidence["run009"][field]:
            raise RuntimeError(f"Matched identity differs at {field}")

    endpoints: list[dict[str, Any]] = []
    endpoint_sources: dict[str, Any] = {}
    for run_name, run_dir in (("run004", RUN004_DIR), ("run009", RUN009_DIR)):
        for verification_row in evidence[run_name]["conditions"]:
            endpoint, sources = _endpoint(run_name, run_dir, verification_row)
            endpoints.append(endpoint)
            endpoint_sources[endpoint["condition_id"]] = sources

    if len(endpoints) != 10:
        raise RuntimeError(f"Expected ten unique endpoints, found {len(endpoints)}")
    run004_controls = [row for row in endpoints if row["run"] == "run004" and row["lambda"] is None]
    if {row["condition_id"] for row in run004_controls} != {"gelu-control", "relu-control"}:
        raise RuntimeError("Run 004 control identities differ")
    if any(row["lambda"] is None for row in endpoints if row["run"] == "run009"):
        raise RuntimeError("Run 009 unexpectedly reran a control")

    naive = _by_method_and_lambda(endpoints, "l1_naive")
    ol1 = _by_method_and_lambda(endpoints, "orthogonal_l1")
    matched_rows = []
    for pressure_weight in LAMBDAS:
        baseline = naive[pressure_weight]
        candidate = ol1[pressure_weight]
        matched_rows.append(
            {
                "lambda": pressure_weight,
                "naive_l1": baseline,
                "ol1": candidate,
                "delta_ol1_minus_naive_l1": {
                    "site_exact_zero_fraction": {
                        site: candidate["site_exact_zero_fraction"][site]
                        - baseline["site_exact_zero_fraction"][site]
                        for site in SITES
                    },
                    "site_near_zero_fraction_1e-3": {
                        site: candidate["site_near_zero_fraction_1e-3"][site]
                        - baseline["site_near_zero_fraction_1e-3"][site]
                        for site in SITES
                    },
                    "R_model": candidate["R_model"] - baseline["R_model"],
                    "final_validation_loss": candidate["final_validation_loss"]
                    - baseline["final_validation_loss"],
                },
            }
        )

    maxima = {
        row["R_model_max"] for row in endpoints if row["topology"] == "A1-H"
    }
    if len(maxima) != 1:
        raise RuntimeError("A1-H R_model_max differs across endpoints")

    result = {
        "schema_version": 1,
        "evidence_status": "complete_verified_matched_cohorts",
        "question": "How do Run 004 naive L1 and Run 009 OL1 compare at matched lambda after one full MiniPile pass?",
        "source_verifications": {
            name: {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": _sha256(path),
            }
            for name, path in VERIFICATIONS.items()
        },
        "matched_identity": {
            "initial_parameter_sha256": evidence["run004"]["initial_parameter_sha256"],
            "training_schedule_sha256": evidence["run004"]["training_schedule_sha256"],
            "optimizer_boundaries_per_condition": 712,
            "training_input_tokens_per_condition": 1_493_172_224,
            "validation": EXPECTED_COVERAGE,
            "topology": "A1-H",
            "gate": "ReLU at h",
            "pressure_site": "h",
            "lambda_grid": list(LAMBDAS),
        },
        "controls_reused_from_run004": [row["condition_id"] for row in run004_controls],
        "R_model_max": maxima.pop(),
        "conditions": endpoints,
        "matched_rows": matched_rows,
        "endpoint_sources": endpoint_sources,
        "limits": [
            "one seed and one Pythia-14M scale",
            "no replicate uncertainty; endpoint differences are descriptive",
            "independently scheduled Pods are scientifically matched but not bitwise identical",
            "R_model is logical-product opportunity, not runtime speedup",
            "attention_output is post-W_o and is not the pre-W_o z site",
            "no finding or manuscript claim is promoted by this analysis",
        ],
    }
    comparison_path = ANALYSIS_DIR / "comparison.json"
    tables_path = ANALYSIS_DIR / "tables.md"
    _write_text_atomic(
        comparison_path,
        json.dumps(result, indent=2, sort_keys=True) + "\n",
    )
    _write_text_atomic(tables_path, _table_markdown(result))
    print(comparison_path)
    print(tables_path)
    return comparison_path, tables_path


if __name__ == "__main__":
    main()
