"""Matched comparison of Run 008 threshold-only and Run 010 all-site OL1."""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parents[1]
SOURCES = {
    "run008": REPO_ROOT
    / "runs/008-2026-08-29-pythia14m-a7-z-post-mixed-threshold-local/artifacts/verification.json",
    "run010": REPO_ROOT
    / "runs/010-2026-08-30-pythia14m-a7-z-post-mixed-threshold-ol1-local/artifacts/verification.json",
}
SITES = ("a", "m", "h", "q_post", "k_post", "v", "z")
IDENTITY_FIELDS = (
    "initial_parameter_sha256",
    "training_schedule_sha256",
    "total_optimizer_steps",
    "total_training_input_tokens",
    "complete_validation_passes",
)


def load_verified(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != "verified" or value.get("evidence_label") != "valid":
        raise RuntimeError(f"Source is not valid verified evidence: {path}")
    return value


def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = (
        left["final_validation_loss"] <= right["final_validation_loss"]
        and left["R_model"] >= right["R_model"]
    )
    strict = (
        left["final_validation_loss"] < right["final_validation_loss"]
        or left["R_model"] > right["R_model"]
    )
    return no_worse and strict


def main() -> Path:
    evidence = {name: load_verified(path) for name, path in SOURCES.items()}
    identity = {field: evidence["run008"].get(field) for field in IDENTITY_FIELDS}
    for field, expected in identity.items():
        if evidence["run010"].get(field) != expected:
            raise RuntimeError(f"Matched identity differs at {field}.")

    by_run = {
        name: {float(row["condition"]["gate_threshold"]): row for row in value["conditions"]}
        for name, value in evidence.items()
    }
    if set(by_run["run008"]) != set(by_run["run010"]):
        raise RuntimeError("The runs do not share one kappa grid.")

    rows = []
    all_points = []
    for kappa in sorted(by_run["run008"]):
        baseline = by_run["run008"][kappa]
        ol1 = by_run["run010"][kappa]
        for run_name, source in (("run008", baseline), ("run010", ol1)):
            all_points.append(
                {
                    "run": run_name,
                    "kappa": kappa,
                    "final_validation_loss": float(source["final_validation_loss"]),
                    "R_model": float(source["R_model"]),
                }
            )
        if dominates(ol1, baseline):
            paired = "run010_dominates"
        elif dominates(baseline, ol1):
            paired = "run008_dominates"
        else:
            paired = "tradeoff"
        rows.append(
            {
                "kappa": kappa,
                "run008": _metrics(baseline),
                "run010": _metrics(ol1),
                "delta_run010_minus_run008": {
                    "final_validation_loss": float(ol1["final_validation_loss"])
                    - float(baseline["final_validation_loss"]),
                    "R_block": float(ol1["R_block"]) - float(baseline["R_block"]),
                    "R_model": float(ol1["R_model"]) - float(baseline["R_model"]),
                    "exact_zero_fraction": {
                        site: float(ol1["selected_site_exact_zero_fractions"][site])
                        - float(baseline["selected_site_exact_zero_fractions"][site])
                        for site in SITES
                    },
                },
                "paired_quality_R_model_relation": paired,
            }
        )

    frontier = [
        point
        for point in all_points
        if not any(dominates(other, point) for other in all_points if other is not point)
    ]
    result = {
        "schema_version": 1,
        "question": "How does seven-site OL1 change Run 008 at each matched kappa?",
        "delta_sign": "run010 minus run008; lower validation loss and higher R_model are favorable",
        "matched_identity": identity,
        "source_files": {
            name: {
                "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for name, path in SOURCES.items()
        },
        "rows": rows,
        "joint_quality_R_model_frontier": sorted(
            frontier, key=lambda row: (row["R_model"], row["final_validation_loss"])
        ),
        "limits": [
            "one seed and one model scale",
            "paired endpoint estimates have no replicate uncertainty",
            "the user-directed interpretation treats first- and second-decimal differences as prone to realization noise; arithmetic dominance is not evidence of material superiority",
            "R_model is logical-product opportunity, not measured runtime speedup",
            "joint-site OL1 effects are not individually attributable",
        ],
    }
    output = ANALYSIS_DIR / "comparison.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(output)
    print(output)
    return output


def _metrics(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "final_validation_loss": float(row["final_validation_loss"]),
        "R_block": float(row["R_block"]),
        "R_model": float(row["R_model"]),
        "exact_zero_fraction": {
            site: float(row["selected_site_exact_zero_fractions"][site]) for site in SITES
        },
    }


if __name__ == "__main__":
    main()
