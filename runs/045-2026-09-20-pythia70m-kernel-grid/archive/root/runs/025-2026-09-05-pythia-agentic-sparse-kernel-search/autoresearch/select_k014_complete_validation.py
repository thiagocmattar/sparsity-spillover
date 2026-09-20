"""Select static K014 layer masks independently for three 70M topologies."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PLAN = HERE / "candidates/k014/SEARCH.json"
CONDITIONS = {
    "A1-H": ("70m/a1h",),
    "A4-OL1": ("70m/a4-0", "70m/a4-0p5"),
    "A7-OL1": ("70m/a7-0", "70m/a7-0p5"),
}
SHORTS = {
    "70m/a1h": "a1h",
    "70m/a4-0": "a4-0",
    "70m/a4-0p5": "a4-0p5",
    "70m/a7-0": "a7-0",
    "70m/a7-0p5": "a7-0p5",
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def geometric_mean(values):
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("Speedups must be positive and finite")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def artifact_name(identifier, mask):
    return f"k014full-70m-{SHORTS[identifier]}-{mask}-rtxpro4500-002"


def collect_one(artifacts, identifier, mask):
    directory = artifacts / artifact_name(identifier, mask)
    needed = [
        directory / "manifest.json",
        directory / "development-quality.json",
        directory / "timing.json",
        directory / "full-validation.json",
        directory / "status.json",
    ]
    if not all(path.exists() for path in needed):
        status = read(directory / "status.json") if (directory / "status.json").exists() else {}
        return {
            "qualified": False,
            "artifact": directory.name,
            "error": status.get("error", "missing-or-incomplete"),
        }
    manifest, quality, timing, validation, status = map(read, needed)
    expected_implementation = f"k014-{mask}"
    if manifest["checkpoint"]["id"] != identifier:
        raise ValueError(f"Checkpoint mismatch for {identifier}/{mask}")
    arguments = manifest["arguments"]
    if (
        arguments["implementation"] != expected_implementation
        or arguments["sites"] != "active"
        or arguments["full_validation"] is not True
        or manifest["checkpoint"].get("partition") != "development"
    ):
        raise ValueError(f"Protocol identity mismatch for {identifier}/{mask}")
    qualified = (
        all(quality["pass"].values())
        and validation.get("complete") is True
        and validation.get("blocks") == 338
        and validation.get("excluded_tail_tokens") == 1444
        and all(validation["pass"].values())
        and status.get("stage") == "complete"
    )
    speedup = timing["matched"]["eager"]["summary"]["candidate"][
        "paired_geomean_speedup"
    ]
    return {
        "qualified": qualified,
        "speedup": speedup,
        "R_model": manifest["checkpoint"]["canonical_logical_products"]["measured"]["R_model"],
        "validation_loss_native": validation["loss"]["native"],
        "validation_loss_candidate": validation["loss"]["candidate"],
        "validation_loss_delta": validation["loss_delta"]["candidate"],
        "artifact": directory.name,
        "error": None if qualified else "complete-validation-gate-failed",
    }


def collect_mask(artifacts, topology, mask, layers):
    rows = {
        identifier: collect_one(artifacts, identifier, mask)
        for identifier in CONDITIONS[topology]
    }
    eligible = all(row["qualified"] for row in rows.values())
    score = geometric_mean([row["speedup"] for row in rows.values()]) if eligible else None
    return {
        "mask": mask,
        "selected_layers": len(layers),
        "layer_indices": layers,
        "eligible": eligible,
        "score": score,
        "conditions": rows,
    }


def choose(rows, tolerance=0.005):
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return None
    best = max(row["score"] for row in eligible)
    tied = [row for row in eligible if best - row["score"] <= tolerance]
    return min(tied, key=lambda row: (row["selected_layers"], row["mask"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=RUN / "artifacts")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = read(PLAN)
    masks = plan["masks"]
    results = {}
    for topology in CONDITIONS:
        rows = [
            collect_mask(args.artifacts, topology, mask, plan["mask_layers"][mask])
            for mask in masks
        ]
        results[topology] = {"masks": rows, "winner": choose(rows)}
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "search_id": plan["search_id"],
        "scientific_status": plan["scientific_status"],
        "selection_rule": plan["selection_rule"],
        "search_used_only_development": True,
        "fresh_holdout_available": False,
        "topologies": results,
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=False)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
