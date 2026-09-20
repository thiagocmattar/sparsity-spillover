"""Select a K011 14M mask from complete-validation development artifacts."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PLAN = HERE / "candidates/k011/FULL_VALIDATION_SEARCH.json"
SHORTS = OrderedDict(
    (
        ("14m/a0", ("a0", "all")),
        ("14m/a1h", ("a1h", "h")),
        ("14m/a4-0", ("a4-0", "active")),
        ("14m/a4-0p5", ("a4-0p5", "active")),
        ("14m/a7-0", ("a7-0", "active")),
        ("14m/a7-0p5", ("a7-0p5", "active")),
    )
)
PRIMARY = ("14m/a4-0", "14m/a4-0p5", "14m/a7-0", "14m/a7-0p5")
LAYER_COUNTS = {
    "prefix1": 1,
    "suffix1": 1,
    "suffix2": 2,
    "suffix3": 3,
    "suffix4": 4,
    "odd": 3,
}


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def speedup(timing):
    return timing["matched"]["eager"]["summary"]["candidate"][
        "paired_geomean_speedup"
    ]


def geometric_mean(values):
    if not values or any(value <= 0 or not math.isfinite(value) for value in values):
        raise ValueError("Speedups must be positive and finite")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def resolve_directory(artifacts, short, mask):
    names = (
        f"k011full-14m-{short}-{mask}-rtxpro4500-002",
        f"k011retry-14m-{short}-{mask}-rtxpro4500-002",
    )
    complete = []
    abandoned = []
    for name in names:
        directory = artifacts / name
        if not directory.exists():
            continue
        status_path = directory / "status.json"
        status = read(status_path) if status_path.exists() else {}
        if status.get("stage") == "complete":
            complete.append(directory)
        else:
            abandoned.append(name)
    if len(complete) > 1:
        raise ValueError(f"Multiple complete artifacts for {short}/{mask}")
    return (complete[0] if complete else None), abandoned


def collect(artifacts, mask):
    conditions = {}
    errors = []
    abandoned_attempts = []
    for identifier, (short, sites) in SHORTS.items():
        directory, abandoned = resolve_directory(artifacts, short, mask)
        abandoned_attempts.extend(abandoned)
        if directory is None:
            errors.append(f"{identifier}:missing-or-skipped")
            continue
        needed = (
            directory / "manifest.json",
            directory / "development-quality.json",
            directory / "timing.json",
            directory / "full-validation.json",
            directory / "status.json",
        )
        if not all(path.exists() for path in needed):
            errors.append(f"{identifier}:missing-or-skipped")
            continue
        manifest, quality, timing, validation, status = map(read, needed)
        if manifest.get("partition") != "development":
            raise ValueError(f"Non-development artifact encountered for {identifier}")
        if manifest["checkpoint"]["id"] != identifier:
            raise ValueError(f"Checkpoint identity mismatch for {identifier}")
        if manifest["arguments"]["mask"] != mask or manifest["arguments"]["sites"] != sites:
            raise ValueError(f"Mask/site identity mismatch for {identifier}")
        qualified = (
            all(quality["pass"].values())
            and validation.get("complete") is True
            and validation.get("blocks") == 338
            and validation.get("excluded_tail_tokens") == 1444
            and all(validation["pass"].values())
            and status.get("stage") == "complete"
        )
        value = speedup(timing)
        conditions[identifier] = {
            "qualified": qualified,
            "speedup": value,
            "validation_loss_native": validation["loss"]["native"],
            "validation_loss_candidate": validation["loss"]["candidate"],
            "validation_loss_delta": validation["loss_delta"]["candidate"],
            "artifact": directory.name,
        }
        if not qualified:
            errors.append(f"{identifier}:unqualified")
    eligible = len(conditions) == len(SHORTS) and not errors
    score = geometric_mean([conditions[item]["speedup"] for item in PRIMARY]) if eligible else None
    return {
        "mask": mask,
        "selected_layers": LAYER_COUNTS[mask],
        "eligible": eligible,
        "primary_geometric_mean_speedup": score,
        "conditions": conditions,
        "errors": errors,
        "abandoned_attempts": sorted(abandoned_attempts),
    }


def choose(rows, tolerance=0.005):
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return None
    best_score = max(row["primary_geometric_mean_speedup"] for row in eligible)
    tied = [
        row
        for row in eligible
        if best_score - row["primary_geometric_mean_speedup"] <= tolerance
    ]
    return min(tied, key=lambda row: (row["selected_layers"], row["mask"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=RUN / "artifacts")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = read(PLAN)
    if plan["shortlist"] != list(LAYER_COUNTS):
        raise ValueError("Selector mask order differs from the predeclared search")
    rows = [collect(args.artifacts, mask) for mask in plan["shortlist"]]
    winner = choose(rows)
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "search_id": plan["search_id"],
        "selection_rule": plan["selection_rule"],
        "primary_conditions": list(PRIMARY),
        "masks": rows,
        "winner": winner,
        "heldout_inspected": False,
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
