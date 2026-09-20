"""Select one κ-independent K015 A7 policy from complete-validation artifacts."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUN = HERE.parent
PLAN = HERE / "candidates/k015/SEARCH.json"
CONDITIONS = (("70m/a7-0", "a7-0"), ("70m/a7-0p5", "a7-0p5"))


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def geometric_mean(values):
    return math.exp(sum(math.log(value) for value in values) / len(values))


def collect_one(artifacts, identifier, short, variant):
    directory = artifacts / f"k015full-70m-{short}-{variant}-rtxpro4500-002"
    needed = [directory / name for name in (
        "manifest.json", "development-quality.json", "timing.json",
        "full-validation.json", "status.json",
    )]
    if not all(path.exists() for path in needed):
        status = read(directory / "status.json") if (directory / "status.json").exists() else {}
        return {"qualified": False, "artifact": directory.name,
                "error": status.get("error", "missing-or-incomplete")}
    manifest, quality, timing, validation, status = map(read, needed)
    if (
        manifest["checkpoint"]["id"] != identifier
        or manifest["checkpoint"].get("partition") != "development"
        or manifest["arguments"]["implementation"] != f"k015-{variant}"
        or manifest["arguments"]["full_validation"] is not True
    ):
        raise ValueError(f"K015 protocol identity mismatch for {identifier}/{variant}")
    qualified = (
        all(quality["pass"].values()) and validation.get("complete") is True
        and validation.get("blocks") == 338 and validation.get("excluded_tail_tokens") == 1444
        and all(validation["pass"].values()) and status.get("stage") == "complete"
    )
    return {
        "qualified": qualified,
        "speedup": timing["matched"]["eager"]["summary"]["candidate"]["paired_geomean_speedup"],
        "R_model": manifest["checkpoint"]["canonical_logical_products"]["measured"]["R_model"],
        "validation_loss_delta": validation["loss_delta"]["candidate"],
        "artifact": directory.name,
        "error": None if qualified else "complete-validation-gate-failed",
    }


def choose(rows, tolerance=0.005):
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        return None
    best = max(row["score"] for row in eligible)
    tied = [row for row in eligible if best - row["score"] <= tolerance]
    return min(tied, key=lambda row: (row["adapted_linear_count"], row["variant"]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, default=RUN / "artifacts")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    plan = read(PLAN)
    rows = []
    for variant, policy in plan["variants"].items():
        conditions = {
            identifier: collect_one(args.artifacts, identifier, short, variant)
            for identifier, short in CONDITIONS
        }
        eligible = all(row["qualified"] for row in conditions.values())
        rows.append({
            "variant": variant,
            "layers": policy["layers"],
            "sites": policy["sites"],
            "adapted_linear_count": len(policy["layers"]) * len(policy["sites"]),
            "eligible": eligible,
            "score": geometric_mean([row["speedup"] for row in conditions.values()]) if eligible else None,
            "conditions": conditions,
        })
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "search_id": plan["search_id"],
        "scientific_status": plan["scientific_status"],
        "selection_rule": plan["selection_rule"],
        "search_used_only_development": True,
        "fresh_holdout_available": False,
        "variants": rows,
        "winner": choose(rows),
    }
    text = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=False)
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
