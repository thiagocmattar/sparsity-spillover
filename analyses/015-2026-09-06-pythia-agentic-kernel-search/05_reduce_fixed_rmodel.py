#!/usr/bin/env python3
"""Reduce fresh-process fixed-R_model baseline-versus-optimized evidence."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = (
    HERE.parents[1]
    / "runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
    / "retrieved/rtxpro4500-004/verification.json"
)
DEFAULT_PHASE18_INPUT = (
    HERE.parents[1]
    / "runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
    / "retrieved/rtxpro4500-004/phase18-verification.json"
)
PHASE17_SPECS = {
    "14m/a4-0p5": ("14M", "A4-OL1", 0.5, "p0", "k013"),
    "14m/a7-0p5": ("14M", "A7-OL1", 0.5, "p0", "k013"),
    "70m/a4-0p5": ("70M", "A4-OL1", 0.5, "k009", "k016"),
    "70m/a7-0p5": ("70M", "A7-OL1", 0.5, "k009", "k016"),
    "410m/a4-0p5": ("410M", "A4-OL1", 0.5, "k004", "k010"),
    "410m/a7-0p5": ("410M", "A7-OL1", 0.5, "k004", "k010"),
}
PHASE18_SPECS = {
    "14m/a4-0p5": ("14M", "A4-OL1", 0.5, "p0", "k001"),
    "14m/a7-0p5": ("14M", "A7-OL1", 0.5, "p0", "k001"),
}
# Retained for import compatibility with the initial reducer tests.
PAIR_SPECS = PHASE17_SPECS
REPEAT = re.compile(r"^fixed-r([123])-")
PHASE18_REPEAT = re.compile(r"^k001fixed-r([123])-")


def architecture_summary(pair_rows: list[dict]) -> list[dict]:
    architecture_rows = []
    ordered_sizes = [size for size in ("14M", "70M", "410M") if any(
        row["model_size"] == size for row in pair_rows
    )]
    for size in ordered_sizes:
        selected = [row for row in pair_rows if row["model_size"] == size and row["qualified_pair"]]
        ratios = [row["optimized_over_baseline"] for row in selected]
        if not ratios:
            raise ValueError(f"No qualified pairs for {size}")
        architecture_rows.append({
            "model_size": size,
            "qualified_pairs": len(ratios),
            "median_optimized_over_baseline": statistics.median(ratios),
            "min_optimized_over_baseline": min(ratios),
            "max_optimized_over_baseline": max(ratios),
            "optimized_faster_pairs": sum(value > 1.0 for value in ratios),
        })
    return architecture_rows


def reduce_rows(
    rows: list[dict],
    pair_specs: dict = PHASE17_SPECS,
    repeat_pattern: re.Pattern = REPEAT,
) -> dict:
    expected_processes = len(pair_specs) * 2 * 3
    if len(rows) != expected_processes:
        raise ValueError(f"Expected {expected_processes} process rows, found {len(rows)}")
    grouped: dict[str, dict[int, dict[str, dict]]] = defaultdict(lambda: defaultdict(dict))
    for row in rows:
        condition = row["condition"]
        if condition not in pair_specs:
            raise ValueError(f"Unexpected condition: {condition}")
        match = repeat_pattern.match(row["label"])
        if match is None:
            raise ValueError(f"Unparseable process label: {row['label']}")
        repeat = int(match.group(1))
        implementation = row["implementation"]
        if implementation in grouped[condition][repeat]:
            raise ValueError(f"Duplicate process: {condition} r{repeat} {implementation}")
        grouped[condition][repeat][implementation] = row

    condition_rows = []
    pair_rows = []
    for condition, (size, family, kappa, baseline, optimized) in pair_specs.items():
        baseline_values = []
        optimized_values = []
        ratios = []
        qualified_ratios = []
        rmodels = set()
        for repeat in (1, 2, 3):
            implementations = grouped[condition][repeat]
            if set(implementations) != {baseline, optimized}:
                raise ValueError(f"Incomplete pair: {condition} r{repeat}")
            baseline_row = implementations[baseline]
            optimized_row = implementations[optimized]
            baseline_speedup = float(baseline_row["paired_geomean_speedup"])
            optimized_speedup = float(optimized_row["paired_geomean_speedup"])
            ratio = optimized_speedup / baseline_speedup
            qualified = bool(baseline_row["candidate_validation_pass"] and optimized_row["candidate_validation_pass"])
            baseline_values.append(baseline_speedup)
            optimized_values.append(optimized_speedup)
            ratios.append(ratio)
            if qualified:
                qualified_ratios.append(ratio)
            rmodels.update((float(baseline_row["R_model"]), float(optimized_row["R_model"])))
            pair_rows.append({
                "condition_id": condition,
                "model_size": size,
                "family": family,
                "kappa": kappa,
                "repeat": repeat,
                "baseline_implementation": baseline,
                "optimized_implementation": optimized,
                "R_model_fraction": float(baseline_row["R_model"]),
                "R_model_percent": 100.0 * float(baseline_row["R_model"]),
                "baseline_speedup": baseline_speedup,
                "optimized_speedup": optimized_speedup,
                "optimized_over_baseline": ratio,
                "baseline_validation_pass": bool(baseline_row["candidate_validation_pass"]),
                "optimized_validation_pass": bool(optimized_row["candidate_validation_pass"]),
                "qualified_pair": qualified,
            })
        if len(rmodels) != 1:
            raise ValueError(f"R_model differs within {condition}: {sorted(rmodels)}")
        if not qualified_ratios:
            raise ValueError(f"No qualified repeat for {condition}")
        condition_rows.append({
            "condition_id": condition,
            "model_size": size,
            "family": family,
            "kappa": kappa,
            "baseline_implementation": baseline,
            "optimized_implementation": optimized,
            "R_model_fraction": next(iter(rmodels)),
            "R_model_percent": 100.0 * next(iter(rmodels)),
            "baseline_median_speedup": statistics.median(baseline_values),
            "optimized_median_speedup": statistics.median(optimized_values),
            "median_optimized_over_baseline": statistics.median(qualified_ratios),
            "min_optimized_over_baseline": min(qualified_ratios),
            "max_optimized_over_baseline": max(qualified_ratios),
            "qualified_repeats": len(qualified_ratios),
            "optimized_faster_repeats": sum(value > 1.0 for value in qualified_ratios),
            "optimized_median_faster": statistics.median(qualified_ratios) > 1.0,
        })

    return {
        "process_rows": rows,
        "pair_rows": pair_rows,
        "condition_rows": condition_rows,
        "architecture_rows": architecture_summary(pair_rows),
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_tables(path: Path, reduction: dict, robustness_rows: list[dict]) -> None:
    lines = [
        "# Fresh-process fixed-R_model optimization", "",
        "All ratios compare independently launched processes at the identical checkpoint and canonical `R_model`.", "",
        "## Condition summaries", "",
        "| Size | Condition | R_model | Baseline | Optimized | Baseline speedup | Optimized speedup | Optimized / baseline | Repeat range | Wins |",
        "|---|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in reduction["condition_rows"]:
        lines.append(
            f"| {row['model_size']} | {row['family']} kappa={row['kappa']:g} | {row['R_model_percent']:.3f}% | "
            f"{row['baseline_implementation'].upper()} | {row['optimized_implementation'].upper()} | "
            f"{row['baseline_median_speedup']:.4f}x | {row['optimized_median_speedup']:.4f}x | "
            f"{row['median_optimized_over_baseline']:.4f}x | "
            f"[{row['min_optimized_over_baseline']:.4f}, {row['max_optimized_over_baseline']:.4f}] | "
            f"{row['optimized_faster_repeats']}/{row['qualified_repeats']} |"
        )
    lines += ["", "## Architecture summaries", "",
              "| Size | Qualified pairs | Median optimized / baseline | Range | Wins |",
              "|---|---:|---:|---:|---:|"]
    for row in reduction["architecture_rows"]:
        lines.append(
            f"| {row['model_size']} | {row['qualified_pairs']} | {row['median_optimized_over_baseline']:.4f}x | "
            f"[{row['min_optimized_over_baseline']:.4f}, {row['max_optimized_over_baseline']:.4f}] | "
            f"{row['optimized_faster_pairs']}/{row['qualified_pairs']} |"
        )
    lines += [
        "", "The range is the minimum-to-maximum across three process-paired repeats, not a confidence interval.",
        "The comparison changes implementation coverage/dispatch while holding the trained checkpoint and canonical logical opportunity fixed.",
        "", "## 14M final robustness-policy contrast", "",
        "K013 is the final all-checkpoint robustness policy, while K001 is the direct P0 kernel optimization used in the primary 14M comparison.", "",
        "| Condition | R_model | P0 speedup | K013 speedup | K013 / P0 | Repeat range |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in robustness_rows:
        lines.append(
            f"| {row['family']} kappa={row['kappa']:g} | {row['R_model_percent']:.3f}% | "
            f"{row['baseline_median_speedup']:.4f}x | {row['optimized_median_speedup']:.4f}x | "
            f"{row['median_optimized_over_baseline']:.4f}x | "
            f"[{row['min_optimized_over_baseline']:.4f}, {row['max_optimized_over_baseline']:.4f}] |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--phase18-input", type=Path, default=DEFAULT_PHASE18_INPUT)
    args = parser.parse_args()
    phase17_verification = json.loads(args.input.read_text(encoding="utf-8-sig"))
    phase18_verification = json.loads(args.phase18_input.read_text(encoding="utf-8-sig"))
    if phase17_verification.get("passed") is not True or phase18_verification.get("passed") is not True:
        raise ValueError("Archive verification did not pass")
    phase17 = reduce_rows(phase17_verification["rows"])
    phase18 = reduce_rows(
        phase18_verification["rows"], PHASE18_SPECS, PHASE18_REPEAT
    )
    primary_pairs = phase18["pair_rows"] + [
        row for row in phase17["pair_rows"] if row["model_size"] != "14M"
    ]
    primary_conditions = phase18["condition_rows"] + [
        row for row in phase17["condition_rows"] if row["model_size"] != "14M"
    ]
    robustness_rows = [
        row for row in phase17["condition_rows"] if row["model_size"] == "14M"
    ]
    reduction = {
        "process_rows": phase17["process_rows"] + phase18["process_rows"],
        "pair_rows": primary_pairs,
        "condition_rows": primary_conditions,
        "architecture_rows": architecture_summary(primary_pairs),
        "robustness_policy_rows": robustness_rows,
        "phase17": phase17,
        "phase18": phase18,
    }
    write_csv(HERE / "fixed-rmodel-process-pairs.csv", primary_pairs)
    write_csv(HERE / "fixed-rmodel-replication.csv", primary_conditions)
    write_csv(HERE / "fixed-rmodel-robustness-policy.csv", robustness_rows)
    (HERE / "fixed-rmodel-reduction.json").write_text(
        json.dumps(reduction, indent=2) + "\n", encoding="utf-8"
    )
    write_tables(HERE / "fixed-rmodel-tables.md", reduction, robustness_rows)
    print(json.dumps({
        "conditions": len(reduction["condition_rows"]),
        "process_pairs": len(reduction["pair_rows"]),
        "all_architecture_medians_above_one": all(
            row["median_optimized_over_baseline"] > 1.0
            for row in reduction["architecture_rows"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
