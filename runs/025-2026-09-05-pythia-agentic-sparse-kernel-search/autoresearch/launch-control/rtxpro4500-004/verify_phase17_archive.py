#!/usr/bin/env python3
"""Verify the retrieved Phase 17 archive before GPU teardown."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path


RUN = "runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
ARTIFACTS = f"{RUN}/artifacts"
CONTROL = f"{ARTIFACTS}/phase17-fixed-rmodel-pairs-rtxpro4500-004"
PAIRS = {
    "14m-a4-0p5": ("p0", "k013", "14m/a4-0p5"),
    "14m-a7-0p5": ("p0", "k013", "14m/a7-0p5"),
    "70m-a4-0p5": ("k009", "k016", "70m/a4-0p5"),
    "70m-a7-0p5": ("k009", "k016", "70m/a7-0p5"),
    "410m-a4-0p5": ("k004", "k010", "410m/a4-0p5"),
    "410m-a7-0p5": ("k004", "k010", "410m/a7-0p5"),
}


def expected_processes() -> dict[str, tuple[str, str]]:
    rows = {}
    for repeat in (1, 2, 3):
        for short, (baseline, winner, condition) in PAIRS.items():
            for implementation in (baseline, winner):
                label = f"fixed-r{repeat}-{short}-{implementation}-rtxpro4500-004"
                rows[label] = (condition, implementation)
    return rows


def verify(path: Path) -> dict:
    expected = expected_processes()
    with tarfile.open(path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}

        def body(name: str) -> bytes:
            member = members.get(name)
            if member is None:
                raise ValueError(f"Missing archive member: {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Unreadable archive member: {name}")
            return stream.read()

        def parsed(name: str) -> dict:
            return json.loads(body(name))

        if body(f"{CONTROL}/exit-code.txt").strip() != b"0":
            raise ValueError("Phase controller did not exit successfully")
        body(f"{CONTROL}/all-checks-finished-utc.txt")
        order_lines = body(f"{CONTROL}/process-order.tsv").decode().splitlines()
        if len(order_lines) != 37:
            raise ValueError(f"Expected header plus 36 order rows, found {len(order_lines)}")

        rmodel_by_condition: dict[str, set[tuple[int, int, float]]] = defaultdict(set)
        checkpoint_by_condition: dict[str, set[tuple[tuple[str, int, str], ...]]] = defaultdict(set)
        candidate_passes = 0
        timed_processes = 0
        process_rows = []
        for label, (condition, implementation) in expected.items():
            prefix = f"{ARTIFACTS}/{label}"
            for required in (
                "manifest.json", "status.json", "development-quality.json",
                "full-validation.json", "timing.json", "timing-samples.jsonl",
            ):
                body(f"{prefix}/{required}")
            if body(f"{CONTROL}/{label}.exit-code.txt").strip() != b"0":
                raise ValueError(f"Probe process failed: {label}")
            manifest = parsed(f"{prefix}/manifest.json")
            status = parsed(f"{prefix}/status.json")
            full = parsed(f"{prefix}/full-validation.json")
            timing = parsed(f"{prefix}/timing.json")
            arguments = manifest["arguments"]
            if not (
                arguments["attempt"] == label
                and arguments["condition"] == condition
                and arguments["implementation"] == implementation
                and arguments["execution"] == ["eager"]
                and arguments["inputs"] == 16
                and arguments["passes"] == 5
                and arguments["full_validation"] is True
            ):
                raise ValueError(f"Argument contract mismatch: {label}")
            if status["stage"] != "complete":
                raise ValueError(f"Incomplete probe stage: {label}")
            if not (
                full["complete"] is True
                and full["blocks"] == 338
                and full["documents"] == 500
                and full["input_tokens"] == 692224
                and full["excluded_tail_tokens"] == 1444
            ):
                raise ValueError(f"Validation coverage mismatch: {label}")
            measured = manifest["checkpoint"]["canonical_logical_products"]["measured"]
            rmodel_by_condition[condition].add((
                int(measured["block_zero_product_count"]),
                int(measured["model_product_count"]),
                float(measured["R_model"]),
            ))
            files = tuple(
                sorted((row["path"], int(row["bytes"]), row["sha256"])
                       for row in manifest["checkpoint"]["files"])
            )
            checkpoint_by_condition[condition].add(files)
            passed = bool(full["pass"]["candidate"])
            candidate_passes += int(passed)
            samples = timing["matched"]["eager"]["summary"]["candidate"]["samples"]
            if samples != 80:
                raise ValueError(f"Expected 80 paired timing samples: {label}")
            timed_processes += 1
            process_rows.append({
                "label": label,
                "condition": condition,
                "implementation": implementation,
                "candidate_validation_pass": passed,
                "R_model": measured["R_model"],
                "paired_geomean_speedup": timing["matched"]["eager"]["summary"]["candidate"]["paired_geomean_speedup"],
            })
        for condition in sorted(rmodel_by_condition):
            if len(rmodel_by_condition[condition]) != 1:
                raise ValueError(f"R_model changed within matched condition: {condition}")
            if len(checkpoint_by_condition[condition]) != 1:
                raise ValueError(f"Checkpoint identity changed within matched condition: {condition}")

    return {
        "passed": True,
        "archive": str(path),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "processes": len(process_rows),
        "timed_processes": timed_processes,
        "candidate_validation_passes": candidate_passes,
        "fixed_rmodel_conditions": len(rmodel_by_condition),
        "rows": process_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive), indent=2))


if __name__ == "__main__":
    main()
