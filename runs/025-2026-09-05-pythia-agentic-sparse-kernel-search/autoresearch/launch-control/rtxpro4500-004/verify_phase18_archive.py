#!/usr/bin/env python3
"""Verify the adaptive P0-to-K001 confirmation in the complete archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path


RUN = "runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
ARTIFACTS = f"{RUN}/artifacts"
CONTROL = f"{ARTIFACTS}/phase18-14m-k001-confirmation-rtxpro4500-004"
CONDITIONS = {
    "14m-a4-0p5": "14m/a4-0p5",
    "14m-a7-0p5": "14m/a7-0p5",
}


def expected_processes() -> dict[str, tuple[str, str, int]]:
    rows = {}
    for repeat in (1, 2, 3):
        for short, condition in CONDITIONS.items():
            for implementation in ("p0", "k001"):
                label = f"k001fixed-r{repeat}-{short}-{implementation}-rtxpro4500-004"
                rows[label] = (condition, implementation, repeat)
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
            raise ValueError("Phase 18 controller did not exit successfully")
        body(f"{CONTROL}/all-checks-finished-utc.txt")
        if len(body(f"{CONTROL}/process-order.tsv").decode().splitlines()) != 13:
            raise ValueError("Phase 18 process-order inventory is incomplete")

        identities: dict[str, set[tuple[int, int, float]]] = defaultdict(set)
        checkpoints: dict[str, set[tuple[tuple[str, int, str], ...]]] = defaultdict(set)
        rows = []
        for label, (condition, implementation, repeat) in expected.items():
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
            identities[condition].add((
                int(measured["block_zero_product_count"]),
                int(measured["model_product_count"]),
                float(measured["R_model"]),
            ))
            checkpoints[condition].add(tuple(sorted(
                (row["path"], int(row["bytes"]), row["sha256"])
                for row in manifest["checkpoint"]["files"]
            )))
            summary = timing["matched"]["eager"]["summary"]["candidate"]
            if summary["samples"] != 80:
                raise ValueError(f"Expected 80 paired timing samples: {label}")
            rows.append({
                "label": label,
                "condition": condition,
                "repeat": repeat,
                "implementation": implementation,
                "candidate_validation_pass": bool(full["pass"]["candidate"]),
                "R_model": measured["R_model"],
                "paired_geomean_speedup": summary["paired_geomean_speedup"],
            })
        for condition in CONDITIONS.values():
            if len(identities[condition]) != 1 or len(checkpoints[condition]) != 1:
                raise ValueError(f"Checkpoint or R_model changed within {condition}")

    return {
        "passed": True,
        "archive": str(path),
        "archive_bytes": path.stat().st_size,
        "archive_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "processes": len(rows),
        "candidate_validation_passes": sum(row["candidate_validation_pass"] for row in rows),
        "fixed_rmodel_conditions": len(identities),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive), indent=2))


if __name__ == "__main__":
    main()
