"""Safely extract and verify the Phase 16 H100 evidence archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


RUN_RELATIVE = Path("runs/025-2026-09-05-pythia-agentic-sparse-kernel-search")
PHASE = "phase16-h100-frozen-transfer-h100nvl-001"


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    args.destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.archive, "r:") as archive:
        archive.extractall(args.destination, filter="data")

    run = args.destination / RUN_RELATIVE
    artifacts = run / "artifacts"
    phase = artifacts / PHASE
    inventory = json.loads((phase / "evidence-inventory.json").read_text(encoding="utf-8"))
    rows = inventory["files"]
    if not rows:
        raise ValueError("Empty H100 evidence inventory")
    for row in rows:
        path = run / row["path"]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"H100 evidence hash mismatch: {row['path']}")

    directories = [
        path
        for path in artifacts.glob("*h100nvl-001")
        if path.is_dir() and not path.name.startswith(("runtime-", "torch_extensions"))
    ]
    if len(directories) != 93:
        raise ValueError(f"Expected 93 Phase 16 artifact directories, found {len(directories)}")
    if (phase / "exit-code.txt").read_text(encoding="utf-8").strip() != "0":
        raise ValueError("Phase 16 did not exit successfully")
    if not (phase / "all-checks-finished-utc.txt").is_file():
        raise ValueError("Phase 16 completion marker missing")
    print(
        json.dumps(
            {
                "archive_sha256": digest(args.archive),
                "verified_files": len(rows),
                "artifact_directories": len(directories),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
