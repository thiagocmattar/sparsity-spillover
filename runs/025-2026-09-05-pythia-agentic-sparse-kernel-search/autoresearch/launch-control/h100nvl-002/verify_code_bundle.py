#!/usr/bin/env python3
"""Verify frozen source identities inside an LF-preserving H100 code bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


RUN_PREFIX = "runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/"
CANDIDATES = ("k010", "k013", "k016")


def verify(archive_path: Path) -> dict:
    checked: list[dict] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        for candidate in CANDIDATES:
            frozen_name = f"{RUN_PREFIX}autoresearch/candidates/{candidate}/FROZEN.json"
            if frozen_name not in members:
                raise ValueError(f"Missing frozen manifest: {frozen_name}")
            frozen_stream = archive.extractfile(members[frozen_name])
            if frozen_stream is None:
                raise ValueError(f"Unreadable frozen manifest: {frozen_name}")
            frozen = json.load(frozen_stream)
            for row in frozen["sources"]:
                member_name = RUN_PREFIX + row["path"]
                if member_name not in members:
                    raise ValueError(f"Missing frozen source: {member_name}")
                stream = archive.extractfile(members[member_name])
                if stream is None:
                    raise ValueError(f"Unreadable frozen source: {member_name}")
                payload = stream.read()
                actual = hashlib.sha256(payload).hexdigest()
                if len(payload) != row["bytes"] or actual != row["sha256"]:
                    raise ValueError(
                        f"Frozen source mismatch: {candidate} {row['path']} "
                        f"bytes={len(payload)} sha256={actual}"
                    )
                checked.append({
                    "candidate": candidate,
                    "path": row["path"],
                    "bytes": len(payload),
                    "sha256": actual,
                })
        phase = RUN_PREFIX + "autoresearch/launch-control/h100nvl-001/phase16-h100-frozen-transfer.sh"
        if phase not in members:
            raise ValueError(f"Missing frozen Phase 16 controller: {phase}")
    return {
        "passed": True,
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "candidate_manifests": list(CANDIDATES),
        "verified_frozen_sources": len(checked),
        "sources": checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive), indent=2))


if __name__ == "__main__":
    main()
