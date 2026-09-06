#!/usr/bin/env python3
"""Verify the LF-preserving Phase 17 source bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


RUN_PREFIX = "runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/"
PROGRAMS = (
    "autoresearch/probe_models.py",
    "autoresearch/probe_k004_flagged_z_models.py",
    "autoresearch/probe_k009_models.py",
    "autoresearch/probe_k013_final.py",
    "autoresearch/probe_k016_final.py",
    "autoresearch/probe_frozen_final.py",
)
WINNERS = ("k010", "k013", "k016")
PHASE = "autoresearch/launch-control/rtxpro4500-004/phase17-fixed-rmodel-pairs.sh"


def payload(archive: tarfile.TarFile, members: dict[str, tarfile.TarInfo], name: str) -> bytes:
    member = members.get(name)
    if member is None:
        raise ValueError(f"Missing bundle member: {name}")
    stream = archive.extractfile(member)
    if stream is None:
        raise ValueError(f"Unreadable bundle member: {name}")
    return stream.read()


def verify(archive_path: Path) -> dict:
    checked: list[dict] = []
    with tarfile.open(archive_path, "r:gz") as archive:
        members = {member.name: member for member in archive.getmembers()}
        required = (*PROGRAMS, PHASE)
        for relative in required:
            body = payload(archive, members, RUN_PREFIX + relative)
            if b"\r\n" in body:
                raise ValueError(f"CRLF payload is not allowed: {relative}")
            checked.append({
                "path": relative,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            })
        for candidate in WINNERS:
            frozen_name = RUN_PREFIX + f"autoresearch/candidates/{candidate}/FROZEN.json"
            frozen = json.loads(payload(archive, members, frozen_name))
            for row in frozen["sources"]:
                body = payload(archive, members, RUN_PREFIX + row["path"])
                if len(body) != row["bytes"] or hashlib.sha256(body).hexdigest() != row["sha256"]:
                    raise ValueError(f"Frozen source mismatch: {candidate} {row['path']}")
    return {
        "passed": True,
        "archive": str(archive_path),
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
        "winner_manifests": list(WINNERS),
        "required_files": checked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.archive), indent=2))


if __name__ == "__main__":
    main()
