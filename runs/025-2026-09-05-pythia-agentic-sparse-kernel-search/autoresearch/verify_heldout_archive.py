"""Audit every held-out payload member and its embedded allowlist hash."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile


INVENTORY_MEMBER = (
    "runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/"
    "prelaunch/transfer-heldout-checkpoints.json"
)


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return not path.is_absolute() and ".." not in path.parts


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with tarfile.open(args.archive, "r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        if any(not member.isfile() or not safe_member_name(member.name) for member in members):
            raise ValueError("Unsafe or non-file archive member")
        stream = archive.extractfile(archive.getmember(INVENTORY_MEMBER))
        if stream is None:
            raise ValueError("Missing inventory stream")
        inventory = json.load(stream)
        expected = {row["path"]: row for row in inventory["files"]}
        observed = {member.name: member for member in members if member.name != INVENTORY_MEMBER}
        if set(observed) != set(expected):
            raise ValueError("Archive membership differs from embedded allowlist")
        for name, row in expected.items():
            member = observed[name]
            if member.size != row["bytes"]:
                raise ValueError(f"Archive member size mismatch: {name}")
            stream = archive.extractfile(member)
            if stream is None or hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                raise ValueError(f"Archive member hash mismatch: {name}")
    print(json.dumps({"archive_bytes": args.archive.stat().st_size, "archive_sha256": digest(args.archive), "conditions": len(inventory["conditions"]), "safe_member_paths": True, "verified_files": len(expected)}, sort_keys=True))


if __name__ == "__main__":
    main()
