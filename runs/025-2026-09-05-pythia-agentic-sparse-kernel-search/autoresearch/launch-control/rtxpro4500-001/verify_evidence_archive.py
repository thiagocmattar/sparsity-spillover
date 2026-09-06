"""Verify every evidence archive member without extracting long Windows paths."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()

    with tarfile.open(args.archive, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive member")
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise ValueError(f"Unsafe or non-file archive member: {member.name}")
        inventory_member = archive.getmember("inventory.json")
        inventory_stream = archive.extractfile(inventory_member)
        if inventory_stream is None:
            raise ValueError("Missing inventory stream")
        inventory = json.load(inventory_stream)
        expected = {row["path"]: row for row in inventory["files"]}
        observed = {member.name: member for member in members if member.name != "inventory.json"}
        if set(observed) != set(expected):
            raise ValueError("Archive membership differs from inventory")
        for name, row in expected.items():
            member = observed[name]
            if member.size != row["bytes"]:
                raise ValueError(f"Evidence size mismatch: {name}")
            stream = archive.extractfile(member)
            if stream is None or hashlib.file_digest(stream, "sha256").hexdigest() != row["sha256"]:
                raise ValueError(f"Evidence hash mismatch: {name}")

    print(
        json.dumps(
            {
                "archive_bytes": args.archive.stat().st_size,
                "archive_sha256": digest(args.archive),
                "safe_member_paths": True,
                "verified_files": len(expected),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
