"""Verify a size-development transfer archive without extracting it."""

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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--digest-record", type=Path, required=True)
    args = parser.parse_args()

    inventory_bytes = args.inventory.read_bytes()
    inventory = json.loads(inventory_bytes)
    digest_record = json.loads(args.digest_record.read_text(encoding="utf-8"))
    actual_digest = digest(args.archive)
    if (
        digest_record["bytes"] != args.archive.stat().st_size
        or digest_record["sha256"] != actual_digest
    ):
        raise ValueError("Archive identity differs from digest record")

    inventory_member = args.inventory.resolve().relative_to(
        Path.cwd().resolve()
    ).as_posix()
    expected = {item["path"]: item for item in inventory["files"]}
    expected_names = set(expected) | {inventory_member}
    with tarfile.open(args.archive, "r:") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or set(names) != expected_names:
            raise ValueError("Archive membership differs from its allowlist")
        for member in members:
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not member.isfile():
                raise ValueError(f"Unsafe or non-file archive member: {member.name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Missing archive stream: {member.name}")
            if member.name == inventory_member:
                if stream.read() != inventory_bytes:
                    raise ValueError("Archived inventory differs from source")
                continue
            row = expected[member.name]
            if (
                member.size != row["bytes"]
                or hashlib.file_digest(stream, "sha256").hexdigest()
                != row["sha256"]
            ):
                raise ValueError(f"Archived input identity mismatch: {member.name}")

    print(
        json.dumps(
            {
                "archive_bytes": args.archive.stat().st_size,
                "archive_sha256": actual_digest,
                "conditions": inventory["conditions"],
                "safe_member_paths": True,
                "verified_files": len(expected),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
