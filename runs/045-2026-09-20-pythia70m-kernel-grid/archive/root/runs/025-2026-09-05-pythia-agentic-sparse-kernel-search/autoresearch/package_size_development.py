"""Build a checkpoint-only development payload for one Pythia size."""

from __future__ import annotations

import argparse
import json
import tarfile
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run025_common import ROOT, RUN, inside, read_json, record, sha256, verify_record


SIZES = {"70m", "410m"}


def development_rows(manifest: dict, size: str) -> list[dict]:
    if size not in SIZES:
        raise ValueError(f"Unsupported transfer size: {size}")
    rows = [
        row
        for row in manifest["checkpoints"]
        if row["size"] == size and row["partition"] == "development"
    ]
    expected = {
        f"{size}/a0",
        f"{size}/a1h",
        f"{size}/a4-0",
        f"{size}/a4-0p5",
        f"{size}/a7-0",
        f"{size}/a7-0p5",
    }
    if {row["id"] for row in rows} != expected:
        raise ValueError("Development endpoint set differs from the frozen design")
    return sorted(rows, key=lambda row: row["id"])


def unique_records(rows: list[dict]) -> list[dict]:
    selected = {}
    for row in rows:
        for item in row["files"] + row["provenance"]:
            previous = selected.setdefault(item["path"], item)
            if previous != item:
                raise ValueError(f"Conflicting identity record: {item['path']}")
    return [selected[path] for path in sorted(selected)]


def write_exclusive_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def verify_inventory(path: Path) -> dict:
    inventory = read_json(path)
    if inventory["partition"] != "development" or inventory["size"] not in SIZES:
        raise ValueError("Unexpected transfer inventory scope")
    for item in inventory["files"]:
        verify_record(item)
    if inventory["bytes"] != sum(item["bytes"] for item in inventory["files"]):
        raise ValueError("Transfer byte total mismatch")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", choices=sorted(SIZES))
    parser.add_argument("--build-tar", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if bool(args.verify) == bool(args.size):
        parser.error("Choose either --verify or --size")
    if args.verify:
        inventory = verify_inventory(args.verify)
        print(
            json.dumps(
                {
                    "bytes": inventory["bytes"],
                    "conditions": inventory["conditions"],
                    "verified_files": len(inventory["files"]),
                },
                sort_keys=True,
            )
        )
        return

    manifest = read_json(RUN / "prelaunch/input_manifest.json")
    if manifest["config_sha256"] != sha256(RUN / "config.json"):
        raise ValueError("Configuration changed after input preparation")
    rows = development_rows(manifest, args.size)
    files = unique_records(rows)
    for item in files:
        verify_record(item)
    inventory = {
        "schema_version": 1,
        "size": args.size,
        "partition": "development",
        "conditions": [row["id"] for row in rows],
        "files": files,
        "bytes": sum(item["bytes"] for item in files),
        "required_existing": {
            "config_sha256": record(RUN / "config.json"),
            "input_manifest_sha256": record(RUN / "prelaunch/input_manifest.json"),
        },
        "excludes": [
            "credentials",
            "optimizer states",
            "token caches",
            "untuned interior-kappa checkpoints",
            "other model sizes",
        ],
    }
    inventory_path = RUN / f"prelaunch/transfer-{args.size}-development.json"
    archive_path = RUN / f"prelaunch/run025-{args.size}-development.tar"
    digest_path = archive_path.with_suffix(".tar.sha256")
    if any(path.exists() for path in (inventory_path, archive_path, digest_path)):
        raise FileExistsError("A size-development payload target already exists")
    write_exclusive_json(inventory_path, inventory)
    if args.build_tar:
        with tarfile.open(archive_path, "x") as archive:
            for item in files:
                archive.add(
                    inside(ROOT, item["path"]),
                    arcname=item["path"],
                    recursive=False,
                )
            archive.add(
                inventory_path,
                arcname=inventory_path.relative_to(ROOT).as_posix(),
                recursive=False,
            )
        write_exclusive_json(digest_path, record(archive_path))
    print(
        json.dumps(
            {
                "archive": str(archive_path) if args.build_tar else None,
                "bytes": inventory["bytes"],
                "conditions": inventory["conditions"],
                "files": len(files),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
