"""Build one checkpoint-only payload for all untouched interior-kappa models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tarfile


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run025_common import ROOT, RUN, inside, read_json, record, sha256, verify_record


SIZES = ("14m", "70m", "410m")
SUFFIXES = ("a4-0p01", "a4-0p05", "a4-0p1", "a7-0p01", "a7-0p05", "a7-0p1")


def heldout_rows(manifest: dict) -> list[dict]:
    rows = [row for row in manifest["checkpoints"] if row["partition"] == "untuned"]
    expected = {f"{size}/{suffix}" for size in SIZES for suffix in SUFFIXES}
    if {row["id"] for row in rows} != expected or len(rows) != len(expected):
        raise ValueError("Held-out checkpoint set differs from the frozen design")
    if {row["size"] for row in rows} != set(SIZES):
        raise ValueError("Held-out model-size coverage differs from the frozen design")
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
    if inventory["partition"] != "heldout_untuned" or inventory["sizes"] != list(SIZES):
        raise ValueError("Unexpected held-out transfer inventory scope")
    if set(inventory["conditions"]) != {
        f"{size}/{suffix}" for size in SIZES for suffix in SUFFIXES
    }:
        raise ValueError("Unexpected held-out transfer conditions")
    for item in inventory["files"]:
        verify_record(item)
    if inventory["bytes"] != sum(item["bytes"] for item in inventory["files"]):
        raise ValueError("Transfer byte total mismatch")
    return inventory


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-tar", action="store_true")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify and args.build_tar:
        parser.error("Choose either --verify or --build-tar")
    if args.verify:
        inventory = verify_inventory(args.verify)
        print(json.dumps({"bytes": inventory["bytes"], "conditions": len(inventory["conditions"]), "verified_files": len(inventory["files"])}, sort_keys=True))
        return

    manifest = read_json(RUN / "prelaunch/input_manifest.json")
    if manifest["config_sha256"] != sha256(RUN / "config.json"):
        raise ValueError("Configuration changed after input preparation")
    rows = heldout_rows(manifest)
    files = unique_records(rows)
    for item in files:
        verify_record(item)
    inventory = {
        "schema_version": 1,
        "sizes": list(SIZES),
        "partition": "heldout_untuned",
        "conditions": [row["id"] for row in rows],
        "files": files,
        "bytes": sum(item["bytes"] for item in files),
        "required_existing": {
            "config": record(RUN / "config.json"),
            "input_manifest": record(RUN / "prelaunch/input_manifest.json"),
            "development_cache": manifest["development"],
            "validation_cache": manifest["validation"],
            "validation_metadata": manifest["validation_metadata"],
        },
        "excludes": [
            "credentials",
            "optimizer states",
            "token-cache payloads",
            "development endpoint checkpoints",
        ],
    }
    inventory_path = RUN / "prelaunch/transfer-heldout-checkpoints.json"
    archive_path = RUN / "prelaunch/run025-heldout-checkpoints.tar"
    digest_path = archive_path.with_suffix(".tar.sha256")
    if any(path.exists() for path in (inventory_path, archive_path, digest_path)):
        raise FileExistsError("A held-out checkpoint payload target already exists")
    write_exclusive_json(inventory_path, inventory)
    if args.build_tar:
        with tarfile.open(archive_path, "x") as archive:
            for item in files:
                archive.add(inside(ROOT, item["path"]), arcname=item["path"], recursive=False)
            archive.add(inventory_path, arcname=inventory_path.relative_to(ROOT).as_posix(), recursive=False)
        write_exclusive_json(digest_path, record(archive_path))
    print(json.dumps({"archive": str(archive_path) if args.build_tar else None, "bytes": inventory["bytes"], "conditions": len(inventory["conditions"]), "files": len(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
