"""Explicit transfer allowlist. Never traverse the repository's arbitrary artifacts."""
from __future__ import annotations

import argparse
import tarfile

from run025_common import ROOT, RUN, inside, read_json, record, select_conditions, sha256, verify_record, write_json


def input_files(manifest, phase):
    files = [manifest[key] for key in ("validation", "validation_metadata", "development")]
    for row in select_conditions(manifest, phase):
        files.extend(row["files"] + row["provenance"])
    return list({row["path"]: row for row in files}.values())


def source_files():
    paths = [*RUN.glob("*.py"), *RUN.glob("*.sh"), RUN / "config.json", RUN / "prelaunch/input_manifest.json",
             *RUN.joinpath("kernels").glob("*.cu"), *RUN.joinpath("upstream").glob("*"),
             *ROOT.joinpath("src/sparsity_research").rglob("*.py")]
    return [record(path) for path in sorted(paths) if path.is_file()]


def verify_inventory(inventory, root=ROOT):
    for row in inventory["files"]:
        verify_record(row, root)
    return len(inventory["files"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["primitive", "calibration", "development", "final"], default="calibration")
    parser.add_argument("--build-tar", action="store_true")
    parser.add_argument("--verify", help="Verify an existing transfer inventory, with no writes")
    args = parser.parse_args()
    if args.verify:
        print(f"Verified {verify_inventory(read_json(args.verify))} allowlisted file hashes")
        return
    manifest = read_json(RUN / "prelaunch/input_manifest.json")
    if manifest["config_sha256"] != sha256(RUN / "config.json"):
        raise ValueError("Configuration changed after preparation")
    rows = input_files(manifest, args.phase) + source_files()
    inventory = {"phase": args.phase, "files": rows, "bytes": sum(row["bytes"] for row in rows),
                 "conditions": [row["id"] for row in select_conditions(manifest, args.phase)],
                 "excludes": ["credentials", "optimizer states", "raw training cache", "unselected checkpoint weights"]}
    verify_inventory(inventory)
    destination = RUN / f"prelaunch/transfer-{args.phase}.json"
    write_json(destination, inventory)
    if args.build_tar:
        archive = RUN / f"prelaunch/run025-{args.phase}.tar"
        # Exclusive creation prevents replacing a payload whose hash was approved.
        with tarfile.open(archive, "x") as stream:
            for row in rows + [record(destination)]:
                stream.add(inside(ROOT, row["path"]), arcname=row["path"], recursive=False)
        write_json(archive.with_suffix(".tar.sha256"), record(archive))
    print(f"{args.phase}: {len(rows)} allowlisted files, {inventory['bytes']:,} bytes; no external transfer")


if __name__ == "__main__":
    main()
