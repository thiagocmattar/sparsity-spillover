"""Archive finalized Run 025 pilot evidence only; never weights or environments."""
from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
SUFFIXES = {".json", ".jsonl", ".txt", ".csv", ".log", ".sh", ".cu", ".py"}
EXCLUDE = {"runtime", "torch_extensions", "venv", "upstream", "SparseLM0.5B", "__pycache__", ".git"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path)
    parser.add_argument("--extract-verify", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if bool(args.build) == bool(args.extract_verify):
        parser.error("Choose build or extract-verify")
    if args.build:
        paths = [p for p in (RUN / "artifacts").rglob("*") if p.is_file()
                 and not (set(p.relative_to(RUN / "artifacts").parts) & EXCLUDE)
                 and (p.suffix in SUFFIXES or p.name.endswith("exit-code"))]
        paths += [*RUN.glob("*.py"), *RUN.glob("*.sh"), *RUN.glob("bootstrap-*.sh"),
                  *RUN.glob("memcheck-*.sh"), RUN / "config.json",
                  *RUN.joinpath("kernels").glob("*.cu"),
                  *RUN.joinpath("prelaunch").glob("*.json"),
                  *RUN.joinpath("launch-control").glob("*.py")]
        paths = sorted(set(paths))
        rows = [{"path": p.relative_to(RUN).as_posix(), "bytes": p.stat().st_size,
                 "sha256": digest(p)} for p in paths]
        inventory = args.build.with_suffix(".inventory.json")
        with inventory.open("x", encoding="utf-8") as stream:
            json.dump({"files": rows, "excludes": sorted(EXCLUDE)}, stream, indent=2)
        with tarfile.open(args.build, "x:gz") as archive:
            archive.add(inventory, arcname="inventory.json", recursive=False)
            for row, path in zip(rows, paths):
                archive.add(path, arcname=row["path"], recursive=False)
        print(json.dumps({"archive": str(args.build), "bytes": args.build.stat().st_size,
                          "sha256": digest(args.build), "files": len(rows)}))
    else:
        if args.destination is None:
            parser.error("Extraction requires a new destination")
        args.destination.mkdir(parents=True, exist_ok=False)
        with tarfile.open(args.extract_verify, "r:gz") as archive:
            archive.extractall(args.destination, filter="data")
        inventory = json.loads((args.destination / "inventory.json").read_text(encoding="utf-8"))
        for row in inventory["files"]:
            path = args.destination / row["path"]
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise ValueError(f"Evidence hash mismatch: {row['path']}")
        print(json.dumps({"verified_files": len(inventory["files"]),
                          "archive_sha256": digest(args.extract_verify)}))


if __name__ == "__main__":
    main()
