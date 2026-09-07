"""Snapshot explicitly finalized autoresearch attempts plus their source files."""
from pathlib import Path
import argparse
import hashlib
import json
import tarfile

RUN = Path(__file__).resolve().parents[1]
EXCLUDE = {"runtime", "torch_extensions", "__pycache__", "bundles", ".git", "retrieved"}
SOURCE_SUFFIX = {".py", ".cu", ".sh", ".json", ".md", ".ps1"}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", action="append", required=True,
                        help="Finalized directory relative to Run025; no environments or input weights")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = []
    for name in args.attempt:
        directory = (RUN / name).resolve()
        relative = directory.relative_to(RUN)
        if not relative.parts or set(relative.parts) & EXCLUDE or not directory.is_dir():
            raise ValueError("Only explicit finalized attempt directories")
        for path in directory.rglob("*"):
            if path.is_file() and not set(path.relative_to(directory).parts) & EXCLUDE:
                if path.suffix == ".tmp":
                    raise ValueError("Temporary artifact: worker may still be writing")
                paths.append(path)
    paths += [p for p in (RUN / "autoresearch").rglob("*") if p.is_file()
              and p.suffix in SOURCE_SUFFIX and not set(p.relative_to(RUN).parts) & (EXCLUDE | {"artifacts"})]
    paths += [*RUN.glob("*.py"), *RUN.glob("*.sh"), RUN / "config.json",
              *RUN.joinpath("kernels").glob("*.cu"), *RUN.joinpath("prelaunch").glob("*.json")]
    paths = sorted(set(paths))
    rows = [{"path": p.relative_to(RUN).as_posix(), "bytes": p.stat().st_size,
             "sha256": digest(p)} for p in paths]
    inventory = args.output.with_suffix(".inventory.json")
    with inventory.open("x", encoding="utf-8") as stream:
        json.dump({"files": rows, "attempts": args.attempt, "excludes": sorted(EXCLUDE)}, stream, indent=2)
    with tarfile.open(args.output, "x:gz", compresslevel=1) as archive:
        archive.add(inventory, arcname="inventory.json", recursive=False)
        for row, path in zip(rows, paths):
            archive.add(path, arcname=row["path"], recursive=False)
            if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
                raise RuntimeError("Artifact changed during snapshot; do not use this archive")
    print(json.dumps({"files": len(rows), "bytes": args.output.stat().st_size,
                      "archive_sha256": digest(args.output)}))


if __name__ == "__main__":
    main()
