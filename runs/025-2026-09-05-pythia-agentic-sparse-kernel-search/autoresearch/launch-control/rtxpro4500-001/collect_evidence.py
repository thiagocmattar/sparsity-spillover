"""Archive and verify Run 025 RTX PRO 4500 evidence without weights or environments."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path


RUN = Path(__file__).resolve().parents[3]
ROOT = RUN.parents[1]
SUFFIXES = {".json", ".jsonl", ".txt", ".csv", ".log", ".sh", ".cu", ".py", ".md"}
EXCLUDE_PARTS = {
    ".git",
    ".pytest_cache",
    "__pycache__",
    "retrieved",
    "runtime",
    "torch_extensions",
    "venv",
}


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def eligible(path: Path, anchor: Path) -> bool:
    relative = path.relative_to(anchor)
    return (
        path.is_file()
        and not (set(relative.parts) & EXCLUDE_PARTS)
        and (path.suffix in SUFFIXES or path.name.endswith("exit-code"))
    )


def source_rows() -> list[tuple[str, Path]]:
    rows: list[tuple[str, Path]] = []
    for anchor in (RUN / "artifacts", RUN / "autoresearch"):
        rows.extend(
            (path.relative_to(RUN).as_posix(), path)
            for path in anchor.rglob("*")
            if eligible(path, anchor)
        )
    for path in [
        *RUN.glob("*.py"),
        *RUN.glob("*.sh"),
        *RUN.glob("*.json"),
        *RUN.joinpath("prelaunch").glob("*.json"),
    ]:
        if path.is_file():
            rows.append((path.relative_to(RUN).as_posix(), path))
    for path in sorted((ROOT / "src/sparsity_research").glob("*.py")):
        rows.append((f"repository/{path.relative_to(ROOT).as_posix()}", path))
    return sorted(set(rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path)
    parser.add_argument("--extract-verify", type=Path)
    parser.add_argument("--destination", type=Path)
    args = parser.parse_args()
    if bool(args.build) == bool(args.extract_verify):
        parser.error("Choose exactly one of --build or --extract-verify")

    if args.build:
        rows = [
            {
                "path": archive_path,
                "bytes": source.stat().st_size,
                "sha256": digest(source),
                "source": source,
            }
            for archive_path, source in source_rows()
        ]
        inventory_path = args.build.with_suffix(".inventory.json")
        inventory = {
            "files": [
                {key: row[key] for key in ("path", "bytes", "sha256")}
                for row in rows
            ],
            "excluded_path_parts": sorted(EXCLUDE_PARTS),
            "excludes_weights_and_environments": True,
        }
        with inventory_path.open("x", encoding="utf-8") as stream:
            json.dump(inventory, stream, indent=2, sort_keys=True)
            stream.write("\n")
        with tarfile.open(args.build, "x:gz") as archive:
            archive.add(inventory_path, arcname="inventory.json", recursive=False)
            for row in rows:
                archive.add(row["source"], arcname=row["path"], recursive=False)
        print(
            json.dumps(
                {
                    "archive": str(args.build),
                    "bytes": args.build.stat().st_size,
                    "files": len(rows),
                    "sha256": digest(args.build),
                },
                sort_keys=True,
            )
        )
        return

    if args.destination is None:
        parser.error("--extract-verify requires a new --destination")
    args.destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(args.extract_verify, "r:gz") as archive:
        archive.extractall(args.destination, filter="data")
    inventory = json.loads(
        (args.destination / "inventory.json").read_text(encoding="utf-8")
    )
    for row in inventory["files"]:
        path = args.destination / row["path"]
        if path.stat().st_size != row["bytes"] or digest(path) != row["sha256"]:
            raise ValueError(f"Evidence hash mismatch: {row['path']}")
    print(
        json.dumps(
            {
                "archive_sha256": digest(args.extract_verify),
                "verified_files": len(inventory["files"]),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
