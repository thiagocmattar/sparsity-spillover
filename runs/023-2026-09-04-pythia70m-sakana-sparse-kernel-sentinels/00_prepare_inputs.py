#!/usr/bin/env python3
"""Build an allowlisted, credential-free Run-023 phase transfer archive."""

from __future__ import annotations

import argparse
import tarfile
from pathlib import Path

from run_config import (
    REPO_ROOT,
    RUN_DIR,
    checkpoint_file_records,
    checkpoint_path,
    conditions_for_phase,
    display_path,
    load_config,
    logical_products_path,
    repo_path,
    sha256_file,
    write_json,
)


RUN_SOURCE_NAMES = (
    "README.md",
    "DEPLOYMENT_PLAYBOOK.md",
    "config.yaml",
    "sakana-pythia70.patch",
    "run_config.py",
    "benchmark_core.py",
    "pythia_sparse.py",
    "00_prepare_inputs.py",
    "00_setup_remote.sh",
    "01_static_preflight.py",
    "02_remote_preflight.py",
    "03_benchmark.py",
    "04_verify.py",
    "05_start_worker.sh",
    "06_monitor.py",
)
FORBIDDEN_FRAGMENTS = (
    ".env",
    "credential",
    "secret",
    "token.json",
    "training_state.pt",
    "optimizer",
    "__pycache__",
)


def transfer_files(config: dict, phase: str) -> list[Path]:
    files = [REPO_ROOT / "pyproject.toml", REPO_ROOT / "README.md"]
    files.extend(sorted((REPO_ROOT / "src").rglob("*.py")))
    files.extend(RUN_DIR / name for name in RUN_SOURCE_NAMES)
    for condition in conditions_for_phase(config, phase):
        checkpoint = checkpoint_path(config, condition)
        files.extend(checkpoint / name for name in checkpoint_file_records(config, condition))
        files.append(logical_products_path(config, condition))
    files.extend(repo_path(config["validation"][name]) for name in ("metadata", "tokens"))
    test_path = REPO_ROOT / "tests" / "test_run_023_pythia70m_sakana_sparse_kernel.py"
    if test_path.is_file():
        files.append(test_path)
    unique = sorted(set(path.resolve() for path in files), key=lambda path: path.as_posix())
    for path in unique:
        relative = path.relative_to(REPO_ROOT.resolve()).as_posix().lower()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Transfer entry must be a regular file: {path}")
        if any(fragment in relative for fragment in FORBIDDEN_FRAGMENTS):
            raise ValueError(f"Forbidden transfer entry: {relative}")
    return unique


def _add_deterministic_file(archive: tarfile.TarFile, path: Path) -> None:
    relative = path.relative_to(REPO_ROOT.resolve()).as_posix()
    info = tarfile.TarInfo(relative)
    info.size = path.stat().st_size
    info.mode = 0o755 if path.suffix in {".sh", ".py"} else 0o644
    info.mtime = info.uid = info.gid = 0
    info.uname = info.gname = ""
    with path.open("rb") as handle:
        archive.addfile(info, handle)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("sentinel", "remainder"), default="sentinel")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    config = load_config()
    files = transfer_files(config, args.phase)
    output = args.output or REPO_ROOT / "tmp" / f"run023-{args.phase}-inputs.tar"
    inventory_path = args.inventory or RUN_DIR / "prelaunch" / f"{args.phase}-input-inventory.json"
    archive_record = None
    if not args.inventory_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
            for path in files:
                _add_deterministic_file(archive, path)
        archive_record = {
            "path": display_path(output),
            "bytes": output.stat().st_size,
            "sha256": sha256_file(output),
        }
    entries = [
        {
            "path": path.relative_to(REPO_ROOT.resolve()).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    inventory = {
        "schema_version": 1,
        "phase": args.phase,
        "condition_ids": config["condition_partition"][args.phase],
        "archive": archive_record,
        "file_count": len(entries),
        "file_bytes": sum(entry["bytes"] for entry in entries),
        "entries": entries,
        "allowlist_only": True,
        "credentials_included": False,
        "training_state_included": False,
        "train_cache_included": False,
        "payload_description": (
            "Run/source code, public MiniPile validation cache, canonical logical-product JSON, "
            f"and final model-only files for the Run-023 {args.phase} conditions."
        ),
    }
    write_json(inventory_path, inventory)
    if archive_record:
        print(f"Wrote {output} ({archive_record['bytes']} bytes, {len(entries)} files)")
        print(f"SHA256 {archive_record['sha256']}")
    else:
        print(f"Wrote inventory only ({inventory['file_bytes']} source bytes, {len(entries)} files)")


if __name__ == "__main__":
    main()
