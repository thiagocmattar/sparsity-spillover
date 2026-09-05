#!/usr/bin/env python3
"""Build the allowlisted, credential-free Run-024 transfer archive."""

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
    "_reuse_run023.py",
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
    "07_summarize.py",
    "08_calibrate_batch.py",
)
FROZEN_RUN023_NAMES = (
    "sakana-pythia70.patch",
    "benchmark_core.py",
    "pythia_sparse.py",
    "02_remote_preflight.py",
    "03_benchmark.py",
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
    run023 = RUN_DIR.parent / "023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels"
    files.extend(run023 / name for name in FROZEN_RUN023_NAMES)
    for condition in conditions_for_phase(config, phase):
        checkpoint = checkpoint_path(config, condition)
        files.extend(checkpoint / name for name in checkpoint_file_records(config, condition))
        files.append(logical_products_path(config, condition))
    files.extend(repo_path(config["validation"][name]) for name in ("metadata", "tokens"))
    test_path = REPO_ROOT / "tests" / "test_run_024_pythia410m_sakana_sparse_kernel.py"
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
    parser.add_argument("--phase", choices=("sentinel",), default="sentinel")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--inventory", type=Path)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    config = load_config()
    files = transfer_files(config, args.phase)
    output = args.output or REPO_ROOT / "tmp" / "run024-sentinel-inputs.tar"
    inventory_path = args.inventory or RUN_DIR / "prelaunch" / "sentinel-input-inventory.json"
    archive_record = None
    if not args.inventory_only:
        output.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output, "w", format=tarfile.PAX_FORMAT) as archive:
            for path in files:
                _add_deterministic_file(archive, path)
        archive_record = {"path": display_path(output), "bytes": output.stat().st_size, "sha256": sha256_file(output)}
    payload = {
        "schema_version": 1,
        "run": config["name"],
        "phase": args.phase,
        "credential_free": True,
        "optimizer_state_included": False,
        "file_count": len(files),
        "payload_bytes": sum(path.stat().st_size for path in files),
        "files": [
            {"path": display_path(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}
            for path in files
        ],
        "archive": archive_record,
    }
    write_json(inventory_path, payload)
    print(f"PASS: Run 024 input inventory ({len(files)} files) -> {inventory_path}")
    if archive_record:
        print(f"Archive: {archive_record}")


if __name__ == "__main__":
    main()

