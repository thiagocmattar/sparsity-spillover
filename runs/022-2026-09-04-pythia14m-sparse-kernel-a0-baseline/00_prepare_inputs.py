#!/usr/bin/env python3
"""Build the exact, credential-free Run-022 transfer archive."""

from __future__ import annotations

import argparse
import io
import tarfile
from pathlib import Path

from run_config import REPO_ROOT, RUN_DIR, display_path, load_config, repo_path, sha256_file, write_json


RUN_SOURCE_NAMES = (
    "README.md",
    "DEPLOYMENT_PLAYBOOK.md",
    "FOLLOWUP_PLAN.md",
    "config.yaml",
    "run_config.py",
    "benchmark_core.py",
    "00_prepare_inputs.py",
    "00_setup_remote.sh",
    "01_static_preflight.py",
    "02_remote_preflight.py",
    "03_benchmark.py",
    "04_verify.py",
    "05_start_worker.sh",
    "06_monitor.py",
)
FORBIDDEN_NAME_FRAGMENTS = (
    ".env",
    "credential",
    "secret",
    "token.json",
    "training_state.pt",
    "optimizer",
)


def transfer_files(config: dict) -> list[Path]:
    files = [REPO_ROOT / "pyproject.toml", REPO_ROOT / "README.md"]
    files.extend(sorted((REPO_ROOT / "src").rglob("*.py")))
    files.extend(RUN_DIR / name for name in RUN_SOURCE_NAMES)
    checkpoint = repo_path(config["model"]["checkpoint"])
    files.extend(checkpoint / name for name in config["model"]["files"])
    files.extend(repo_path(config["validation"][name]) for name in ("metadata", "tokens"))
    test = REPO_ROOT / "tests" / "test_run_022_sparse_kernel_a0_baseline.py"
    if test.is_file():
        files.append(test)
    unique = sorted(set(path.resolve() for path in files), key=lambda path: path.as_posix())
    for path in unique:
        relative = path.relative_to(REPO_ROOT.resolve()).as_posix().lower()
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"Transfer entry must be a regular file: {path}")
        if any(fragment in relative for fragment in FORBIDDEN_NAME_FRAGMENTS):
            raise ValueError(f"Forbidden transfer entry: {relative}")
    return unique


def _add_deterministic_file(archive: tarfile.TarFile, path: Path) -> None:
    relative = path.relative_to(REPO_ROOT.resolve()).as_posix()
    data = path.read_bytes()
    info = tarfile.TarInfo(relative)
    info.size = len(data)
    info.mode = 0o755 if path.suffix in {".sh"} or path.name.endswith(".py") else 0o644
    info.mtime = info.uid = info.gid = 0
    info.uname = info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "tmp" / "run022-inputs.tar")
    parser.add_argument("--inventory", type=Path, default=RUN_DIR / "prelaunch" / "input-inventory.json")
    args = parser.parse_args()
    config = load_config()
    files = transfer_files(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(args.output, "w", format=tarfile.PAX_FORMAT) as archive:
        for path in files:
            _add_deterministic_file(archive, path)
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
        "archive": {
            "path": display_path(args.output),
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
        },
        "file_count": len(entries),
        "file_bytes": sum(entry["bytes"] for entry in entries),
        "entries": entries,
        "allowlist_only": True,
        "credentials_included": False,
        "training_state_included": False,
        "payload_description": "Run code, public MiniPile validation cache, and the trained A0 model checkpoint only.",
    }
    write_json(args.inventory, inventory)
    print(f"Wrote {args.output} ({inventory['archive']['bytes']} bytes, {len(entries)} files)")
    print(f"SHA256 {inventory['archive']['sha256']}")


if __name__ == "__main__":
    main()
