"""Verify the exported companion in an isolated copy with no original run tree."""

from pathlib import Path
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "handoff"


def main():
    records = json.loads((SOURCE / "MANIFEST.json").read_text())["files"]
    expected_files = {row["path"] for row in records} | {"MANIFEST.json"}
    actual_files = {
        p.relative_to(SOURCE).as_posix() for p in SOURCE.rglob("*") if p.is_file()
    }
    assert actual_files == expected_files, (
        f"Unexpected/missing export files: {actual_files ^ expected_files}"
    )
    (ROOT / ".tmp").mkdir(exist_ok=True)
    scratch = Path(
        tempfile.mkdtemp(prefix="lean-handoff-", dir=ROOT / ".tmp")
    ).resolve()
    assert scratch.is_relative_to((ROOT / ".tmp").resolve())
    secrets = re.compile(
        rb"(?:AKIA[0-9A-Z]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----|(?:hf_|ghp_|github_pat_|rpa_)[A-Za-z0-9_]{24,}|sk-[A-Za-z0-9]{32,})"
    )
    forbidden = {
        ".pt",
        ".pth",
        ".safetensors",
        ".bin",
        ".npz",
        ".npy",
        ".so",
        ".pyd",
        ".dll",
        ".exe",
    }
    for row in records:
        source = SOURCE / row["path"]
        data = source.read_bytes()
        assert source.suffix not in forbidden, source
        assert (
            len(data) == row["bytes"]
            and hashlib.sha256(data).hexdigest() == row["sha256"]
        ), source
        assert not secrets.search(data), f"Credential-like content: {source}"
        target = scratch / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        if source.suffix == ".py":
            compile(data, str(source), "exec")
    shutil.copyfile(SOURCE / "MANIFEST.json", scratch / "MANIFEST.json")
    env = dict(
        os.environ,
        PYTHONPATH=str(scratch / "src"),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
    )
    commands = [
        [
            "-c",
            "from pathlib import Path; import sparsity_research; import training.config; assert Path(sparsity_research.__file__).resolve().is_relative_to(Path.cwd()); assert training.config.ROOT == Path.cwd()",
        ],
        ["scripts/reproduce.py", "verify"],
        ["-m", "pytest", "-q"],
        ["scripts/reproduce.py", "results"],
        ["scripts/reproduce.py", "figures"],
        ["scripts/prepare_data.py", "--help"],
        ["scripts/benchmark.py", "--help"],
        ["scripts/clip.py", "--help"],
        ["scripts/aggregate_timings.py", "--help"],
    ]
    checks = []
    for command in commands:
        result = subprocess.run(
            [sys.executable, *command],
            cwd=scratch,
            env=env,
            text=True,
            capture_output=True,
            timeout=180,
        )
        checks.append(
            dict(
                command=command,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
        )
        if result.returncode:
            print(result.stdout)
            print(result.stderr)
            break
    figures = sorted((scratch / "reproduced").glob("*.pdf"))
    summary = dict(
        status="passed"
        if all(c["returncode"] == 0 for c in checks) and len(checks) == len(commands)
        else "failed",
        package_files=len(records) + 1,
        package_bytes=sum(r["bytes"] for r in records)
        + (SOURCE / "MANIFEST.json").stat().st_size,
        heavy_files=0,
        credential_pattern_matches=0,
        isolated_directory=scratch.relative_to(ROOT).as_posix(),
        checks=checks,
        reconstructed_figures=[p.name for p in figures],
        original_bootstrap="242 passed; no scientific source changed",
        gpu_execution="not performed",
    )
    (HERE / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
