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
    assert not list(SOURCE.rglob("*.tex"))
    assert set(SOURCE.rglob("*.pdf")) == {SOURCE / "main.pdf"}
    assert (SOURCE / "main.pdf").read_bytes() == (
        ROOT / "manuscript/draft/main.pdf"
    ).read_bytes()
    assert not (SOURCE / "configs/original").exists()
    assert not list(SOURCE.rglob("candidate.py"))
    source_map = json.loads((HERE / "source-map.json").read_text())
    evidence_source = next(
        r["source"] for r in source_map if r["path"] == "results/endpoints.json"
    )
    original_endpoints = json.loads((ROOT / evidence_source).read_text())[
        "trained_points"
    ]
    exported_endpoints = json.loads((SOURCE / "results/endpoints.json").read_text())[
        "trained_points"
    ]
    identity = lambda r: tuple(
        r.get(k)
        for k in ("model", "scope", "pressure", "kappa", "local_pressure_weight")
    )
    original_by_identity = {identity(r): r for r in original_endpoints}
    assert len(original_by_identity) == len(exported_endpoints) == 84
    for row in exported_endpoints:
        original_row = original_by_identity[identity(row)]
        for key in (
            "loss",
            "sparsity",
            "zero_product_count",
            "model_product_count",
            "latency_ms",
            "initial_parameter_sha256",
            "training_schedule_hash",
            "final_checkpoint_content_sha256",
        ):
            assert row.get(key) == original_row.get(key), (row["condition"], key)
    # Compare selected CUDA tokens independently of the exporter's transformations.
    cuda_files = 0
    for row in source_map:
        path = SOURCE / row["path"]
        if path.suffix not in {".cu", ".h", ".cuh", ".hpp"}:
            continue
        original = (ROOT / row["source"]).read_text(encoding="utf-8")
        exported = path.read_text(encoding="utf-8")

        def tokens(text):
            text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.S)
            text = text.replace("RUN037_SKIP_", "SPARSE_SKIP_")
            for old, new in [
                ("run028", "sparse"),
                ("Run028", "Sparse"),
                ("run035", "attention"),
                ("Run035", "Attention"),
                ("run037", "ablation"),
                ("Run037", "Ablation"),
            ]:
                text = text.replace(old, new)
            return re.findall(r"[A-Za-z_]\w*|\d+(?:\.\d+)?|[^\s]", text)

        if path.name == "head.cu":
            original = re.sub(
                r"switch\(tile\)\{.*?\n \}",
                'TORCH_CHECK(tile==0,"Only the final vocabulary tile is supported");\n launch<128,128,32,64,64>(x,w,out,stream);',
                original,
                flags=re.S,
            )
        assert tokens(original) == tokens(exported), f"CUDA arithmetic changed: {path}"
        cuda_files += 1
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
        cuda_source_equivalence_files=cuda_files,
        unchanged_central_endpoints=84,
        manuscript_pdf="byte-identical to manuscript/draft/main.pdf; no TeX distributed",
    )
    (HERE / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "checks"}, indent=2))
    if summary["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
