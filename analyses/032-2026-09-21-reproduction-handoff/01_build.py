"""Export the paper's core methods, final kernels, configs and compact results."""

from pathlib import Path
import hashlib
import json
import re
import subprocess
import ast
import sys
import yaml

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / "handoff"


def run(n):
    return next((ROOT / "runs").glob(f"{n:03}-*"))


def analysis(n):
    return next((ROOT / "analyses").glob(f"{n:03}-*"))


def main():
    OUT.mkdir(exist_ok=True)
    provenance = []
    emitted = set()

    def save(relative, data):
        target = OUT / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        emitted.add(target)

    def copy(source, relative, changes=()):
        data = source.read_bytes()
        for old, new in changes:
            assert old.encode() in data, (source, old)
            data = data.replace(old.encode(), new.encode())
        save(relative, data)
        provenance.append(
            {
                "path": relative,
                "source": source.relative_to(ROOT).as_posix(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "changes": [{"before": a, "after": b} for a, b in changes],
            }
        )

    for p in (ROOT / "src/sparsity_research").glob("*.py"):
        copy(p, "src/sparsity_research/" + p.name)
    for name in (
        "data",
        "sites",
        "pressure",
        "metrics",
        "ceilings",
        "evaluation",
        "logical_capture",
        "pythia_integration",
    ):
        copy(ROOT / f"tests/test_{name}.py", f"tests/test_{name}.py")
    copy(run(4) / "optimizer_boundary.py", "training/optimizer.py")
    copy(run(30) / "clipping.py", "training/clipping.py")
    source = (run(4) / "diagnostics.py").read_text(encoding="utf-8")
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.ClassDef) and n.name == "AttentionOutputCapture"
    )
    with (OUT / "training/clipping.py").open("a", encoding="utf-8") as f:
        f.write("\n\n" + ast.get_source_segment(source, node) + "\n")
    provenance[-1]["changes"].append(
        "Append AttentionOutputCapture unchanged from Run004 diagnostics.py"
    )
    provenance.append(
        {
            "path": "training/clipping.py",
            "source": (run(4) / "diagnostics.py").relative_to(ROOT).as_posix(),
            "source_sha256": hashlib.sha256(
                (run(4) / "diagnostics.py").read_bytes()
            ).hexdigest(),
            "changes": ["Extract AttentionOutputCapture unchanged"],
        }
    )
    copy(
        run(15) / "optimizer_boundary.py",
        "training/ol1.py",
        [
            (
                "from _reuse_run004 import load_run004_module",
                "from training import optimizer as _BASE",
            ),
            ("from run_config import EXPECTED_ACTIVE_SITES", ""),
            (
                '_BASE = load_run004_module("_run015_frozen_run004_optimizer_boundary", "optimizer_boundary.py")',
                "",
            ),
            (
                "if tuple(pressure.sites) != EXPECTED_ACTIVE_SITES:",
                "if not pressure.sites:",
            ),
            (
                "_expected_pressure_capture_names(model)",
                "_expected_pressure_capture_names(model, pressure.sites)",
            ),
            (
                "def _expected_pressure_capture_names(model: Any)",
                "def _expected_pressure_capture_names(model: Any, sites)",
            ),
            ("for site in EXPECTED_ACTIVE_SITES", "for site in sites"),
        ],
    )
    copy(
        run(4) / "initialization.py",
        "training/initialization.py",
        [
            (
                "EXPECTED_ARCHITECTURE.items()",
                '{k: v for k, v in EXPECTED_ARCHITECTURE.items() if k not in {"num_hidden_layers", "hidden_size", "intermediate_size", "num_attention_heads"}}.items()',
            )
        ],
    )
    copy(
        run(4) / "06_build_cache_from_hf.py",
        "scripts/prepare_data.py",
        [("RUN_DIR.parents[1]", "RUN_DIR.parent")],
    )
    for size, n in [("70m", 18), ("410m", 19)]:
        copy(run(n) / "architecture_config.json", f"configs/architectures/{size}.json")
    for doc in ("DATA", "METHODS", "METRICS", "DEFINITIONS"):
        copy(ROOT / f"research/{doc}.md", f"docs/reference/{doc}.md")
    sources = {
        "endpoints": analysis(30) / "data/full-trained-results.json",
        "14m-figure": analysis(27) / "data/figure-data.json",
        "operation-latency": analysis(24) / "data/operation-latency.json",
        "pressure-placement": analysis(21) / "pressure-scope/data/evidence.json",
        "ol1-geometry": analysis(24) / "data/ol1-appendix.json",
        "base-training": analysis(21) / "data/a0-optimization.json",
        "70m-controls": analysis(28) / "data/retained-controls.json",
        "70m-sessions": analysis(30) / "data/session-comparison.json",
        "kernel-structure": analysis(24) / "data/kernel-appendix.json",
        "clipping": analysis(24) / "data/compact-results-appendix.json",
    }
    for name, source in sources.items():
        copy(source, f"results/{name}.json")
        d = json.loads(source.read_text(encoding="utf-8"))
        save(
            f"results/{name}.json",
            (json.dumps(d, separators=(",", ":")) + "\n").encode(),
        )
        provenance[-1]["changes"].append("Compact JSON whitespace; values unchanged")
    for p in (ROOT / "manuscript/draft").glob("*.tex"):
        if p.name in {
            "methodology.tex",
            "methodology-appendix.tex",
            "experimental-appendix.tex",
            "kernel-appendix.tex",
            "training-results.tex",
        }:
            copy(p, "docs/paper/" + p.name)
    for p in (ROOT / "manuscript/draft/figures").rglob("*.pdf"):
        copy(
            p,
            "results/paper-figures/"
            + p.relative_to(ROOT / "manuscript/draft/figures").as_posix(),
        )
    copy(
        ROOT / "manuscript/artifacts/pythia-architecture-map.tex",
        "docs/paper/architecture.tex",
    )
    pending = ["k050"]
    seen = set()
    while pending:
        name = pending.pop()
        if name in seen:
            continue
        seen.add(name)
        folder = run(28) / "candidates" / name
        for p in folder.iterdir():
            if p.is_file() and (
                p.suffix in {".py", ".cu", ".h", ".cuh", ".hpp"}
                or p.name.startswith("LICENSE")
            ):
                copy(p, f"kernels/14m/candidates/{name}/{p.name}")
                if p.suffix == ".py":
                    pending += re.findall(r"candidates/(k\d+)/", p.read_text())
    for name in ("k018", "k019"):
        for p in (run(26) / "autoresearch/candidates" / name).iterdir():
            if p.is_file() and p.suffix in {".py", ".cu"}:
                copy(p, f"kernels/base/autoresearch/candidates/{name}/{p.name}")
    copy(run(45) / "hz_adapter.py", "kernels/base/adapter.py")
    copy(run(27) / "kernel.cu", "kernels/base/kernel.cu")
    copy(
        run(37) / "site_controls.py",
        "kernels/ablation14m/site_controls.py",
        [
            (
                "from io_utils import RUN, module, sha",
                "from support import ROOT, module, sha256 as sha\nRUN = ROOT/'ablation14m'",
            ),
            ("from frozen_replay import R28", "from replay import R28"),
        ],
    )
    copy(run(37) / "controls.py", "kernels/ablation14m/modes.py")
    copy(run(37) / "diagnostics.py", "kernels/ablation14m/diagnostics.py")
    for p in (run(37) / "candidate").iterdir():
        if p.is_file() and (
            p.suffix in {".cu", ".h", ".cuh", ".hpp"} or p.name.startswith("LICENSE")
        ):
            copy(p, "kernels/ablation14m/candidate/" + p.name)
    for folder in (
        "base70",
        "kernel",
        *(
            "candidates/" + c
            for c in (
                "opt001",
                "opt008",
                "opt019",
                "opt024",
                "opt025",
                "opt032",
                "opt063",
                "opt073",
            )
        ),
    ):
        for p in (run(45) / folder).rglob("*"):
            if p.is_file() and (
                p.suffix in {".py", ".cu", ".h", ".cuh", ".hpp", ".json"}
                or p.name.startswith("LICENSE")
            ):
                copy(p, "kernels/70m/" + p.relative_to(run(45)).as_posix())
    for name in ("controls.py", "hz_adapter.py"):
        copy(run(45) / name, "kernels/70m/" + name)
    copy(
        run(28) / "23_dependencies.py",
        "kernels/fetch_dependencies.py",
        [
            (
                "from common import RUN,ROOT,record,read_json,write_json,verify_record,sha256",
                "from support import RUN,ROOT,record,read_json,write_json,verify_record,sha256",
            )
        ],
    )
    for dep in ("flash", "cutlass"):
        for p in (run(28) / "runtime/vendor" / dep).glob("LICENSE*"):
            copy(p, f"LICENSES/{dep}-{p.name}")
    copy(run(25) / "upstream/LICENSE", "LICENSES/sparser-LICENSE")
    copy(run(45) / "provenance/pip-freeze.txt", "environment/kernel-pip-freeze.txt")
    copy(run(25) / "measurement.py", "kernels/measurement.py")
    p = run(25) / "autoresearch/dense_probe/probe.py"
    source = p.read_text(encoding="utf-8")
    tree = ast.parse(source)
    extracted = '"""Retained CUDA graph runner and paired host/event timer."""\nimport time\nimport torch\nfrom measurement import paired_schedule\n\n'
    extracted += (
        "\n\n".join(
            ast.get_source_segment(source, n)
            for n in tree.body
            if isinstance(n, (ast.ClassDef, ast.FunctionDef))
            and n.name in ("DenseRunner", "paired_probe")
        )
        + "\n"
    )
    save("kernels/timing.py", extracted.encode())
    provenance.append(
        {
            "path": "kernels/timing.py",
            "source": p.relative_to(ROOT).as_posix(),
            "source_sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
            "changes": [
                "Extract DenseRunner and paired_probe unchanged; standalone imports"
            ],
        }
    )
    for n in (4, 9, 11, 12, 13, 14, 15, 18, 19, 32, 34, 41, 43, 44, 46):
        source = run(n) / "config.yaml"
        d = yaml.safe_load(source.read_text())
        d.pop("runpod", None)
        save(
            f"configs/original/run{n:03}.json",
            (json.dumps(d, indent=2) + "\n").encode(),
        )
        provenance.append(
            {
                "path": f"configs/original/run{n:03}.json",
                "source": source.relative_to(ROOT).as_posix(),
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "changes": ["Convert YAML to JSON and omit cloud deployment settings"],
            }
        )
    for p in sorted((HERE / "package").rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            copy(p, p.relative_to(HERE / "package").as_posix())
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "release_config", OUT / "training/config.py"
    )
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    save(
        "configs/paper-grid.json",
        (
            json.dumps(
                [config.resolve(config.identifier(r)) for r in config.paper_rows()],
                indent=2,
            )
            + "\n"
        ).encode(),
    )
    provenance.append(
        {
            "path": "configs/paper-grid.json",
            "sources": ["training/config.py", "results/endpoints.json"],
            "changes": [
                "Resolve the executed 84-condition grid using the bundled config resolver"
            ],
        }
    )
    save(
        "PROVENANCE.json",
        (
            json.dumps(
                {
                    "source_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                    ).strip(),
                    "files": provenance,
                    "scope": "Lean companion; core primitives and CUDA sources plus portable entry points. No experiment launched.",
                },
                indent=2,
            )
            + "\n"
        ).encode(),
    )
    files = []
    for p in sorted(emitted):
        data = p.read_bytes()
        files.append(
            {
                "path": p.relative_to(OUT).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    save("MANIFEST.json", (json.dumps({"files": files}, indent=2) + "\n").encode())
    print(
        json.dumps(
            {
                "files": len(files),
                "bytes": sum(r["bytes"] for r in files),
                "14m_dependencies": sorted(seen),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
