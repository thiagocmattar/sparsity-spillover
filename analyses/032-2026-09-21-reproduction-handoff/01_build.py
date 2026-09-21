"""Export the paper's core methods, final kernels, configs and compact results."""

from pathlib import Path
import hashlib
import json
import subprocess
import ast
import sys
from release_kernels import build as build_kernels
from release_results import clean_results

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
    previous = (
        json.loads((OUT / "MANIFEST.json").read_text())["files"]
        if (OUT / "MANIFEST.json").exists()
        else []
    )
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
        if p.name != "artifacts.py":
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
    site_path = OUT / "src/sparsity_research/sites.py"
    site_text = site_path.read_text(encoding="utf-8")
    tree = ast.parse(site_text)
    registry = next(
        n
        for n in tree.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "_TOPOLOGY_ROWS" for t in n.targets)
    )
    selected = [
        (name, sites)
        for name, sites in ast.literal_eval(registry.value)
        if name in {"A0", "A1-H", "A4-Z", "A7-Z-POST"}
    ]
    selected.insert(2, ("HZ", ("h", "z")))
    replacement = "_TOPOLOGY_ROWS = " + repr(tuple(selected))
    site_text = site_text.replace(
        ast.get_source_segment(site_text, registry), replacement
    )
    save("src/sparsity_research/sites.py", site_text.encode())
    next(r for r in provenance if r["path"] == "src/sparsity_research/sites.py")[
        "changes"
    ].append("Keep only the five paper topologies, including HZ")
    for name in ("sites", "pythia_integration"):
        target = OUT / f"tests/test_{name}.py"
        text = (
            target.read_text(encoding="utf-8")
            .replace("A6-POST", "A7-Z-POST")
            .replace("a6_post", "a7_post")
        )
        text = text.replace(
            '["a", "m", "h", "q_post", "k_post", "v"]',
            '["a", "m", "h", "q_post", "k_post", "v", "z"]',
        )
        text = (
            "\n".join(
                line
                for line in text.splitlines()
                if not any(
                    x in line
                    for x in (
                        'TOPOLOGIES["A2"]',
                        'TOPOLOGIES["A5-QK-PRE"]',
                        'TOPOLOGIES["A6-POST"]',
                    )
                )
            )
            + "\n"
        )
        # Remove the obsolete six-site suffix assertion after the topology rename.
        text = text.replace(
            '    assert TOPOLOGIES["A7-Z-POST"].active_sites[-3:] == ("q_post", "k_post", "v")\n',
            "",
        )
        save(f"tests/test_{name}.py", text.encode())
        next(r for r in provenance if r["path"] == f"tests/test_{name}.py")[
            "changes"
        ].append(
            "Exercise the seven-site paper topology instead of unused legacy topologies"
        )
    copy(run(4) / "optimizer_boundary.py", "training/optimizer.py")
    copy(
        run(30) / "clipping.py",
        "training/clipping.py",
        [
            (
                "Run-local retained TEAL helpers, copied from Run 019; see helper-provenance.json.",
                "Empirical activation-clipping calibration and full-validation evaluation.",
            )
        ],
    )
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
            (
                "Run 015 FP16 boundary with verified four-site A4-OL1 capture.",
                "FP16 optimizer boundary with explicit pressure-site capture.",
            ),
            (
                "Run 015's recipe boundary accepts orthogonal_l1 only.",
                "This recipe boundary accepts orthogonal_l1 only.",
            ),
            (
                "Run 015 OL1 must capture exactly all four A4 sites.",
                "OL1 requires at least one pressure site.",
            ),
            ("Run 015 pressure capture mismatch:", "Pressure capture mismatch:"),
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
        d = clean_results(name, json.loads(source.read_text(encoding="utf-8")), sources)
        save(
            f"results/{name}.json",
            (json.dumps(d, separators=(",", ":")) + "\n").encode(),
        )
        provenance[-1]["changes"].append(
            "Keep paper measurements; replace archive labels with scientific identifiers"
        )
    copy(ROOT / "manuscript/draft/main.pdf", "main.pdf")
    build_kernels(run, ROOT, save, provenance)
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
    (HERE / "source-map.json").write_text(json.dumps(provenance, indent=2) + "\n")
    public = [
        {
            "path": row["path"],
            "source_sha256": row.get("source_sha256"),
            "adapted": bool(row.get("changes")),
        }
        for row in provenance
    ]
    save(
        "PROVENANCE.json",
        (
            json.dumps(
                {
                    "source_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
                    ).strip(),
                    "files": public,
                    "scope": "Final methods and measured paper results. Development history is not distributed.",
                },
                indent=2,
            )
            + "\n"
        ).encode(),
    )
    # Delete only obsolete files from our own previous export, after checking identity.
    for row in previous:
        path = (OUT / row["path"]).resolve()
        if path not in {p.resolve() for p in emitted} and path.exists():
            assert path.is_relative_to(OUT.resolve()), path
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"], (
                f"Modified obsolete export file: {path}"
            )
            path.unlink()
    for path in sorted(OUT.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir() and not any(path.iterdir()):
            assert path.resolve().is_relative_to(OUT.resolve())
            path.rmdir()
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
