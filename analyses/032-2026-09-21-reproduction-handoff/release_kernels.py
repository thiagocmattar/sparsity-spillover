"""Extract final components directly; development installers stay in the archive."""

import ast
import hashlib
import re


def public_names(text):
    text = re.sub(r"_run\d+_", "_sparsity_", text)
    text = re.sub(r"run\d+(?:_opt\d+|_k\d+)?", "sparsity", text, flags=re.I)
    text = re.sub(r"\bK\d{3}\b", "specialized", text, flags=re.I)
    return text


def minimal_imports(text):
    tree = ast.parse(text)
    loaded = {
        n.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
    }
    kept = []
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            node.names = [
                n for n in node.names if (n.asname or n.name.split(".")[0]) in loaded
            ]
            if not node.names:
                continue
        kept.append(node)
    tree.body = kept
    return ast.unparse(tree) + "\n"


def build(run, root, save, provenance):
    def record(source, target, details):
        provenance.append(
            dict(
                path=target,
                source=source.relative_to(root).as_posix(),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                changes=details,
            )
        )

    def nodes(path, names):
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        found = [n for n in tree.body if getattr(n, "name", None) in names]
        assert {n.name for n in found} == set(names), (path, names)
        return found

    def extract(source, target, names, description, replacements=()):
        body = nodes(source, names)
        imports = """from functools import lru_cache
from pathlib import Path
from types import MethodType
import hashlib
import math
import os
import torch
from support import ROOT, module, sha256, read_json, verify_record, write_json
from sparsity_research.sites import FixedOneSidedThreshold, FixedSymmetricThreshold
"""
        text = (
            '"""'
            + description
            + '"""\n'
            + imports
            + "\n"
            + ast.unparse(ast.Module(body=body, type_ignores=[]))
            + "\n"
        )
        text = text.replace("RUN / 'runtime/vendor", "ROOT / 'runtime/vendor")
        for old, new in replacements:
            assert old in text, (source, old)
            text = text.replace(old, new)
        save(target, minimal_imports(public_names(text)).encode())
        record(
            source,
            target,
            [
                "Extract only " + ", ".join(names),
                "Rename private attributes and build labels; relocate imports",
            ],
        )

    def cuda(source, target, changes=()):
        text = source.read_text(encoding="utf-8")
        for old, new in changes:
            assert old in text, (source, old)
            text = text.replace(old, new)
        # Only internal experiment labels in comments and ablation macro names change.
        text = re.sub(r"RUN037_SKIP_", "SPARSE_SKIP_", text)
        text = re.sub(r"(?m)^//.*(?:Run\d{3}|opt\d{3}|K\d{3}).*\n", "", text)
        for old, new in [
            ("run028", "sparse"),
            ("Run028", "Sparse"),
            ("run035", "attention"),
            ("Run035", "Attention"),
            ("run037", "ablation"),
            ("Run037", "Ablation"),
        ]:
            text = text.replace(old, new)
        save(target, text.encode())
        record(
            source,
            target,
            [
                "Final selected CUDA component; arithmetic unchanged",
                "Rename internal C++ identifiers and ablation macros; remove development comments",
                *(["Remove unused vocabulary tile branches"] if changes else []),
            ],
        )

    small = run(28) / "candidates"
    large = run(45)
    for folder, source, parts in [
        (
            "model_14m/normalization",
            small / "k050",
            ("extension", "gate_spec", "NormPair", "bind_pair"),
        ),
        ("model_14m/input_projection", small / "k042", ("extension", "Projection")),
        ("model_14m/output_projection", small / "k049", ("extension", "Joint")),
        (
            "model_14m/attention/implementation",
            small / "k035",
            ("extension", "Attention"),
        ),
        (
            "model_70m/normalization",
            large / "kernel/norm",
            ("extension", "gate_spec", "NormPair", "bind_pair"),
        ),
        (
            "model_70m/attention/implementation",
            large / "candidates/opt073/attention",
            ("extension", "Attention"),
        ),
    ]:
        extract(
            source / "candidate.py",
            "kernels/" + folder + ".py",
            parts,
            folder.replace("/", " ").replace("_", " "),
        )
        dest = "kernels/" + folder.rsplit("/", 1)[0]
        for p in source.iterdir():
            if p.suffix in {".cu", ".h", ".cuh", ".hpp"} or p.name.startswith(
                "LICENSE"
            ):
                cuda(p, dest + "/" + p.name)
    extract(
        large / "candidates/opt063/joint.py",
        "kernels/model_70m/output_projection.py",
        ("extension", "Joint"),
        "Parallel inspection and sparse output projections",
    )
    cuda(large / "candidates/opt063/joint.cu", "kernels/model_70m/joint.cu")
    extract(
        large / "candidates/opt025/head.py",
        "kernels/model_70m/vocabulary.py",
        ("extension", "Head"),
        "Dense full-vocabulary projection with the selected fixed tile",
    )
    original = (large / "candidates/opt025/head.cu").read_text()
    start = original.index(" switch(tile){")
    end = original.index("\n C10_CUDA_KERNEL_LAUNCH_CHECK();", start)
    cuda(
        large / "candidates/opt025/head.cu",
        "kernels/model_70m/head.cu",
        [
            (
                original[start:end],
                ' TORCH_CHECK(tile==0,"Only the final vocabulary tile is supported");\n launch<128,128,32,64,64>(x,w,out,stream);',
            )
        ],
    )

    rotary = run(26) / "autoresearch/candidates/k019/candidate.py"
    base = nodes(rotary, ("extension", "RopeGate"))
    adapter = nodes(large / "hz_adapter.py", ("RopeGate",))[0]
    base[1].body = [
        n for n in base[1].body if getattr(n, "name", None) != "__init__"
    ] + [n for n in adapter.body if getattr(n, "name", None) == "__init__"]
    text = (
        '''"""Fused QKV layout, partial RoPE and existing post-RoPE gates."""
from functools import lru_cache
from pathlib import Path
import os
import hashlib
import torch
from sparsity_research.sites import FixedSymmetricThreshold
'''
        + ast.unparse(ast.Module(body=base, type_ignores=[]))
        + "\n"
    )
    save("kernels/rotary/implementation.py", public_names(text).encode())
    record(
        rotary,
        "kernels/rotary/implementation.py",
        [
            "Retain extension and RopeGate call; support absent gates using final adapter constructor"
        ],
    )
    record(
        large / "hz_adapter.py",
        "kernels/rotary/implementation.py",
        ["Extract final RopeGate constructor"],
    )
    cuda(rotary.with_name("kernel.cu"), "kernels/rotary/kernel.cu")
    extract(
        run(26) / "autoresearch/candidates/k018/candidate.py",
        "kernels/block_forward.py",
        ("identity", "layer_forward"),
        "Parallel attention/FFN branches and rounded residual additions",
    )
    extract(
        small / "k020/candidate.py",
        "kernels/attention_forward.py",
        ("attention_forward",),
        "Attention dispatch preserving existing activation taps and gates",
    )
    extract(
        large / "controls.py",
        "kernels/controls.py",
        ("NativeJoint",),
        "Same-checkpoint native output-projection control",
    )

    source = run(37) / "site_controls.py"
    body = nodes(source, ("extension", "Joint", "Attention"))
    text = (
        '''"""Independent execution switches for the published six-path ablation."""
from functools import lru_cache
from pathlib import Path
import hashlib
import os
import torch
from support import ROOT, module, sha256 as sha
HERE = Path(__file__).resolve().parent
base_joint = module('output_projection_base', ROOT/'model_14m/output_projection.py')
'''
        + ast.unparse(ast.Module(body=body, type_ignores=[]))
        + "\n"
    )
    text = text.replace("RUN / 'candidate'", "HERE / 'cuda'").replace(
        "R28 / 'runtime/vendor", "ROOT / 'runtime/vendor"
    )
    text = text.replace("RUN037_SKIP_", "SPARSE_SKIP_")
    save("kernels/ablation/implementation.py", public_names(text).encode())
    record(
        source,
        "kernels/ablation/implementation.py",
        ["Extract only final ablation classes and extension; direct component import"],
    )
    for p in (run(37) / "candidate").iterdir():
        if p.is_file() and (
            p.suffix in {".cu", ".h", ".cuh", ".hpp"} or p.name.startswith("LICENSE")
        ):
            cuda(p, "kernels/ablation/cuda/" + p.name)
    for source, name in [
        (run(37) / "controls.py", "modes"),
        (run(37) / "diagnostics.py", "diagnostics"),
    ]:
        text = source.read_text(encoding="utf-8").replace(
            "from common import write_json", "from support import write_json"
        )
        save("kernels/ablation/" + name + ".py", public_names(text).encode())
        record(
            source,
            "kernels/ablation/" + name + ".py",
            ["Rename private attributes and relocate imports"],
        )
