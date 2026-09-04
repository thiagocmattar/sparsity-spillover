#!/usr/bin/env python3
"""Compile and correctness-gate every derived ELL shape on one Hopper GPU."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import subprocess
from pathlib import Path

import torch

from benchmark_core import (
    pack_exact_ell_reference,
    relative_errors,
    unpack_exact_ell,
    validate_exact_ell,
)
from run_config import RUN_DIR, load_config, write_json


SHAPES = (
    ("run022_n128_regression", 256, 512, 128),
    ("attention_output_projection", 256, 512, 512),
    ("qkv_projection", 256, 512, 1536),
    ("mlp_w1", 256, 512, 2048),
    ("mlp_w2", 256, 2048, 512),
    ("attention_qk_q_left", 2048, 64, 2048),
    ("attention_pv_vt_left", 64, 2048, 2048),
)


def gpu_snapshot() -> str:
    return subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    ).strip()


def load_sparse_ops(upstream_dir: Path) -> None:
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    loader_path = upstream_dir.resolve() / "custom_models" / "hybrid_modules" / "sparse_ops_loader.py"
    specification = importlib.util.spec_from_file_location("run023_sparse_ops_loader", loader_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load derived operator loader: {loader_path}")
    loader = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loader)
    loader.load_sparse_ops(verbose=True)


def shape_check(name: str, rows: int, inner: int, columns: int, *, generator: torch.Generator, tolerance: float) -> dict:
    device = torch.device("cuda:0")
    matrix = torch.randn((rows, inner), generator=generator, device=device, dtype=torch.bfloat16)
    keep = torch.rand((rows, inner), generator=generator, device=device) < 0.35
    matrix.masked_fill_(~keep, 0)
    matrix[0].zero_()
    if rows > 1:
        matrix[1] = -torch.rand((inner,), generator=generator, device=device, dtype=torch.bfloat16)
    if rows > 2:
        matrix[2] = torch.randn((inner,), generator=generator, device=device, dtype=torch.bfloat16)
    rhs = torch.randn((inner, columns), generator=generator, device=device, dtype=torch.bfloat16).contiguous()

    values = torch.empty_like(matrix)
    indices = torch.empty_like(matrix, dtype=torch.uint16)
    counts = torch.empty((rows,), dtype=torch.int32, device=device)
    output = torch.empty((rows, columns), dtype=torch.bfloat16, device=device)
    torch.ops.sparse_ops.dense_to_ell_exact_out(matrix, values, indices, counts)
    contract = validate_exact_ell(matrix, values, indices, counts, torch=torch)
    reference_values, reference_indices, reference_counts = pack_exact_ell_reference(matrix, torch=torch)
    valid = torch.arange(inner, device=device).unsqueeze(0) < counts.unsqueeze(1)
    if not bool(torch.equal(counts, reference_counts)):
        raise RuntimeError(f"{name}: CUDA and reference row counts differ.")
    if not bool(torch.equal(values[valid], reference_values[valid])):
        raise RuntimeError(f"{name}: CUDA pack changed signed values.")
    if not bool(
        torch.equal(
            indices.to(torch.int32)[valid],
            reference_indices.to(torch.int32)[valid],
        )
    ):
        raise RuntimeError(f"{name}: CUDA pack changed column order.")

    torch.ops.sparse_ops.ell_spmm_raw_out(
        values, indices, counts, rhs, rows, inner, columns, inner, inner, output
    )
    allocated_output = torch.ops.sparse_ops.ell_spmm_raw(
        values, indices, counts, rhs, rows, inner, columns, inner, inner
    )
    torch.cuda.synchronize()
    dense_bf16 = matrix @ rhs
    reference_fp32 = matrix.float() @ rhs.float()
    errors = {
        "out_vs_fp32": relative_errors(output, reference_fp32, torch=torch),
        "out_vs_dense_bf16": relative_errors(output, dense_bf16, torch=torch),
        "allocated_vs_reused_out": relative_errors(allocated_output, output, torch=torch),
    }
    if errors["out_vs_fp32"]["relative_l2"] > tolerance:
        raise RuntimeError(f"{name}: relative-L2 error exceeded {tolerance}: {errors}")
    if errors["allocated_vs_reused_out"]["maximum_absolute"] != 0:
        raise RuntimeError(f"{name}: allocating and reusable output paths differ.")

    # A second launch into the same buffers proves that no stale workspace values
    # are consumed, including the explicitly empty first row.
    matrix.mul_(-1)
    torch.ops.sparse_ops.dense_to_ell_exact_out(matrix, values, indices, counts)
    torch.ops.sparse_ops.ell_spmm_raw_out(
        values, indices, counts, rhs, rows, inner, columns, inner, inner, output
    )
    torch.cuda.synchronize()
    reuse_error = relative_errors(output, matrix.float() @ rhs.float(), torch=torch)
    if reuse_error["relative_l2"] > tolerance:
        raise RuntimeError(f"{name}: reusable workspace failed: {reuse_error}")
    return {
        "name": name,
        "M": rows,
        "K": inner,
        "N": columns,
        "contract": contract,
        "errors": errors,
        "workspace_reuse_error": reuse_error,
        "negative_values_preserved": True,
        "empty_row_preserved": bool((output[0] == 0).all().item()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RUN_DIR / "prelaunch" / "remote-preflight.json")
    args = parser.parse_args()
    config = load_config()
    expected = config["runtime"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    capability = torch.cuda.get_device_capability(0)
    if tuple(capability) != tuple(expected["compute_capability"]):
        raise RuntimeError(f"Exact Hopper compute capability 9.0 required, got {capability}.")
    if torch.__version__.split("+")[0] != expected["pythia_torch"]:
        raise RuntimeError(f"PyTorch mismatch: {torch.__version__}")
    cuda_version = tuple(int(part) for part in str(torch.version.cuda).split(".")[:2])
    if cuda_version < (12, 8):
        raise RuntimeError(f"CUDA runtime >= 12.8 required, got {torch.version.cuda}.")
    load_sparse_ops(args.upstream_dir)
    generator = torch.Generator(device="cuda").manual_seed(23023)
    tolerance = config["measurement"]["correctness_relative_l2_tolerance"]
    records = [
        shape_check(name, rows, inner, columns, generator=generator, tolerance=tolerance)
        for name, rows, inner, columns in SHAPES
    ]
    if records[0]["errors"]["out_vs_fp32"]["relative_l2"] > tolerance:
        raise RuntimeError("The Run-022 N=128 regression remains unfixed.")
    result = {
        "schema_version": 1,
        "passed": True,
        "derivative_label": config["upstream"]["derivative_label"],
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": list(capability),
        "gpu": gpu_snapshot(),
        "upstream_directory": str(args.upstream_dir.resolve()),
        "operators": [
            "sparse_ops::dense_to_ell_exact_out",
            "sparse_ops::ell_spmm_raw_out",
        ],
        "shape_checks": records,
        "run022_n128_regression_fixed": True,
    }
    write_json(args.output, result)
    print(f"PASS: derived ELL remote preflight ({len(records)} shapes) -> {args.output}")


if __name__ == "__main__":
    main()
