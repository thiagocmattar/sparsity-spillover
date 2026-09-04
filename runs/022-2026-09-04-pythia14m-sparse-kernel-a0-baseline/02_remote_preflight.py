#!/usr/bin/env python3
"""Compile and test the upstream raw ELL operator on the selected Hopper GPU."""

from __future__ import annotations

import argparse
import importlib.util
import os
import platform
import subprocess
from pathlib import Path

import torch

from benchmark_core import pack_exact_ell, relative_errors, unpack_exact_ell, validate_raw_ell_contract
from run_config import RUN_DIR, load_config, write_json


def _gpu_snapshot() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=name,uuid,driver_version,memory.total,power.limit",
        "--format=csv,noheader,nounits",
    ]
    return subprocess.check_output(command, text=True).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=RUN_DIR / "prelaunch" / "remote-preflight.json")
    args = parser.parse_args()
    config = load_config()
    expected = config["runtime"]
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required.")
    device = torch.device("cuda:0")
    capability = torch.cuda.get_device_capability(device)
    if tuple(capability) != tuple(expected["required_compute_capability"]):
        raise RuntimeError(f"Exact Hopper compute capability 9.0 required, got {capability}.")
    if torch.__version__.split("+")[0] != expected["pythia_torch"]:
        raise RuntimeError(f"PyTorch mismatch: {torch.__version__}")
    cuda_version = tuple(int(part) for part in str(torch.version.cuda).split(".")[:2])
    if cuda_version < (12, 8):
        raise RuntimeError(f"CUDA runtime >= 12.8 required, got {torch.version.cuda}.")
    os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
    loader_path = args.upstream_dir.resolve() / "custom_models" / "hybrid_modules" / "sparse_ops_loader.py"
    specification = importlib.util.spec_from_file_location("run022_sparse_ops_loader", loader_path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"Could not load upstream operator loader: {loader_path}")
    loader = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(loader)
    loader.load_sparse_ops(verbose=True)

    generator = torch.Generator(device=device).manual_seed(22022)
    matrix = torch.randn((256, 512), generator=generator, device=device, dtype=torch.bfloat16)
    keep = torch.rand((256, 512), generator=generator, device=device) < 0.25
    matrix.masked_fill_(~keep, 0)
    matrix[0].zero_()
    matrix[1] = torch.randn((512,), generator=generator, device=device, dtype=torch.bfloat16)
    rhs = torch.randn((512, 128), generator=generator, device=device, dtype=torch.bfloat16).contiguous()
    bias = torch.randn((128,), generator=generator, device=device, dtype=torch.bfloat16)
    values, indices, counts, stride = pack_exact_ell(matrix, torch=torch, alignment=8)
    contract = validate_raw_ell_contract(
        values,
        indices,
        counts,
        rhs,
        overflow_threshold=stride,
        torch=torch,
    )
    reconstructed = unpack_exact_ell(values, indices, counts, columns=512, torch=torch)
    if not bool(torch.equal(matrix, reconstructed)):
        raise RuntimeError("Exact ELL pack/unpack round trip failed.")
    dense = matrix @ rhs + bias
    sparse = torch.ops.sparse_ops.ell_spmm_raw(
        values, indices, counts, rhs, 256, 512, 128, stride, stride
    ) + bias
    reference = matrix.float() @ rhs.float() + bias.float()
    torch.cuda.synchronize()
    errors = {
        "sparse_vs_dense_bf16": relative_errors(sparse, dense, torch=torch),
        "sparse_vs_fp32": relative_errors(sparse, reference, torch=torch),
        "dense_bf16_vs_fp32": relative_errors(dense, reference, torch=torch),
    }
    tolerance = config["measurement"]["correctness_relative_l2_tolerance"]
    if errors["sparse_vs_fp32"]["relative_l2"] > tolerance:
        raise RuntimeError(f"Raw ELL relative-L2 error exceeds {tolerance}: {errors}")
    result = {
        "schema_version": 1,
        "passed": True,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "compute_capability": list(capability),
        "gpu": _gpu_snapshot(),
        "upstream_directory": str(args.upstream_dir.resolve()),
        "operator": "torch.ops.sparse_ops.ell_spmm_raw",
        "contract": contract,
        "overflow_threshold": stride,
        "skipped_rows": 0,
        "exact_pack_round_trip": True,
        "errors": errors,
    }
    write_json(args.output, result)
    print(f"PASS: remote raw-ELL preflight -> {args.output}")


if __name__ == "__main__":
    main()
