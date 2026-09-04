"""Bounded post-mortem for the released raw ELL kernel's output-width boundary."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


RUN_DIR = Path(
    "/workspace/sparsity-spillover/"
    "runs/022-2026-09-04-pythia14m-sparse-kernel-a0-baseline"
)
UPSTREAM = Path("/workspace/run022-state/sparser-faster-llms")
OUTPUT = Path("/workspace/run022-state/postmortem-raw-ell-widths.json")
sys.path.insert(0, str(RUN_DIR))

from benchmark_core import pack_exact_ell, relative_errors  # noqa: E402


os.environ["TORCH_CUDA_ARCH_LIST"] = "9.0a"
loader_path = UPSTREAM / "custom_models/hybrid_modules/sparse_ops_loader.py"
spec = importlib.util.spec_from_file_location("run022_postmortem_loader", loader_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"Cannot load {loader_path}")
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
loader.load_sparse_ops(verbose=False)

device = torch.device("cuda:0")
generator = torch.Generator(device=device).manual_seed(22022)
matrix = torch.randn((256, 512), generator=generator, device=device, dtype=torch.bfloat16)
keep = torch.rand((256, 512), generator=generator, device=device) < 0.25
matrix.masked_fill_(~keep, 0)
matrix[0].zero_()
matrix[1] = torch.randn((512,), generator=generator, device=device, dtype=torch.bfloat16)
values, indices, counts, stride = pack_exact_ell(matrix, torch=torch, alignment=8)


def evaluate(rhs: torch.Tensor) -> dict[str, float]:
    width = rhs.shape[1]
    actual = torch.ops.sparse_ops.ell_spmm_raw(
        values, indices, counts, rhs, 256, 512, width, stride, stride
    )
    reference = matrix.float() @ rhs.float()
    torch.cuda.synchronize()
    return relative_errors(actual, reference, torch=torch)


widths = {}
rhs_128 = None
for width in (128, 256, 512, 2048):
    rhs = torch.randn((512, width), generator=generator, device=device, dtype=torch.bfloat16).contiguous()
    if width == 128:
        rhs_128 = rhs
    widths[str(width)] = evaluate(rhs)

assert rhs_128 is not None
padded = torch.zeros((512, 256), device=device, dtype=torch.bfloat16)
padded[:, :128] = rhs_128
padded_actual = torch.ops.sparse_ops.ell_spmm_raw(
    values, indices, counts, padded, 256, 512, 256, stride, stride
)[:, :128]
padded_reference = matrix.float() @ rhs_128.float()
torch.cuda.synchronize()

result = {
    "schema_version": 1,
    "purpose": "postmortem_only_not_run022_benchmark",
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "compute_capability": list(torch.cuda.get_device_capability()),
    "shape": {"M": 256, "K": 512, "row_density": 0.25, "ell_stride": stride},
    "direct_widths": widths,
    "padded_128_through_256": relative_errors(padded_actual, padded_reference, torch=torch),
}
OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(result, indent=2, sort_keys=True))
