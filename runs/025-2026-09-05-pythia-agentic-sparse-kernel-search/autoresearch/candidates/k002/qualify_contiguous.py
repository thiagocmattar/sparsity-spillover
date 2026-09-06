"""Append-only K002 full-length native-Flash qualification, separate from code-001.

Synthetic primitives only: no model quality or full-model speedup claim follows.
No infrastructure operations. Compilation needs an external bounded timeout.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

HERE = Path(__file__).resolve().parent
RUN = HERE.parents[2]
sys.path.insert(0, str(RUN))
from measurement import numerical_gate, paired_timing, timing_summary
from run025_common import config, inside, record, write_json

SPEC = importlib.util.spec_from_file_location("run025_k002_contiguous", HERE / "attention.py")
K002 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(K002)


def make_inputs(heads, dimension, pattern, phase, generator):
    q, k, v = [torch.randn(1, heads, 2048, dimension, generator=generator).bfloat16() for _ in range(3)]
    if pattern == "all_zero_q":
        q.zero_()
    elif pattern == "alternating_zero_tiles":
        q.reshape(1, heads, 128, 16, dimension)[:, :, ::2] = 0
    elif pattern == "single_active_row":
        active = q[:, :, 1024:1025].clone()
        q.zero_()
        q[:, :, 1024:1025] = active
    elif pattern != "signed_dense":
        raise ValueError(pattern)
    if phase:
        q = q.roll(1, dims=-2)
        v = -v.roll(3, dims=-2)
    return q, k, v


def strided_copy(tensor):
    backing = torch.empty((*tensor.shape[:-1], tensor.shape[-1] * 2),
                          dtype=tensor.dtype, device=tensor.device)
    view = backing[..., ::2]
    view.copy_(tensor)
    return view


def expected_fast_tiles(q, block_m):
    # Explicit diagnostic sync outside all performance timing.
    return (q.reshape(q.shape[0], q.shape[1], 2048 // block_m, block_m, q.shape[-1]) == 0).all(-1).all(-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--block-m", type=int, choices=[16, 32], default=16)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("--passes", type=int, default=3)
    args = parser.parse_args()
    if args.seconds <= 0 or args.passes <= 0:
        raise ValueError("Positive duration/passes required")
    out = inside(HERE / "artifacts", args.attempt)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    rows = []

    def check():
        if time.monotonic() - started >= args.seconds:
            raise TimeoutError("Contiguous qualification budget exhausted")

    def emit(stage, **fields):
        value = {"stage": stage, "elapsed_seconds": time.monotonic() - started, **fields}
        write_json(out / "status.json", value)
        print(json.dumps(value), flush=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU performance substitute")
        cfg = config()["calibration"]
        bounds = {"relative_l2": cfg["primitive_relative_l2"],
                  "atol": cfg["primitive_atol"], "rtol": cfg["primitive_rtol"]}
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        write_json(out / "manifest.json", {"arguments": vars(args), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(), "bounds": bounds,
            "shape_scope": [[1, 4, 2048, 32], [1, 8, 2048, 64], [1, 16, 2048, 64]],
            "sources": [record(HERE / name) for name in ("attention.py", "kernels.py", "qualify_contiguous.py")],
            "evidence": "Synthetic attention primitives; no whole-model correctness or speedup claim",
            "references": ["native automatic SDPA on strided operands", "forced FLASH SDPA on contiguous operands"]})
        for heads, dimension in ((4, 32), (8, 64), (16, 64)):
            workspace = None
            timing_inputs = []
            for pattern in ("all_zero_q", "alternating_zero_tiles", "single_active_row", "signed_dense"):
                for phase in (0, 1):
                    check()
                    # Reset generator so phase 1 tests changed values/pattern,
                    # not an unrelated random realization.
                    generator = torch.Generator().manual_seed(2502 + heads)
                    contiguous = tuple(t.cuda().contiguous() for t in make_inputs(heads, dimension, pattern, phase, generator))
                    q, k, v = contiguous
                    strided = tuple(strided_copy(t) for t in contiguous)
                    if workspace is None:
                        workspace = K002.Workspace(q, args.block_m)
                    before = time.monotonic()
                    with torch.inference_mode():
                        native_strided = F.scaled_dot_product_attention(*strided, is_causal=True)
                        # This must fail visibly if FLASH is unsupported, never
                        # silently replace the reference with math SDPA.
                        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                            native_flash = F.scaled_dot_product_attention(*contiguous, is_causal=True)
                        actual = K002.attention(*contiguous, block_m=args.block_m, workspace=workspace).clone()
                        actual_strided = K002.attention(*strided, block_m=args.block_m, workspace=workspace).clone()
                    torch.cuda.synchronize()
                    gates = {"contiguous_vs_flash": numerical_gate(native_flash, actual, **bounds),
                             "strided_vs_flash": numerical_gate(native_flash, actual_strided, **bounds),
                             "strided_vs_native_strided": numerical_gate(native_strided, actual_strided, **bounds),
                             "native_backend_comparison": numerical_gate(native_strided, native_flash, **bounds)}
                    expected_flags = expected_fast_tiles(q, args.block_m)
                    flags_correct = bool(torch.equal(workspace.fast_tiles.bool(), expected_flags))
                    row = {"heads": heads, "dimension": dimension, "tokens": 2048,
                           "pattern": pattern, "changed_input_phase": phase,
                           "gates": gates, "flags_correct": flags_correct,
                           "fast_tiles": int(workspace.fast_tiles.sum()), "total_tiles": workspace.fast_tiles.numel(),
                           "workspace_bytes": workspace.bytes,
                           "seconds_including_compile_and_references": time.monotonic() - before,
                           "pass": flags_correct and all(gate["pass"] for gate in gates.values())}
                    rows.append(row)
                    write_json(out / "primitive-gates.json", rows)
                    emit("qualified_case", completed_cases=len(rows), total_cases=24, case=row)
                    if not row["pass"]:
                        raise RuntimeError("K002 contiguous/native-Flash fixed primitive gate failed")
                    if phase == 0:
                        timing_inputs.append(contiguous)
                    del native_strided, native_flash, actual, actual_strided, strided
            if args.timing:
                selected = ["native_flash"]
                def set_mode(mode):
                    selected[0] = mode
                def forward(inputs):
                    if selected[0] == "native_flash":
                        return F.scaled_dot_product_attention(*inputs, is_causal=True)
                    return K002.attention(*inputs, block_m=args.block_m, workspace=workspace)
                def save_sample(row):
                    with (out / f"timing-h{heads}-d{dimension}.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(row) + "\n")
                with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                    samples = paired_timing(forward, set_mode, timing_inputs,
                        modes=["native_flash", "k002"], passes=args.passes, warmups=4,
                        seed=2504, check_deadline=check, sample_sink=save_sample)
                write_json(out / f"timing-h{heads}-d{dimension}.json", {
                    "summary": timing_summary(samples, reference="native_flash"),
                    "input_order": ["all_zero_q", "alternating_zero_tiles", "single_active_row", "signed_dense"],
                    "measurement": "synthetic attention only; contiguous resident operands on both paths; no graph capture",
                    "included": "prefix construction, GPU branch detection, all attention operations",
                    "excluded": "input generation/staging, shape-static workspace allocation, compile/setup"})
            del timing_inputs, workspace
        emit("complete", cases=len(rows), all_fixed_gates_passed=True)
    except BaseException as error:
        emit("failed_or_incomplete", error=str(error), completed_cases=len(rows))
        (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
