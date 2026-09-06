"""GPU-only primitive qualification for K002; no launches or model selection."""
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
from measurement import numerical_gate
from run025_common import config, inside, record, write_json

spec = importlib.util.spec_from_file_location("run025_k002_qualification", HERE / "attention.py")
K002 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(K002)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--block-m", type=int, choices=[16, 32], default=16)
    parser.add_argument("--max-length", type=int, choices=[129, 2048], default=2048)
    args = parser.parse_args()
    out = inside(HERE / "artifacts", args.attempt)
    out.mkdir(parents=True, exist_ok=False)
    rows = []
    started = time.monotonic()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; no CPU replacement")
        cfg = config()["calibration"]
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        generator = torch.Generator().manual_seed(2502)
        lengths = [17, 129] + ([2048] if args.max_length == 2048 else [])
        patterns = ["zero_q", "zero_k", "zero_v", "signed_dense", "mixed_zero_tiles", "isolated_active_row"]
        write_json(out / "manifest.json", {"arguments": vars(args), "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "sources": [record(HERE / file) for file in ("attention.py", "kernels.py", "qualify.py")],
            "primitive_gates": {key: cfg[key] for key in ("primitive_relative_l2", "primitive_atol", "primitive_rtol")}})
        for d in (32, 64):
            for t in lengths:
                for pattern in patterns:
                    # Last-axis striding verifies the actual stride parameters.
                    tensors = [torch.randn(1, 2, t, d * 2, generator=generator).bfloat16().cuda()[..., ::2]
                               for _ in range(3)]
                    q, k, v = tensors
                    if pattern == "zero_q":
                        q.zero_()
                    elif pattern == "zero_k":
                        k.zero_()
                    elif pattern == "zero_v":
                        v.zero_()
                    elif pattern == "mixed_zero_tiles":
                        q[..., :min(t, args.block_m * 2), :] = 0
                    elif pattern == "isolated_active_row":
                        last = q[..., -1:, :].clone()
                        q.zero_()
                        q[..., -1:, :] = last
                    workspace = K002.Workspace(q, args.block_m)
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    before = time.monotonic()
                    with torch.cuda.stream(stream), torch.inference_mode():
                        actual = K002.attention(q, k, v, block_m=args.block_m, workspace=workspace).clone()
                        native = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                        with sdpa_kernel(SDPBackend.MATH):
                            oracle = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), is_causal=True)
                    stream.synchronize()
                    elapsed = time.monotonic() - before
                    bounds = {"relative_l2": cfg["primitive_relative_l2"],
                              "atol": cfg["primitive_atol"], "rtol": cfg["primitive_rtol"]}
                    native_gate = numerical_gate(native, actual, **bounds)
                    fp32_gate = numerical_gate(oracle, actual, **bounds)
                    expected_flags = []
                    for head in range(q.shape[1]):
                        for offset in range(0, t, args.block_m):
                            expected_flags.append(bool((q[0, head, offset:offset + args.block_m] == 0).all()))
                    flags = workspace.fast_tiles.cpu().reshape(-1).tolist()
                    row = {"d": d, "t": t, "pattern": pattern, "block_m": args.block_m,
                           "elapsed_including_first_compile_seconds": elapsed,
                           "native_gate": native_gate, "fp32_gate": fp32_gate,
                           "workspace_bytes": workspace.bytes, "fast_tiles": sum(flags),
                           "total_tiles": len(flags), "flags_correct": flags == expected_flags}
                    row["pass"] = native_gate["pass"] and fp32_gate["pass"] and row["flags_correct"]
                    rows.append(row)
                    write_json(out / "primitive-gates.json", rows)
                    print(json.dumps(row), flush=True)
                    if not row["pass"]:
                        raise RuntimeError(f"K002 primitive failed: D{d}, T{t}, {pattern}")
                    # Reuse the same scratch on changed Q/V to expose stale state.
                    with torch.inference_mode():
                        q.zero_()
                        v.mul_(-1)
                        repeated = K002.attention(q, k, v, block_m=args.block_m, workspace=workspace).clone()
                        expected = v.float().cumsum(-2) / torch.arange(1, t + 1, device="cuda").view(1, 1, -1, 1)
                        repeated_gate = numerical_gate(expected, repeated, **bounds)
                    row["reused_workspace_gate"] = repeated_gate
                    row["pass"] &= repeated_gate["pass"]
                    write_json(out / "primitive-gates.json", rows)
                    if not row["pass"]:
                        raise RuntimeError("K002 stale workspace or prefix discrepancy")
                    if pattern == "signed_dense" and t == 129:
                        # A graph replay must observe changed operands, not the
                        # sparsity pattern encountered at capture time.
                        with torch.inference_mode():
                            graph_stream = torch.cuda.Stream()
                            graph_stream.wait_stream(torch.cuda.current_stream())
                            with torch.cuda.stream(graph_stream):
                                for _ in range(3):
                                    K002.attention(q, k, v, block_m=args.block_m, workspace=workspace)
                            torch.cuda.current_stream().wait_stream(graph_stream)
                            torch.cuda.synchronize()
                            graph = torch.cuda.CUDAGraph()
                            with torch.cuda.graph(graph):
                                captured = K002.attention(q, k, v, block_m=args.block_m, workspace=workspace)
                            q.fill_(.5)
                            v.mul_(-1)
                            graph.replay()
                            fresh_native = F.scaled_dot_product_attention(q, k, v, is_causal=True)
                            graph_gate = numerical_gate(fresh_native, captured, **bounds)
                        row["changed_input_graph_gate"] = graph_gate
                        row["pass"] &= graph_gate["pass"]
                        write_json(out / "primitive-gates.json", rows)
                        if not row["pass"]:
                            raise RuntimeError("K002 changed-input graph replay failed")
        write_json(out / "status.json", {"status": "complete", "cases": len(rows),
                   "elapsed_seconds": time.monotonic() - started, "pass": all(row["pass"] for row in rows)})
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed_or_incomplete", "error": str(error),
                   "elapsed_seconds": time.monotonic() - started})
        (out / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
