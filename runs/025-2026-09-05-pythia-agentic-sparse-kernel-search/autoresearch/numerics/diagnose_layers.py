"""Candidate-only P0 numerical diagnosis; no timing or validation qualification.

Run on the retained A1-H-14M checkpoint with the original config/gates unchanged.
Hooks execute isolated linears on *native* inputs and never replace native values.
The optional prefix sweep then measures propagation through the unchanged model.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.nn import functional as F

RUN = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(RUN))
from measurement import numerical_gate
from p0 import Adapter
from run025_common import ROOT, inside, read_json, record, verify_record, write_json


def compare(reference, actual, gate):
    result = numerical_gate(reference, actual, **gate)
    delta = (actual.float() - reference.float()).abs()
    excess = delta - (gate["atol"] + gate["rtol"] * reference.float().abs())
    result.update(elements=reference.numel(), unequal_elements=int((actual != reference).sum()),
                  violating_elements=int((excess > 0).sum()),
                  maximum_tolerance_excess=float(excess.max()))
    return result


def fp64_samples(x, weight, bias, native, p0, oracle, count=8):
    """Only selected scalar dots, not a full slow FP64 matrix multiplication."""
    flat = x.reshape(-1, x.shape[-1])
    difference = (p0.float() - native.float()).abs().reshape(-1)
    count = min(count, int((difference != 0).sum()))
    if not count:
        return []
    indices = difference.topk(count).indices
    n = weight.shape[0]
    rows, cols = indices // n, indices % n
    values = (flat[rows].double() * weight[cols].double()).sum(-1)
    if bias is not None:
        values += bias[cols].double()
    rounded = values.bfloat16()
    result = []
    for j, index in enumerate(indices.tolist()):
        result.append({"flat_index": index, "row": int(rows[j]), "column": int(cols[j]),
                       "fp64_dot_with_bias": float(values[j]), "fp64_rounded_bf16": float(rounded[j]),
                       "native": float(native.reshape(-1)[index]),
                       "p0": float(p0.reshape(-1)[index]),
                       "fp32_fused": float(oracle.reshape(-1)[index])})
    return result


def fused_oracle(x, weight_t, bias, packed, output):
    result = x.float() @ weight_t.float()
    if bias.numel():
        result += bias.float()
    output.copy_(result.bfloat16())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", default="14m/a1h")
    parser.add_argument("--blocks", type=int, nargs="+", default=[1])
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--prefix-sweep", action="store_true", help="One extra full forward per projection")
    parser.add_argument("--seconds", type=int, default=600)
    args = parser.parse_args()
    if args.seconds <= 0 or not args.blocks or any(block < 0 or block >= 338 for block in args.blocks):
        raise ValueError("Invalid diagnostic time/block scope")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; helper unit tests are not deployment evidence")
    import transformers
    sys.path.insert(0, str(ROOT / "src"))
    from sparsity_research.pythia import load_checkpoint_pythia

    destination = inside(RUN / "artifacts", args.attempt)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def check():
        if time.monotonic() - started > args.seconds:
            raise TimeoutError("Diagnostic deadline; partial evidence is incomplete")
    def save(stage, **fields):
        write_json(destination / "status.json", {"stage": stage, "elapsed_seconds": time.monotonic() - started, **fields})
        print(stage, fields, flush=True)

    cfg = read_json(RUN / "config.json")
    manifest = read_json(RUN / "prelaunch/input_manifest.json")
    condition = next(row for row in manifest["checkpoints"] if row["id"] == args.condition)
    for item in condition["files"] + condition["provenance"]:
        verify_record(item)
    tokens = np.memmap(verify_record(manifest["validation"]), dtype=np.int32, mode="r")
    gates = {kind: {"relative_l2": cfg["calibration"][f"{kind}_relative_l2"],
                    "atol": cfg["calibration"][f"{kind}_atol"],
                    "rtol": cfg["calibration"][f"{kind}_rtol"]} for kind in ("primitive", "logit")}
    torch.manual_seed(2503)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, ROOT / condition["checkpoint"], torch=torch)
    model = model.cuda().bfloat16().eval()
    model.set_attn_implementation("sdpa")
    model.config.use_cache = False
    adapter = Adapter(model)
    write_json(destination / "identity.json", {
        "condition": args.condition, "blocks": args.blocks, "sequence_length": 2048,
        "scope": "selected-block correctness diagnosis; not full validation and not performance",
        "sources": [record(Path(__file__)), record(RUN / "p0.py"), record(RUN / "kernels/twell_pythia.cu"), record(RUN / "config.json")],
        "torch": torch.__version__, "transformers": transformers.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "checkpoint": condition, "gates": gates})
    with torch.inference_mode():
        for block in args.blocks:
            check()
            ids = torch.tensor(np.array(tokens[block * 2048:(block + 1) * 2048]), device="cuda", dtype=torch.long)[None]
            layer_rows, execution_order = [], []
            handles = []
            adapter.set_mode("native")
            for ordinal, (_, name, original, wrapped, site) in enumerate(adapter.entries):
                def hook(module, inputs, output, ordinal=ordinal, wrapped=wrapped, name=name, site=site):
                    check()
                    x = inputs[0]
                    preserved = x.clone()
                    wrapped.mode = "p0"
                    direct = wrapped(x).clone()
                    flat = x.reshape(-1, x.shape[-1])
                    oracle = F.linear(flat.float(), module.weight.float(), module.bias.float() if module.bias is not None else None).bfloat16().view_as(output)
                    bf16_repeat = F.linear(x, module.weight, module.bias)
                    row = {"ordinal": ordinal, "execution_index": len(execution_order), "layer": ordinal // 4,
                           "module": name, "site": site, "input_shape": list(x.shape),
                           "input_zero_count": int((x == 0).sum()), "input_elements": x.numel(),
                           "input_max_abs": float(x.abs().max()), "input_unchanged": bool(torch.equal(x, preserved)),
                           "native_repeat": compare(output, bf16_repeat, gates["primitive"]),
                           "p0_vs_native": compare(output, direct, gates["primitive"]),
                           "fp32_fused_vs_native": compare(output, oracle, gates["primitive"]),
                           "p0_vs_fp32_fused": compare(oracle, direct, gates["primitive"]),
                           "fp64_selected_dots": fp64_samples(x, module.weight, module.bias, output, direct, oracle)}
                    layer_rows.append(row)
                    execution_order.append(ordinal)
                    write_json(destination / f"block-{block}-layers.json", layer_rows)
                handles.append(original.register_forward_hook(hook))
            try:
                reference = model(input_ids=ids, use_cache=False).logits.clone()
            finally:
                for handle in handles:
                    handle.remove()
            save("isolated_linears_complete", block=block, operations=len(layer_rows))

            comparisons = []
            def evaluate(label):
                check()
                actual = model(input_ids=ids, use_cache=False).logits.clone()
                comparisons.append({"label": label, **compare(reference, actual, gates["logit"])})
                write_json(destination / f"block-{block}-propagation.json", comparisons)
                return actual
            adapter.set_mode("native")
            evaluate("native_repeat_after_probe")
            adapter.set_mode("adapter_dense")
            evaluate("adapter_dense")
            adapter.set_mode("p0")
            p0_logits = evaluate("p0")
            # A clone after *each* adapted operation falsifies output-buffer aliasing.
            clone_hooks = [wrapped.register_forward_hook(lambda module, inputs, output: output.clone())
                           for _, _, _, wrapped, _ in adapter.entries]
            try:
                cloned_logits = evaluate("p0_with_cloned_projection_outputs")
            finally:
                for handle in clone_hooks:
                    handle.remove()
            write_json(destination / f"block-{block}-alias.json", compare(p0_logits, cloned_logits, gates["logit"]))
            del p0_logits, cloned_logits
            for _, _, _, wrapped, _ in adapter.entries:
                wrapped.backend = fused_oracle
            evaluate("fp32_fused_oracle")
            for _, _, _, wrapped, _ in adapter.entries:
                wrapped.backend = None
            if args.prefix_sweep:
                adapter.set_mode("native")
                for count, ordinal in enumerate(execution_order, 1):
                    parent, name, _, wrapped, _ = adapter.entries[ordinal]
                    wrapped.mode = "p0"
                    setattr(parent, name, wrapped)
                    evaluate(f"first_{count}_executed_projections_p0")
            adapter.set_mode("native")
            save("block_complete", block=block, comparisons=len(comparisons))
    save("complete", qualification="diagnosis only; fixed gates unchanged; not full validation")


if __name__ == "__main__":
    main()
