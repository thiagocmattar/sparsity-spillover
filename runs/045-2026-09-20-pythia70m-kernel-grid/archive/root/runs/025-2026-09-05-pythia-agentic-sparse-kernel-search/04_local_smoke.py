"""Retained 14M checkpoint round-trip on CPU. Explicitly NOT CUDA evidence."""
import sys
import time

import torch
from transformers import AutoModelForCausalLM

from run025_common import ROOT, RUN, read_json, verify_record, write_json
sys.path.insert(0, str(ROOT / "src"))
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata
from measurement import numerical_gate
from p0 import Adapter


def reference_backend(x, weight, bias, packed, out):
    # Wire-format round trips are tested separately; this checks integration math.
    value = x.float() @ weight.float()
    if bias.numel():
        value += bias.float()
    out.copy_(value)


def main():
    torch.set_num_threads(4)
    inputs = torch.arange(8).unsqueeze(0)
    manifest = read_json(RUN / "prelaunch/input_manifest.json")
    rows = []
    for name in ("14m/a0", "14m/a1h", "14m/a4-0p5", "14m/a7-0p5"):
        row = next(row for row in manifest["checkpoints"] if row["id"] == name)
        for file in row["files"]:
            verify_record(file)
        before = time.monotonic()
        model = load_checkpoint_pythia(AutoModelForCausalLM, ROOT / row["checkpoint"], torch=torch).bfloat16().eval()
        model.set_attn_implementation("sdpa")
        topology = topology_metadata(model)
        adapter = Adapter(model, backend=reference_backend)
        with torch.inference_mode():
            adapter.set_mode("native")
            reference = model(input_ids=inputs, use_cache=False).logits.clone()
            adapter.set_mode("adapter_dense")
            dense_identical = torch.equal(reference, model(input_ids=inputs, use_cache=False).logits)
            adapter.set_mode("p0")
            actual = model(input_ids=inputs, use_cache=False).logits
            gate = numerical_gate(reference, actual, relative_l2=.02, atol=.25, rtol=.02)
            adapter.set_mode("native")
            restored_identical = torch.equal(reference, model(input_ids=inputs, use_cache=False).logits)
        result = {"condition": name, "device": "cpu", "batch": 1, "sequence_length": 8,
                  "backend": "FP32 mathematical oracle, NOT compiled CUDA", "topology": topology,
                  "dense_adapter_bitwise_identical": dense_identical, "native_restored_bitwise_identical": restored_identical,
                  "gate": gate, "elapsed_seconds": time.monotonic() - before}
        rows.append(result)
        write_json(RUN / "prelaunch/local-smoke.json", rows)
        print(f"{name}: {result}", flush=True)
        if not (dense_identical and restored_identical and gate["pass"]):
            raise RuntimeError(f"CPU integration smoke failed: {name}")


if __name__ == "__main__":
    main()
