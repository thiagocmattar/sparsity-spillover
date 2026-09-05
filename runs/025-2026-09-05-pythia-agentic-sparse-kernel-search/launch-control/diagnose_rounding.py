"""Read-only diagnosis of the first A1-H calibration gate failure; no timing claim."""
from pathlib import Path
import sys
import numpy as np
import torch
import transformers

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN))
from run025_common import ROOT, read_json, write_json, verify_record
from p0 import Adapter
from measurement import numerical_gate
sys.path.insert(0, str(ROOT / "src"))
from sparsity_research.pythia import load_checkpoint_pythia


def fp32_fused(x, weight, bias, packed, output):
    output.copy_((x.float() @ weight.float() + bias.float()).bfloat16())


def fp32_split(x, weight, bias, packed, output):
    output.copy_((x.float() @ weight.float()).bfloat16() + bias)


def main():
    out = RUN / "artifacts/rounding-diagnostic-001"
    out.mkdir(exist_ok=False)
    cfg = read_json(RUN / "config.json")["calibration"]
    manifest = read_json(RUN / "prelaunch/input_manifest.json")
    row = next(r for r in manifest["checkpoints"] if r["id"] == "14m/a1h")
    for item in row["files"] + row["provenance"]:
        verify_record(item)
    model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, ROOT / row["checkpoint"], torch=torch)
    model = model.cuda().bfloat16().eval()
    model.set_attn_implementation("sdpa")
    model.config.use_cache = False
    adapter = Adapter(model)
    tokens = np.memmap(verify_record(manifest["validation"]), mode="r", dtype=np.int32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    results = []
    with torch.inference_mode():
        for block in (0, 1):
            ids = torch.tensor(np.array(tokens[block*2048:(block+1)*2048]), device="cuda", dtype=torch.long)[None]
            adapter.set_mode("native")
            reference = model(input_ids=ids, use_cache=False).logits.clone()
            for label, mode, backend in (("adapter_dense", "adapter_dense", None),
                                         ("p0", "p0", None),
                                         ("fp32_fused_oracle", "p0", fp32_fused),
                                         ("fp32_split_bias_oracle", "p0", fp32_split)):
                for _, _, _, wrapped, _ in adapter.entries:
                    wrapped.backend = backend
                adapter.set_mode(mode)
                actual = model(input_ids=ids, use_cache=False).logits
                gate = numerical_gate(reference, actual, relative_l2=cfg["logit_relative_l2"],
                                      atol=cfg["logit_atol"], rtol=cfg["logit_rtol"])
                excess = (actual.float() - reference.float()).abs() - (cfg["logit_atol"] + cfg["logit_rtol"] * reference.float().abs())
                results.append({"block": block, "variant": label, **gate,
                                "violating_logits": int((excess > 0).sum()),
                                "maximum_tolerance_excess": float(excess.max())})
                write_json(out / "results.json", results)
                print(results[-1], flush=True)
    write_json(out / "complete.json", {"complete": True, "purpose": "diagnosis only; original gates unchanged"})


if __name__ == "__main__":
    main()
