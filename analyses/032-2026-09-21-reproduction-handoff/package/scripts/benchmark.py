"""One fresh-process full-validation qualification and paired CUDA graph benchmark."""

import argparse
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "kernels")]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--control", choices=["native-hz", "hz-skips-off"])
    p.add_argument("--hz-mode", choices=["A", "B", "C", "D"],
                   help="14M h,z factorial switches; all other kernels stay frozen")
    p.add_argument(
        "--operation-mode",
        choices=[
            "frozen",
            "full",
            "off",
            "projection",
            "without-a",
            "without-m",
            "without-h",
            "without-z",
            "without-qk",
            "without-pv",
        ],
    )
    p.add_argument(
        "--diagnostics",
        action="store_true",
        help="Additional untimed 14M actual-operand/MMA counter pass",
    )
    p.add_argument("--replicate", type=int, required=True, choices=[1, 2, 3])
    a = p.parse_args()
    import numpy as np
    import torch
    import transformers
    from sparsity_research.pythia import load_checkpoint_pythia
    from sparsity_research.data import file_sha256, FULL_VALIDATION_SHA256
    from training.config import register_hz
    from measurement import numerical_gate
    from timing import DenseRunner, paired_probe
    from install import install

    if torch.cuda.get_device_name() != "NVIDIA GeForce RTX 5090":
        raise RuntimeError("Paper timing contract requires RTX 5090")
    if (
        torch.__version__.split("+")[0],
        transformers.__version__,
        np.__version__,
        torch.version.cuda,
    ) != ("2.11.0", "5.12.1", "2.5.0", "12.8"):
        raise RuntimeError("Install the pinned kernel runtime")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
        raise RuntimeError("Paper uses the default cuBLAS workspace")
    register_hz()
    torch.manual_seed(2801)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    cache = ROOT / "data/tokenized/minipile-pythia-14m-full/validation/tokens.int32.bin"
    if file_sha256(cache) != FULL_VALIDATION_SHA256:
        raise ValueError("Validation hash mismatch")
    tokens = np.memmap(cache, dtype=np.int32, mode="r")
    if divmod(len(tokens), 2048) != (338, 1444):
        raise ValueError("Incomplete validation")
    a.output.mkdir(parents=True, exist_ok=False)

    def save(name, data):
        (a.output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")

    status = dict(
        status="running",
        replicate=a.replicate,
        backend="optimized",
        control=a.control,
        operation_mode=a.operation_mode,
        hz_mode=a.hz_mode,
        checkpoint=str(a.checkpoint),
        weight_sha256=file_sha256(a.checkpoint / "model.safetensors"),
        gpu=torch.cuda.get_device_name(),
        runtime=dict(
            torch=torch.__version__,
            transformers=transformers.__version__,
            cuda=torch.version.cuda,
        ),
    )
    save("result.json", status)

    def ids(i):
        return torch.tensor(
            tokens[i * 2048 : (i + 1) * 2048].copy(), device="cuda", dtype=torch.long
        )[None]

    def load(attention):
        m = (
            load_checkpoint_pythia(
                transformers.AutoModelForCausalLM, a.checkpoint, torch=torch
            )
            .to("cuda", dtype=torch.bfloat16)
            .eval()
        )
        m.set_attn_implementation(attention)
        m.config.use_cache = False
        return m

    try:
        eager = load("eager")
        native = load("sdpa")
        candidate = load("sdpa")
        losses = {k: [] for k in ("eager", "native", "candidate")}
        checks = []
        with torch.inference_mode():
            status["kernel"] = install(
                candidate, "optimized", a.control, a.operation_mode, a.hz_mode
            )
            runners = {
                k: DenseRunner(
                    lambda x, m=m: m(input_ids=x, use_cache=False).logits,
                    ids(0),
                    "graph",
                )
                for k, m in [("native", native), ("candidate", candidate)]
            }
            for runner in runners.values():
                runner.prepare()
            for i in range(338):
                x = ids(i)
                reference = eager(input_ids=x, use_cache=False).logits
                targets = x[:, 1:].reshape(-1)

                def loss(y):
                    return float(
                        torch.nn.functional.cross_entropy(
                            y[:, :-1].float().reshape(-1, 50304), targets
                        )
                    )

                losses["eager"].append(loss(reference))
                row = {"index": i}
                for name, runner in runners.items():
                    runner.stage(x)
                    y = runner()
                    row[name] = numerical_gate(
                        reference, y, relative_l2=0.02, atol=0.25, rtol=0.02
                    )
                    losses[name].append(loss(y))
                checks.append(row)
                if i % 32 == 0:
                    print(
                        json.dumps({"qualification_blocks": i + 1, "total": 338}),
                        flush=True,
                    )
            means = {k: math.fsum(v) / len(v) for k, v in losses.items()}
            qualified = all(r[m]["pass"] for r in checks for m in runners) and all(
                abs(means[m] - means["eager"]) <= 0.001 for m in runners
            )
            save(
                "qualification.json",
                dict(
                    blocks=338,
                    documents=500,
                    excluded_tail=1444,
                    mean_losses=means,
                    checks=checks,
                    qualified=qualified,
                ),
            )
            if not qualified:
                raise RuntimeError("Numerical qualification failed; no timing claim")
            indices = np.random.default_rng(2504).choice(338, 64, replace=False)
            inputs = [ids(int(i)) for i in indices]
            with (a.output / "timing.jsonl").open("w") as log:

                def sink(row):
                    log.write(json.dumps(row) + "\n")
                    log.flush()

                samples = paired_probe(
                    runners, inputs, passes=7, seed=2504 + a.replicate - 1, sink=sink
                )
            geomeans = {
                m: math.exp(
                    math.fsum(math.log(r["host_ms"]) for r in samples if r["mode"] == m)
                    / 448
                )
                for m in runners
            }
            if a.diagnostics:
                if candidate.config.hidden_size != 128 or a.control:
                    raise ValueError(
                        "Counter harness covers final 14M sparse-path modes only"
                    )
                from support import module
                from sparsity_research.ceilings import architecture_ceiling

                c = candidate.config
                architecture = architecture_ceiling(
                    c.topology_id,
                    layers=c.num_hidden_layers,
                    hidden_size=c.hidden_size,
                    ffn_size=c.intermediate_size,
                    sequence_length=2048,
                    vocabulary_size=50304,
                )
                diag = module("lean_counters", ROOT / "kernels/ablation/diagnostics.py")
                diag.collect(
                    candidate,
                    native,
                    tokens,
                    a.output / "kernel-diagnostics.json",
                    lambda stage, **kw: print(
                        json.dumps(dict(stage=stage, **kw)), flush=True
                    ),
                    architecture,
                )
        status.update(
            status="qualified",
            qualified=True,
            mean_losses=means,
            latency_ms=geomeans,
            timing_indices=indices.tolist(),
            same_checkpoint_speedup=geomeans["native"] / geomeans["candidate"],
            samples_per_implementation=448,
        )
        save("result.json", status)
    except Exception as exc:
        status.update(status="failed", qualified=False, error=str(exc))
        save("result.json", status)
        raise


if __name__ == "__main__":
    main()
