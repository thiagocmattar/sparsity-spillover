"""Paired full-model graphs and eager-anchored numerical qualification."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from run_config import load_config, write_json, RUN_DIR, REPO_ROOT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--condition", required=True)
    p.add_argument("--attempt", required=True)
    p.add_argument("--replicate", type=int, default=1)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--local-calibration", action="store_true")
    args = p.parse_args()
    if not args.attempt.replace("-", "").isalnum() or args.replicate not in (1,2,3):
        raise ValueError("Unique alphanumeric attempt and replicate1..3 required")
    dest = HERE/"artifacts/attempts"/args.attempt
    dest.mkdir(parents=True, exist_ok=False)
    config = load_config()
    cfg = config["latency"]
    result = dict(status="running", arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  config=cfg, scientific_measurement=not args.smoke and not args.local_calibration)
    started = time.monotonic()
    def emit(stage, **fields):
        row = dict(stage=stage, elapsed_seconds=time.monotonic()-started, **fields)
        with (dest/"events.jsonl").open("a") as handle:
            handle.write(json.dumps(row)+"\n")
        print(json.dumps(row), flush=True)
    write_json(dest/"manifest.json", result)
    try:
        import torch
        import numpy as np
        import transformers
        import port
        from source_identity import identity
        from sparsity_research.pythia import load_checkpoint_pythia
        actual_runtime = dict(python=f"{sys.version_info.major}.{sys.version_info.minor}",
            torch=torch.__version__.split("+")[0], transformers=transformers.__version__, cuda_runtime=torch.version.cuda)
        if actual_runtime != config["runtime"]:
            raise RuntimeError(f"Pinned runtime mismatch: {actual_runtime}")
        result["source_identity"] = identity()
        if not args.local_calibration and torch.cuda.get_device_name() != cfg["gpu"]:
            raise RuntimeError("Final measurement requires RTX5090")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG"):
            raise RuntimeError("Canonical cuBLAS workspace required")
        torch.manual_seed(cfg["runtime_seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        result["runtime"] = dict(python=platform.python_version(), torch=torch.__version__,
            transformers=transformers.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
            gpu_uuid=str(getattr(torch.cuda.get_device_properties(0), "uuid", "unavailable")))
        checkpoint_files = []
        for path in sorted(args.checkpoint.iterdir()):
            if path.name == "training_state.pt" or not path.is_file():
                continue
            with path.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            checkpoint_files.append(dict(path=path.name, bytes=path.stat().st_size, sha256=digest))
        result["checkpoint_files"] = checkpoint_files
        cache = REPO_ROOT/"data/tokenized/minipile-pythia-14m-full/validation/tokens.int32.bin"
        with cache.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != "51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451":
            raise RuntimeError("Validation cache identity mismatch")
        data = np.memmap(cache, dtype=np.int32, mode="r")
        assert divmod(len(data),2048) == (338,1444)
        result["cache_sha256"] = digest
        indices = np.random.default_rng(cfg["timing_seed"]).choice(338,cfg["timing_inputs"],replace=False)
        if args.smoke:
            indices = indices[:4]
        inputs = [torch.tensor(data[i*2048:(i+1)*2048].copy(), device="cuda", dtype=torch.long)[None] for i in indices]
        models, runners = {}, {}
        with torch.inference_mode():
            for mode in ("native", "kernel"):
                emit("load", mode=mode)
                model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, args.checkpoint, torch=torch).to("cuda", dtype=torch.bfloat16).eval()
                model.set_attn_implementation("sdpa")
                model.config.use_cache = False
                if mode == "kernel":
                    result["implementation"] = port.install(model)
                else:
                    eager = port.replay.dense.DenseRunner(lambda ids,m=model:m(input_ids=ids,use_cache=False).logits,inputs[0].clone(),"native")
                    eager.prepare()
                    runners["native"] = eager
                models[mode] = model
                emit("capture", mode=mode)
                runner = port.replay.dense.DenseRunner(lambda ids,m=model:port.replay.scaffold.forward(m,ids),inputs[0].clone(),"graph")
                runner.prepare()
                runners[mode+"_graph"] = runner
            result["setup_seconds"] = {k:r.setup_seconds for k,r in runners.items()}
            for runner in runners.values():
                for ids in inputs:
                    runner.stage(ids)
                    runner()
            emit("timing", inputs=len(inputs))
            samples = port.replay.dense.paired_probe({k:r for k,r in runners.items() if k.endswith("_graph")}, inputs,
                passes=2 if args.smoke else cfg["timing_passes"], seed=cfg["timing_seed"]+args.replicate-1)
            summaries = {}
            for mode in ("native_graph", "kernel_graph"):
                values = [r["host_ms"] for r in samples if r["mode"] == mode]
                summaries[mode] = dict(geomean_host_ms=math.exp(math.fsum(map(math.log,values))/len(values)), samples=len(values))
            write_json(dest/"timing.json", dict(indices=indices.tolist(), samples=samples, summary=summaries))
            blocks = 4 if args.smoke else 338
            emit("qualification", blocks=blocks)
            stream = (torch.tensor(data[i*2048:(i+1)*2048].copy(),device="cuda",dtype=torch.long)[None] for i in range(blocks))
            bounds = dict(logit_atol=cfg["logit_atol"], logit_rtol=cfg["logit_rtol"],
                          logit_relative_l2=cfg["relative_l2_max"], validation_loss_atol=cfg["pooled_loss_delta_max"])
            quality = port.replay.dense.compare_inputs(runners, stream, bounds, progress=lambda **f:emit("qualification",**f))
            quality.update(blocks=blocks, documents=500 if blocks==338 else None,
                input_tokens=blocks*2048, excluded_tail_tokens=1444 if blocks==338 else None)
            write_json(dest/"quality.json", quality)
            result.update(status="completed", qualified=all(quality["pass"].values()), qualification=quality["pass"],
                losses=quality["loss"], timing=summaries, peak_reserved_bytes=torch.cuda.max_memory_reserved())
            if args.replicate == 1:
                from runtime_diagnostics import collect
                diagnostics = collect(models["kernel"], data, port.replay.scaffold.forward, emit, blocks=blocks)
                write_json(dest/"runtime-diagnostics.json", diagnostics)
            if not result["qualified"]:
                result["status"] = "unqualified"
    except BaseException as error:
        result.update(status="failed", error=str(error), traceback=traceback.format_exc())
        raise
    finally:
        result["elapsed_seconds"] = time.monotonic()-started
        write_json(dest/"manifest.json", result)


if __name__ == "__main__":
    main()
