"""Development-only K002 full-model probe; dense linears or explicit composition.

Uses the frozen model probe's selected runners and the existing numerical gates.
No checkpoint, gate, RoPE placement or original evaluator is changed.
"""
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import transformers

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from probe_models import (RUN, ROOT, DenseRunner, SelectedRunner, compare_inputs,
    paired_probe, matched_summaries, select_sites, development_condition,
    create_adapter, source_records, config, inside, read_json, record, sha256,
    verify_record, write_json, load_checkpoint_pythia, topology_metadata)

ATTENTION_NAME = "run025_k002_exact_zero_query"


def load_attention():
    spec = importlib.util.spec_from_file_location("run025_probe_attention_k002", HERE / "candidates/k002/attention.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register_attention(interface):
    from transformers.models.gpt_neox.modeling_gpt_neox import ALL_ATTENTION_FUNCTIONS
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS
    ALL_ATTENTION_FUNCTIONS.register(ATTENTION_NAME, interface)
    # A custom attention name also needs the native SDPA mask construction.
    ALL_MASK_ATTENTION_FUNCTIONS.register(ATTENTION_NAME, ALL_MASK_ATTENTION_FUNCTIONS["sdpa"])


class AttentionAdapter:
    def __init__(self, model, linears=None):
        self.model, self.linears = model, linears

    def set_mode(self, mode):
        if mode not in ("native", "candidate"):
            raise ValueError(mode)
        self.model.set_attn_implementation("sdpa" if mode == "native" else ATTENTION_NAME)
        if self.linears is not None:
            self.linears.set_mode(mode)


def branch_snapshot(model):
    """Actual flags from last invocation, never inferred from marginal zeros."""
    rows = []
    for layer, block in enumerate(model.gpt_neox.layers):
        workspace = getattr(block.attention, "_run025_k002_workspace", None)
        if workspace is None:
            raise RuntimeError(f"Layer {layer} did not execute K002; refusing mislabeled dense fallback")
        flags = workspace.fast_tiles.detach().cpu()
        if not bool(((flags == 0) | (flags == 1)).all()):
            raise RuntimeError(f"Layer {layer} has unwritten/invalid K002 branch flags")
        positions = torch.nonzero(flags.reshape(-1) != 0).flatten().tolist()
        rows.append({"layer": layer, "shape_b_h_query_tiles": list(flags.shape),
                     "query_tile_rows": workspace.block_m, "sequence_length": workspace.shape[-2],
                     "fast_tiles": len(positions), "total_tiles": flags.numel(),
                     "fast_tile_flat_indices": positions, "flat_index_order": "B,H,query_tile",
                     "workspace_bytes": workspace.bytes})
    return rows


def invalidate_branch_flags(model):
    """Untimed sentinel prevents stale workspace flags from proving a fallback ran."""
    for block in model.gpt_neox.layers:
        workspace = getattr(block.attention, "_run025_k002_workspace", None)
        if workspace is not None:
            workspace.fast_tiles.fill_(255)


def diagnostic_profile(forward, ids):
    with torch.inference_mode(), torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        forward(ids)
        torch.cuda.synchronize()
    rows = [{"name": event.key, "calls": event.count,
             "self_cpu_us": event.self_cpu_time_total, "self_device_us": event.self_device_time_total}
            for event in profiler.key_averages()]
    return {"scope": "one extra eager full forward; instrumented costs, not primary timings",
            "operators": rows,
            "attention_and_prefix_operators": [row for row in rows if any(term in row["name"].lower()
                 for term in ("attention", "prefix", "zero_query", "flash", "softmax", "bmm"))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--linears", choices=("none", "k001", "k003"), default="none")
    parser.add_argument("--sites", choices=("h", "all", "active"), default="active")
    parser.add_argument("--execution", nargs="+", choices=("eager", "graph"), default=["eager", "graph"])
    parser.add_argument("--sdpa-backend", choices=("auto", "flash"), default="auto")
    parser.add_argument("--inputs", type=int, default=16)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.inputs <= 64 or args.passes <= 0 or args.seconds <= 0 or len(set(args.execution)) != len(args.execution):
        raise ValueError("Invalid probe counts or duplicated execution mode")
    directory = inside(RUN / "artifacts", args.attempt)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def emit(stage, **fields):
        row = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage,
               "elapsed_seconds": time.monotonic() - started, **fields}
        write_json(directory / "status.json", row)
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)
    def check():
        if time.monotonic() - started >= args.seconds:
            raise TimeoutError("Attention probe budget exceeded; external timeout must bound compilation")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; helper tests are not performance evidence")
        cfg = config()
        for name, actual in (("torch", torch.__version__.split("+")[0]), ("numpy", np.__version__),
                             ("transformers", transformers.__version__)):
            if cfg["runtime"][name] != actual:
                raise RuntimeError(f"Pinned {name} mismatch")
        manifest = read_json(RUN / "prelaunch/input_manifest.json")
        if manifest["config_sha256"] != sha256(RUN / "config.json"):
            raise ValueError("Manifest config identity mismatch")
        row = development_condition(manifest, args.condition)
        for item in row["files"] + row["provenance"]:
            verify_record(item)
        length = cfg["inputs"]["sequence_length"]
        development = np.memmap(verify_record(manifest["development"]), mode="r", dtype=np.int32).reshape(-1, length)
        if length != 2048 or len(development) != cfg["inputs"]["development_blocks"]:
            raise ValueError("Full-sequence development input contract mismatch")
        torch.manual_seed(2503)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, ROOT / row["checkpoint"], torch=torch)
        model = model.cuda().bfloat16().eval()
        model.set_attn_implementation("sdpa")
        model.config.use_cache = False
        topology = topology_metadata(model)
        sites = select_sites(args.sites, topology) if args.linears != "none" else frozenset()
        if args.linears != "none" and not sites:
            raise ValueError("Empty linear selection would mislabel composed candidate")
        linear_adapter = create_adapter(model, args.linears, sites) if args.linears != "none" else None
        attention = load_attention()
        register_attention(attention.interface)
        adapter = AttentionAdapter(model, linear_adapter)
        sources = source_records(args.linears if args.linears != "none" else "p0")
        if args.linears == "none":
            sources = [item for item in sources if not item["path"].endswith(("/p0.py", "/kernels/twell_pythia.cu"))]
        sources += [record(Path(__file__)), record(HERE / "candidates/k002/attention.py"),
                    record(HERE / "candidates/k002/kernels.py")]
        write_json(directory / "manifest.json", {"arguments": vars(args), "checkpoint": row,
            "input_manifest": record(RUN / "prelaunch/input_manifest.json"), "development": manifest["development"],
            "validation": manifest["validation"] if args.full_validation else None, "sources": sources,
            "fixed_gates": cfg["calibration"], "topology": topology,
            "candidate_identity": {"attention": "k002", "linears": args.linears, "linear_sites": sorted(sites)},
            "baseline": "native SDPA and native linears", "sdpa_backend": args.sdpa_backend,
            "workload": "B1 T2048 complete logits including unchanged dense LM head",
            "precision": "BF16; TF32 and reduced BF16 reduction disabled",
            "torch": torch.__version__, "transformers": transformers.__version__, "numpy": np.__version__,
            "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
            "timer": "mode selection and identical static input staging excluded for both; prefix and branch detection included",
            "attribution": "attention-only" if args.linears == "none" else "joint linear+attention; not an isolated attention gain"})
        inputs = [torch.tensor(array.copy(), device="cuda", dtype=torch.long)[None] for array in development[:args.inputs]]
        static = inputs[0].clone()
        forward = lambda ids: model(input_ids=ids, use_cache=False, output_attentions=False).logits
        definitions = [("native", "native", "native")]
        if "eager" in args.execution:
            definitions += [("candidate", "candidate", "native")]
        if "graph" in args.execution:
            definitions += [("native_graph", "native", "graph"), ("candidate_graph", "candidate", "graph")]
        backend = contextlib.nullcontext()
        if args.sdpa_backend == "flash":
            from torch.nn.attention import SDPBackend, sdpa_kernel
            backend = sdpa_kernel(SDPBackend.FLASH_ATTENTION)
        runners, setup = {}, {}
        with backend, torch.inference_mode():
            for label, mode, execution in definitions:
                check()
                emit("preparing", mode=label)
                runner = SelectedRunner(DenseRunner(forward, static, execution), adapter, mode)
                runner.prepare()
                for ids in inputs:
                    check()
                    runner.stage(ids)
                    runner()
                torch.cuda.synchronize()
                runners[label] = runner
                setup[label] = {"prepared": True, "execution": execution,
                                "implementation": "native" if mode == "native" else "k002",
                                "setup_seconds": runner.runner.setup_seconds}
                write_json(directory / "setup.json", setup)
            # This also proves an actual sparse kernel path was reached, rather
            # than accepting an unreported dense-only interface fallback.
            adapter.set_mode("candidate")
            invalidate_branch_flags(model)
            forward(inputs[0])
            write_json(directory / "initial-branches.json", branch_snapshot(model))
            quality = compare_inputs(runners, inputs, cfg["calibration"], check=check,
                                     progress=lambda **fields: emit("development_quality", **fields))
            quality.update(blocks=len(inputs), input_tokens=len(inputs) * length,
                           scope="development training inputs; not validation")
            write_json(directory / "development-quality.json", quality)
            if not all(quality["pass"].values()):
                raise RuntimeError("Fixed development logit/loss gate failed; K002 is unqualified")
            def sink(sample):
                check()
                with (directory / "timing-samples.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(sample, allow_nan=False) + "\n")
            timed = {key: value for key, value in runners.items() if key != "native" or "eager" in args.execution}
            samples = paired_probe(timed, inputs, passes=args.passes, seed=cfg["inputs"]["timing_seed"], sink=sink)
            summary = matched_summaries(samples)
            write_json(directory / "timing.json", {"matched": summary, "qualification": "development only",
                                                   "raw_samples": "timing-samples.jsonl"})
            emit("timed", matched=summary)
            branch_totals = {"fast_tiles": 0, "total_tiles": 0, "input_blocks": len(inputs)}
            for index, ids in enumerate(inputs):
                check()
                adapter.set_mode("candidate")
                invalidate_branch_flags(model)
                torch.cuda.synchronize()
                before = time.perf_counter()
                forward(ids)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - before
                before_readback = time.perf_counter()
                branches = branch_snapshot(model)
                readback = time.perf_counter() - before_readback
                for branch in branches:
                    branch_totals["fast_tiles"] += branch["fast_tiles"]
                    branch_totals["total_tiles"] += branch["total_tiles"]
                write_json(directory / "branches" / f"input-{index:03d}.json", {
                    "development_input_index": index, "per_layer": branches,
                    "diagnostic_full_forward_seconds": elapsed, "branch_readback_seconds_excluded": readback,
                    "scope": "extra untimed diagnostic forward; not primary timing samples"})
            write_json(directory / "branch-summary.json", branch_totals)
            for mode in ("native", "candidate"):
                check()
                adapter.set_mode(mode)
                write_json(directory / f"{mode}-operator-profile.json", diagnostic_profile(forward, inputs[0]))
            if args.full_validation:
                tokens = np.memmap(verify_record(manifest["validation"]), mode="r", dtype=np.int32)
                verify_record(manifest["validation_metadata"])
                blocks, tail = divmod(len(tokens), length)
                if blocks != 338 or tail != 1444:
                    raise ValueError("Complete 338-block/1444-tail validation coverage mismatch")
                val_inputs = (torch.tensor(np.array(tokens[i * length:(i + 1) * length]), device="cuda", dtype=torch.long)[None]
                              for i in range(blocks))
                validation = compare_inputs(runners, val_inputs, cfg["calibration"], check=check,
                                             progress=lambda **fields: emit("validation", **fields))
                validation.update(blocks=blocks, documents=500, input_tokens=blocks * length,
                                  excluded_tail_tokens=tail, complete=True, validation_cache=manifest["validation"])
                write_json(directory / "full-validation.json", validation)
                if not all(validation["pass"].values()):
                    raise RuntimeError("Fixed full-validation logit/loss gate failed; candidate unqualified")
            for item in sources:
                verify_record(item)
            emit("complete", full_validation=args.full_validation, actual_branch_totals=branch_totals)
    except BaseException as error:
        emit("failed_or_incomplete", error=str(error))
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
