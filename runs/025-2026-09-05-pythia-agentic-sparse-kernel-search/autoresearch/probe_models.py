"""Development-only full-logit P0/K001/K003 probe with matched eager/graph baselines."""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch
import transformers

HERE = Path(__file__).resolve().parent
RUN = HERE.parent
ROOT = RUN.parents[1]
sys.path[:0] = [str(HERE), str(RUN), str(ROOT / "src")]
from dense_probe.probe import DenseRunner, paired_probe, compare_inputs
from measurement import timing_summary
from run025_common import config, inside, read_json, record, sha256, verify_record, write_json
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata

LINEAR_SITES = frozenset(("a", "m", "h", "z"))
IMPLEMENTATIONS = ("p0", "k001", "k003")


def select_sites(selection, topology):
    if selection == "h":
        return frozenset(("h",))
    if selection == "all":
        return LINEAR_SITES
    if selection == "active":
        return LINEAR_SITES.intersection(topology["active_sites"])
    raise ValueError(f"Unknown site selection: {selection}")


def development_condition(manifest, identifier):
    row = next((row for row in manifest["checkpoints"] if row["id"] == identifier), None)
    if row is None or row.get("partition") != "development":
        raise ValueError("Model probe requires a declared development endpoint; interior checkpoints are held out")
    return row


class SiteAdapter:
    """Run-local dispatch wrapper; immutable P0 and K001 source remain untouched."""
    def __init__(self, adapter, implementation, sites):
        if implementation not in IMPLEMENTATIONS or not set(sites) <= LINEAR_SITES:
            raise ValueError("Unsupported implementation or sites")
        self.adapter, self.implementation, self.sites = adapter, implementation, frozenset(sites)

    def set_mode(self, mode):
        if mode not in ("native", "candidate"):
            raise ValueError(mode)
        self.adapter.set_mode("native" if mode == "native" else self.implementation)
        for parent, name, original, _, site in self.adapter.entries:
            if site not in self.sites:
                setattr(parent, name, original)


class SelectedRunner:
    """Set Python/module behavior before capture and before each input staging.

    Mode switching is excluded symmetrically from native/candidate timings. A
    graph replay executes its captured kernels, never a freshly relabeled graph.
    """
    def __init__(self, runner, adapter, mode):
        self.runner, self.adapter, self.mode = runner, adapter, mode

    def prepare(self):
        self.adapter.set_mode(self.mode)
        self.runner.prepare()

    def stage(self, ids):
        self.adapter.set_mode(self.mode)
        self.runner.stage(ids)

    def __call__(self):
        return self.runner()


def matched_summaries(samples):
    result = {}
    for label, baseline, candidate in (("eager", "native", "candidate"),
                                        ("cuda_graph", "native_graph", "candidate_graph")):
        selected = [row for row in samples if row["mode"] in (baseline, candidate)]
        if selected:
            if {row["mode"] for row in selected} != {baseline, candidate}:
                raise ValueError("A matched speedup requires both dense and candidate modes")
            result[label] = {"reference": baseline, "summary": timing_summary(selected, reference=baseline)}
    return result


def create_adapter(model, implementation, sites):
    if implementation not in IMPLEMENTATIONS:
        raise ValueError("Unsupported candidate identity")
    if implementation == "p0":
        from p0 import Adapter
    else:
        source = HERE / "candidates" / implementation / "candidate.py"
        spec = importlib.util.spec_from_file_location(f"run025_probe_{implementation}", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        Adapter = module.Adapter
    return SiteAdapter(Adapter(model), implementation, sites)


def source_records(implementation):
    sources = [Path(__file__), HERE / "dense_probe/probe.py", RUN / "measurement.py",
               RUN / "run025_common.py", RUN / "config.json"]
    sources += sorted((ROOT / "src/sparsity_research").glob("*.py"))
    sources += ([RUN / "p0.py", RUN / "kernels/twell_pythia.cu"] if implementation == "p0" else
                [HERE / "candidates" / implementation / name for name in ("candidate.py", "kernel.cu")])
    return [record(source) for source in sources]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, required=True)
    parser.add_argument("--sites", choices=("h", "all", "active"), default="active")
    parser.add_argument("--execution", nargs="+", choices=("eager", "graph"), default=["eager", "graph"])
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--inputs", type=int, default=16)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.inputs <= 64 or args.passes < 1 or args.seconds < 1 or len(set(args.execution)) != len(args.execution):
        raise ValueError("Invalid count, duration or duplicate execution mode")
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
            raise TimeoutError("Model probe deadline; external process timeout must bound compilation")
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU helper tests are not performance evidence")
        cfg = config()
        for name, actual in (("torch", torch.__version__.split("+")[0]), ("numpy", np.__version__),
                             ("transformers", transformers.__version__)):
            if actual != cfg["runtime"][name]:
                raise RuntimeError(f"Pinned {name} mismatch")
        manifest = read_json(RUN / "prelaunch/input_manifest.json")
        if manifest["config_sha256"] != sha256(RUN / "config.json"):
            raise ValueError("Input manifest config identity mismatch")
        row = development_condition(manifest, args.condition)
        for item in row["files"] + row["provenance"]:
            verify_record(item)
        length = cfg["inputs"]["sequence_length"]
        development = np.memmap(verify_record(manifest["development"]), dtype=np.int32, mode="r").reshape(-1, length)
        if length != 2048 or len(development) != cfg["inputs"]["development_blocks"]:
            raise ValueError("Unexpected full-sequence development coverage")
        torch.manual_seed(2503)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, ROOT / row["checkpoint"], torch=torch)
        model = model.cuda().bfloat16().eval()
        model.set_attn_implementation("sdpa")
        model.config.use_cache = False
        topology = topology_metadata(model)
        sites = select_sites(args.sites, topology)
        if not sites:
            raise ValueError("No selected sparse sites; refuse to label native-only execution as candidate")
        adapter = create_adapter(model, args.implementation, sites)
        sources = source_records(args.implementation)
        identity = {"arguments": vars(args), "checkpoint": row, "development": manifest["development"],
                    "validation": manifest["validation"] if args.full_validation else None,
                    "validation_metadata": manifest["validation_metadata"] if args.full_validation else None,
                    "input_manifest": record(RUN / "prelaunch/input_manifest.json"),
                    "sources": sources, "topology": topology, "selected_linear_sites": sorted(sites),
                    "sparse_linear_count": sum(site in sites for *_, site in adapter.adapter.entries),
                    "attention": "unchanged dense SDPA", "workload": "B1 T2048 complete logits including dense LM head",
                    "torch": torch.__version__, "numpy": np.__version__, "transformers": transformers.__version__,
                    "python": platform.python_version(), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(),
                    "precision": "BF16 parameters/activations; TF32 and reduced BF16 reduction disabled",
                    "fixed_calibration_gates": cfg["calibration"], "mode_switch_and_staging": "outside every timer",
                    "scope": "development endpoints only; selected train inputs unless full validation requested"}
        write_json(directory / "manifest.json", identity)
        inputs = [torch.tensor(array.copy(), device="cuda", dtype=torch.long)[None] for array in development[:args.inputs]]
        static_input = inputs[0].clone()
        forward = lambda ids: model(input_ids=ids, use_cache=False, output_attentions=False).logits
        runners, setup = {}, {}
        # Native eager remains the numerical reference even in graph-only timing.
        definitions = [("native", "native", "native")]
        if "eager" in args.execution:
            definitions += [("candidate", "candidate", "native")]
        if "graph" in args.execution:
            definitions += [("native_graph", "native", "graph"), ("candidate_graph", "candidate", "graph")]
        with torch.inference_mode():
            for label, mode, execution in definitions:
                check()
                emit("preparing", mode=label)
                runner = SelectedRunner(DenseRunner(forward, static_input, execution), adapter, mode)
                runner.prepare()
                for ids in inputs:
                    check()
                    runner.stage(ids)
                    runner()
                torch.cuda.synchronize()
                runners[label] = runner
                setup[label] = {"prepared": True, "selected_implementation": "native" if mode == "native" else args.implementation,
                                "execution": execution, "setup_seconds": runner.runner.setup_seconds}
                write_json(directory / "setup.json", setup)
            quality = compare_inputs(runners, inputs, cfg["calibration"], check=check,
                                     progress=lambda **fields: emit("development_quality", **fields))
            quality.update(blocks=len(inputs), input_tokens=len(inputs) * length,
                           coverage="first declared development training blocks; not validation")
            write_json(directory / "development-quality.json", quality)
            if not all(quality["pass"].values()):
                raise RuntimeError("Fixed development logit/loss gate failed; candidate is unqualified")
            def sink(sample):
                check()
                with (directory / "timing-samples.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(sample, allow_nan=False) + "\n")
            timed = {key: value for key, value in runners.items() if key != "native" or "eager" in args.execution}
            samples = paired_probe(timed, inputs, passes=args.passes, seed=cfg["inputs"]["timing_seed"], sink=sink)
            summary = matched_summaries(samples)
            write_json(directory / "timing.json", {"matched": summary, "qualification": "development gates only",
                                                   "raw_samples": "timing-samples.jsonl"})
            emit("timed", matched=summary)
            if args.full_validation:
                tokens = np.memmap(verify_record(manifest["validation"]), dtype=np.int32, mode="r")
                verify_record(manifest["validation_metadata"])
                blocks, tail = divmod(len(tokens), length)
                if blocks != 338 or tail != 1444:
                    raise ValueError("Full 338-block/1444-tail validation contract mismatch")
                val_inputs = (torch.tensor(np.array(tokens[i * length:(i + 1) * length]), device="cuda", dtype=torch.long)[None]
                              for i in range(blocks))
                validation = compare_inputs(runners, val_inputs, cfg["calibration"], check=check,
                                             progress=lambda **fields: emit("validation", **fields))
                validation.update(blocks=blocks, documents=500, input_tokens=blocks * length,
                                  excluded_tail_tokens=tail, validation_cache=manifest["validation"], complete=True)
                write_json(directory / "full-validation.json", validation)
                if not all(validation["pass"].values()):
                    raise RuntimeError("Fixed complete-validation logit/loss gate failed; candidate is unqualified")
            for item in sources:
                verify_record(item)
            emit("complete", full_validation=args.full_validation, implementation=args.implementation,
                 selected_sites=sorted(sites), qualified_scope="full validation" if args.full_validation else "development only")
    except BaseException as error:
        emit("failed_or_incomplete", error=str(error))
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
