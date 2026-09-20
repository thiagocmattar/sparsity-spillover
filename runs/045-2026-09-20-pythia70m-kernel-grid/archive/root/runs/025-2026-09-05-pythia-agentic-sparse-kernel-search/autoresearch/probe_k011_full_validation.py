"""Development-only K011 mask probe with complete 338-block validation."""

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

from dense_probe.probe import DenseRunner, compare_inputs, paired_probe
from measurement import timing_summary
from run025_common import config, inside, read_json, record, sha256, verify_record, write_json
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata


IMPLEMENTATION = "k011"
LINEAR_SITES = frozenset(("a", "m", "h", "z"))
SEARCH_PLAN = HERE / "candidates/k011/FULL_VALIDATION_SEARCH.json"


def development_condition(manifest, identifier):
    row = next((item for item in manifest["checkpoints"] if item["id"] == identifier), None)
    if row is None or row.get("size") != "14m" or row.get("partition") != "development":
        raise ValueError("K011 complete-validation search accepts only 14M development conditions")
    allowed = set(read_json(SEARCH_PLAN)["partition_boundary"]["development_conditions"])
    if identifier not in allowed:
        raise ValueError("Condition is outside the predeclared K011 development search")
    return row


def select_sites(selection, topology):
    if selection == "all":
        return LINEAR_SITES
    if selection == "h":
        return frozenset(("h",))
    if selection == "active":
        return LINEAR_SITES.intersection(topology["active_sites"])
    raise ValueError(f"Unknown site selection: {selection}")


def candidate_module():
    source = HERE / "candidates/k011/candidate.py"
    spec = importlib.util.spec_from_file_location("run025_k011_complete_validation", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class SiteAdapter:
    def __init__(self, adapter, sites):
        if not set(sites) <= LINEAR_SITES:
            raise ValueError("Unsupported K011 linear sites")
        self.adapter = adapter
        self.sites = frozenset(sites)

    def set_mode(self, mode):
        if mode not in ("native", "candidate"):
            raise ValueError(mode)
        self.adapter.set_mode("native" if mode == "native" else IMPLEMENTATION)
        for parent, name, original, _, site in self.adapter.entries:
            if site not in self.sites:
                setattr(parent, name, original)


class SelectedRunner:
    def __init__(self, runner, adapter, mode):
        self.runner = runner
        self.adapter = adapter
        self.mode = mode

    def prepare(self):
        self.adapter.set_mode(self.mode)
        self.runner.prepare()

    def stage(self, ids):
        self.adapter.set_mode(self.mode)
        self.runner.stage(ids)

    def __call__(self):
        return self.runner()


def source_records():
    sources = [
        Path(__file__),
        SEARCH_PLAN,
        HERE / "dense_probe/probe.py",
        RUN / "measurement.py",
        RUN / "run025_common.py",
        RUN / "config.json",
        HERE / "candidates/k011/candidate.py",
        HERE / "candidates/k001/candidate.py",
        HERE / "candidates/k001/kernel.cu",
    ]
    sources += sorted((ROOT / "src/sparsity_research").glob("*.py"))
    return [record(source) for source in sources]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--sites", choices=("h", "all", "active"), default="active")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--inputs", type=int, default=16)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    if not args.full_validation:
        raise ValueError("K011 mask search requires --full-validation")
    if not 1 <= args.inputs <= 64 or args.passes < 1 or args.seconds < 1:
        raise ValueError("Invalid timing or duration parameters")

    plan = read_json(SEARCH_PLAN)
    if args.mask not in plan["shortlist"]:
        raise ValueError("Mask is outside the predeclared complete-validation shortlist")
    directory = inside(RUN / "artifacts", args.attempt)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def emit(stage, **fields):
        row = {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            "elapsed_seconds": time.monotonic() - started,
            **fields,
        }
        write_json(directory / "status.json", row)
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row), flush=True)

    def check():
        if time.monotonic() - started >= args.seconds:
            raise TimeoutError("K011 complete-validation probe deadline")

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU tests are not performance evidence")
        cfg = config()
        for name, actual in (
            ("torch", torch.__version__.split("+")[0]),
            ("numpy", np.__version__),
            ("transformers", transformers.__version__),
        ):
            if actual != cfg["runtime"][name]:
                raise RuntimeError(f"Pinned {name} mismatch")
        manifest = read_json(RUN / "prelaunch/input_manifest.json")
        if manifest["config_sha256"] != sha256(RUN / "config.json"):
            raise ValueError("Input manifest config identity mismatch")
        row = development_condition(manifest, args.condition)
        for item in row["files"] + row["provenance"]:
            verify_record(item)

        length = cfg["inputs"]["sequence_length"]
        development = np.memmap(
            verify_record(manifest["development"]), dtype=np.int32, mode="r"
        ).reshape(-1, length)
        if length != 2048 or len(development) != cfg["inputs"]["development_blocks"]:
            raise ValueError("Unexpected full-sequence development coverage")

        torch.manual_seed(2503)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        model = load_checkpoint_pythia(
            transformers.AutoModelForCausalLM, ROOT / row["checkpoint"], torch=torch
        )
        model = model.cuda().bfloat16().eval()
        model.set_attn_implementation("sdpa")
        model.config.use_cache = False
        topology = topology_metadata(model)
        sites = select_sites(args.sites, topology)
        if not sites:
            raise ValueError("No selected sparse sites")
        module = candidate_module()
        adapter = SiteAdapter(module.Adapter(model, mask_id=args.mask), sites)
        sources = source_records()

        identity = {
            "arguments": vars(args),
            "checkpoint": row,
            "partition": "development",
            "development": manifest["development"],
            "validation": manifest["validation"],
            "validation_metadata": manifest["validation_metadata"],
            "input_manifest": record(RUN / "prelaunch/input_manifest.json"),
            "search_plan": record(SEARCH_PLAN),
            "candidate_coverage": adapter.adapter.coverage(),
            "sources": sources,
            "topology": topology,
            "selected_linear_sites": sorted(sites),
            "attention": "dense SDPA except selected attention projection linears",
            "workload": "B1 T2048 complete logits including dense LM head",
            "torch": torch.__version__,
            "numpy": np.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "precision": "BF16 parameters/activations; TF32 and reduced BF16 reduction disabled",
            "fixed_calibration_gates": cfg["calibration"],
            "mode_switch_and_staging": "outside every timer",
            "scope": "development-only mask search; complete 338-block validation; held-out checkpoint conditions forbidden",
        }
        write_json(directory / "manifest.json", identity)

        inputs = [
            torch.tensor(array.copy(), device="cuda", dtype=torch.long)[None]
            for array in development[: args.inputs]
        ]
        static_input = inputs[0].clone()
        forward = lambda ids: model(
            input_ids=ids, use_cache=False, output_attentions=False
        ).logits
        runners = {}
        setup = {}
        with torch.inference_mode():
            for label, mode in (("native", "native"), ("candidate", "candidate")):
                check()
                emit("preparing", mode=label)
                runner = SelectedRunner(
                    DenseRunner(forward, static_input, "native"), adapter, mode
                )
                runner.prepare()
                for ids in inputs:
                    check()
                    runner.stage(ids)
                    runner()
                torch.cuda.synchronize()
                runners[label] = runner
                setup[label] = {
                    "prepared": True,
                    "selected_implementation": "native" if mode == "native" else IMPLEMENTATION,
                    "execution": "eager",
                    "setup_seconds": runner.runner.setup_seconds,
                }
                write_json(directory / "setup.json", setup)

            quality = compare_inputs(
                runners,
                inputs,
                cfg["calibration"],
                check=check,
                progress=lambda **fields: emit("development_quality", **fields),
            )
            quality.update(
                blocks=len(inputs),
                input_tokens=len(inputs) * length,
                coverage="fixed train-split pre-timing calibration blocks",
            )
            write_json(directory / "development-quality.json", quality)
            if not all(quality["pass"].values()):
                raise RuntimeError("Fixed pre-timing calibration gate failed")

            def sink(sample):
                check()
                with (directory / "timing-samples.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(sample, allow_nan=False) + "\n")

            samples = paired_probe(
                runners,
                inputs,
                passes=args.passes,
                seed=cfg["inputs"]["timing_seed"],
                sink=sink,
            )
            timing = {
                "matched": {
                    "eager": {
                        "reference": "native",
                        "summary": timing_summary(samples, reference="native"),
                    }
                },
                "qualification": "complete validation required in this artifact",
                "raw_samples": "timing-samples.jsonl",
            }
            write_json(directory / "timing.json", timing)
            emit("timed", matched=timing["matched"])

            tokens = np.memmap(
                verify_record(manifest["validation"]), dtype=np.int32, mode="r"
            )
            verify_record(manifest["validation_metadata"])
            blocks, tail = divmod(len(tokens), length)
            if blocks != 338 or tail != 1444:
                raise ValueError("Full 338-block/1444-tail validation contract mismatch")
            validation_inputs = (
                torch.tensor(
                    np.array(tokens[index * length : (index + 1) * length]),
                    device="cuda",
                    dtype=torch.long,
                )[None]
                for index in range(blocks)
            )
            validation = compare_inputs(
                runners,
                validation_inputs,
                cfg["calibration"],
                check=check,
                progress=lambda **fields: emit("validation", **fields),
            )
            validation.update(
                blocks=blocks,
                documents=500,
                input_tokens=blocks * length,
                excluded_tail_tokens=tail,
                validation_cache=manifest["validation"],
                complete=True,
            )
            write_json(directory / "full-validation.json", validation)
            if not all(validation["pass"].values()):
                raise RuntimeError("Fixed complete-validation gate failed")

        for item in sources:
            verify_record(item)
        emit(
            "complete",
            implementation=IMPLEMENTATION,
            mask=args.mask,
            selected_sites=sorted(sites),
            qualified_scope="development checkpoint with complete validation",
        )
    except BaseException as error:
        emit("failed_or_incomplete", error=str(error))
        (directory / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
