"""Frozen all-size final evaluator with complete validation and matched timing."""

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


LINEAR_SITES = frozenset(("a", "m", "h", "z"))
SPECS = {
    "k009": {
        "size": "70m",
        "freeze": "candidates/k009/FROZEN.json",
        "candidate": "candidates/k009/candidate.py",
        "parents": ("candidates/k001/candidate.py", "candidates/k001/kernel.cu"),
        "site_policy": "topology",
    },
    "k010": {
        "size": "410m",
        "freeze": "candidates/k010/FROZEN.json",
        "candidate": "candidates/k010/candidate.py",
        "parents": ("candidates/k004/candidate.py", "candidates/k004/kernels.py"),
        "site_policy": "z_only",
    },
    "k012": {
        "size": "14m",
        "freeze": "candidates/k012/FROZEN.json",
        "candidate": "candidates/k012/candidate.py",
        "parents": ("candidates/k001/candidate.py", "candidates/k001/kernel.cu"),
        "site_policy": "topology",
    },
}
IMPLEMENTATIONS = tuple(SPECS)


def select_sites(selection, topology, implementation):
    policy = SPECS[implementation]["site_policy"]
    if policy == "z_only":
        if selection != "active":
            raise ValueError("K010 final policy requires --sites active and executes fixed z")
        return frozenset(("z",))
    if selection == "h":
        return frozenset(("h",))
    if selection == "all":
        return LINEAR_SITES
    if selection == "active":
        return LINEAR_SITES.intersection(topology["active_sites"])
    raise ValueError(f"Unknown site selection: {selection}")


def condition_row(manifest, identifier, implementation, freeze):
    row = next((item for item in manifest["checkpoints"] if item["id"] == identifier), None)
    if row is None or row.get("size") != SPECS[implementation]["size"]:
        raise ValueError("Condition does not match the frozen architecture policy")
    development = {
        item["condition"] for item in freeze["selection_evidence"]["development_evaluation"]
    }
    heldout = set(freeze["heldout_boundary"]["conditions"])
    if row.get("partition") == "development" and identifier in development:
        return row
    if row.get("partition") == "untuned" and identifier in heldout:
        return row
    raise ValueError("Condition is outside the frozen development/held-out boundary")


def verify_freeze(manifest, identifier, implementation):
    spec = SPECS[implementation]
    path = HERE / spec["freeze"]
    freeze = read_json(path)
    if freeze.get("candidate_id") != implementation:
        raise ValueError("Unexpected frozen candidate identity")
    row = condition_row(manifest, identifier, implementation, freeze)
    for item in freeze["sources"]:
        verify_record(item, root=RUN)
    return row, freeze, path


class SiteAdapter:
    def __init__(self, adapter, implementation, sites):
        if implementation not in IMPLEMENTATIONS or not set(sites) <= LINEAR_SITES:
            raise ValueError("Unsupported implementation or sites")
        self.adapter = adapter
        self.implementation = implementation
        self.sites = frozenset(sites)

    def set_mode(self, mode):
        if mode not in ("native", "candidate"):
            raise ValueError(mode)
        self.adapter.set_mode("native" if mode == "native" else self.implementation)
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


def matched_summaries(samples):
    result = {}
    for label, baseline, candidate in (
        ("eager", "native", "candidate"),
        ("cuda_graph", "native_graph", "candidate_graph"),
    ):
        selected = [row for row in samples if row["mode"] in (baseline, candidate)]
        if selected:
            if {row["mode"] for row in selected} != {baseline, candidate}:
                raise ValueError("A matched speedup requires dense and candidate modes")
            result[label] = {
                "reference": baseline,
                "summary": timing_summary(selected, reference=baseline),
            }
    return result


def create_adapter(model, implementation, sites):
    source = HERE / SPECS[implementation]["candidate"]
    spec = importlib.util.spec_from_file_location(f"run025_final_{implementation}", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return SiteAdapter(module.Adapter(model), implementation, sites)


def source_records(implementation):
    spec = SPECS[implementation]
    sources = [
        Path(__file__),
        HERE / "dense_probe/probe.py",
        RUN / "measurement.py",
        RUN / "run025_common.py",
        RUN / "config.json",
        HERE / spec["candidate"],
        HERE / spec["freeze"],
    ]
    sources += [HERE / path for path in spec["parents"]]
    sources += sorted((ROOT / "src/sparsity_research").glob("*.py"))
    return [record(source) for source in sources]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--implementation", choices=IMPLEMENTATIONS, required=True)
    parser.add_argument("--sites", choices=("h", "all", "active"), default="active")
    parser.add_argument(
        "--execution", nargs="+", choices=("eager", "graph"), default=["eager"]
    )
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--inputs", type=int, default=16)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--seconds", type=int, default=1800)
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    if not args.full_validation:
        raise ValueError("Frozen final evaluation requires --full-validation")
    if (
        not 1 <= args.inputs <= 64
        or args.passes < 1
        or args.seconds < 1
        or len(set(args.execution)) != len(args.execution)
    ):
        raise ValueError("Invalid count, duration, or duplicate execution mode")

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
            raise TimeoutError("Final probe deadline; external timeout must bound compilation")

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
        row, freeze, freeze_path = verify_freeze(
            manifest, args.condition, args.implementation
        )
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
        sites = select_sites(args.sites, topology, args.implementation)
        if not sites:
            raise ValueError("No selected sparse sites")
        adapter = create_adapter(model, args.implementation, sites)
        sources = source_records(args.implementation)
        candidate_coverage = adapter.adapter.coverage()

        identity = {
            "arguments": vars(args),
            "checkpoint": row,
            "partition": row["partition"],
            "development": manifest["development"],
            "validation": manifest["validation"],
            "validation_metadata": manifest["validation_metadata"],
            "input_manifest": record(RUN / "prelaunch/input_manifest.json"),
            "frozen_candidate": record(freeze_path),
            "frozen_policy": freeze["policy"],
            "candidate_coverage": candidate_coverage,
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
            "scope": "frozen final checkpoint; complete 338-block validation required",
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
        definitions = [("native", "native", "native")]
        if "eager" in args.execution:
            definitions.append(("candidate", "candidate", "native"))
        if "graph" in args.execution:
            definitions += [
                ("native_graph", "native", "graph"),
                ("candidate_graph", "candidate", "graph"),
            ]

        with torch.inference_mode():
            for label, mode, execution in definitions:
                check()
                emit("preparing", mode=label)
                runner = SelectedRunner(
                    DenseRunner(forward, static_input, execution), adapter, mode
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
                    "selected_implementation": (
                        "native" if mode == "native" else args.implementation
                    ),
                    "execution": execution,
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

            timed = {
                key: value
                for key, value in runners.items()
                if key != "native" or "eager" in args.execution
            }
            samples = paired_probe(
                timed,
                inputs,
                passes=args.passes,
                seed=cfg["inputs"]["timing_seed"],
                sink=sink,
            )
            summary = matched_summaries(samples)
            write_json(
                directory / "timing.json",
                {
                    "matched": summary,
                    "qualification": "complete validation is required in this artifact",
                    "raw_samples": "timing-samples.jsonl",
                },
            )
            emit("timed", matched=summary)

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
            implementation=args.implementation,
            partition=row["partition"],
            selected_sites=sorted(sites),
            qualified_scope="frozen checkpoint with complete validation",
        )
    except BaseException as error:
        emit("failed_or_incomplete", error=str(error))
        (directory / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
