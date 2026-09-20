"""Three-way K001/K006/native full-model development and validation probe."""

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
import probe_models as base
from measurement import timing_summary
from run025_common import (
    config,
    inside,
    read_json,
    record,
    sha256,
    verify_record,
    write_json,
)
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata


def load_candidate(identifier):
    source = HERE / "candidates" / identifier / "candidate.py"
    spec = importlib.util.spec_from_file_location(
        f"run025_portfolio_{identifier}", source
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


K001 = load_candidate("k001")
K006 = load_candidate("k006")


class PortfolioAdapter:
    """Mutually exclusive site-filtered installation of two candidate adapters."""

    def __init__(self, model, sites):
        self.sites = frozenset(sites)
        self.adapters = {"k001": K001.Adapter(model), "k006": K006.Adapter(model)}

    def set_mode(self, mode):
        if mode not in {"native", "k001", "k006"}:
            raise ValueError(mode)
        for adapter in self.adapters.values():
            adapter.set_mode("native")
        if mode == "native":
            return
        adapter = self.adapters[mode]
        adapter.set_mode(mode)
        for parent, name, original, _wrapped, site in adapter.entries:
            if site not in self.sites:
                setattr(parent, name, original)

    def coverage(self):
        return {
            "selected_sites": sorted(self.sites),
            "k001": self.adapters["k001"].coverage(),
            "k006": self.adapters["k006"].coverage(),
            "mode_exclusivity": "restore both adapters before every selected installation",
        }


def source_records():
    sources = [
        Path(__file__),
        HERE / "probe_models.py",
        HERE / "dense_probe/probe.py",
        RUN / "measurement.py",
        RUN / "run025_common.py",
        RUN / "config.json",
        HERE / "candidates/k001/candidate.py",
        HERE / "candidates/k001/kernel.cu",
        HERE / "candidates/k005/candidate.py",
        HERE / "candidates/k005/kernel.cu",
        HERE / "candidates/k006/candidate.py",
    ]
    sources += sorted((ROOT / "src/sparsity_research").glob("*.py"))
    return [record(path) for path in sources]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--sites", choices=("h", "all", "active"), default="active")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--inputs", type=int, default=16)
    parser.add_argument("--passes", type=int, default=5)
    parser.add_argument("--seconds", type=int, default=600)
    parser.add_argument("--full-validation", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.inputs <= 64 or args.passes < 1 or args.seconds < 1:
        raise ValueError("Invalid development counts or deadline")
    destination = inside(RUN / "artifacts", args.attempt)
    destination.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def emit(stage, **fields):
        row = {
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "stage": stage,
            "elapsed_seconds": time.monotonic() - started,
            **fields,
        }
        write_json(destination / "status.json", row)
        with (destination / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    def check():
        if time.monotonic() - started >= args.seconds:
            raise TimeoutError("Portfolio model probe deadline")

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required; CPU checks are not performance evidence")
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
            raise ValueError("Input manifest/config identity mismatch")
        row = base.development_condition(manifest, args.condition)
        for item in row["files"] + row["provenance"]:
            verify_record(item)
        length = cfg["inputs"]["sequence_length"]
        development = np.memmap(
            verify_record(manifest["development"]), dtype=np.int32, mode="r"
        ).reshape(-1, length)
        if length != 2048 or len(development) != cfg["inputs"]["development_blocks"]:
            raise ValueError("Unexpected development block coverage")

        torch.manual_seed(2503)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        model = load_checkpoint_pythia(
            transformers.AutoModelForCausalLM,
            ROOT / row["checkpoint"],
            torch=torch,
        )
        model = model.cuda().bfloat16().eval()
        model.set_attn_implementation("sdpa")
        model.config.use_cache = False
        topology = topology_metadata(model)
        sites = base.select_sites(args.sites, topology)
        if not sites:
            raise ValueError("No selected linear sites")
        adapter = PortfolioAdapter(model, sites)
        sources = source_records()
        write_json(
            destination / "manifest.json",
            {
                "arguments": vars(args),
                "checkpoint": row,
                "canonical_R_model": row.get("canonical_logical_products", {}).get(
                    "measured", {}
                ).get("R_model"),
                "development": manifest["development"],
                "validation": manifest["validation"] if args.full_validation else None,
                "input_manifest": record(RUN / "prelaunch/input_manifest.json"),
                "sources": sources,
                "topology": topology,
                "portfolio": adapter.coverage(),
                "workload": "BF16 B1 T2048 complete logits including dense LM head",
                "attention": "unchanged dense SDPA",
                "precision": "BF16; TF32 and reduced BF16 reduction disabled",
                "fixed_gates": cfg["calibration"],
                "timer": "resident input; equal staging/mode selection excluded",
                "scope": "development endpoints unless full validation requested",
                "python": platform.python_version(),
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "numpy": np.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
            },
        )
        inputs = [
            torch.tensor(array.copy(), device="cuda", dtype=torch.long)[None]
            for array in development[: args.inputs]
        ]
        static_input = inputs[0].clone()
        forward = lambda ids: model(
            input_ids=ids, use_cache=False, output_attentions=False
        ).logits
        runners = {}
        with torch.inference_mode():
            for mode in ("native", "k001", "k006"):
                check()
                emit("preparing", mode=mode)
                runner = base.SelectedRunner(
                    base.DenseRunner(forward, static_input, "native"), adapter, mode
                )
                runner.prepare()
                for ids in inputs:
                    check()
                    runner.stage(ids)
                    runner()
                torch.cuda.synchronize()
                runners[mode] = runner
            quality = base.compare_inputs(
                runners,
                inputs,
                cfg["calibration"],
                check=check,
                progress=lambda **fields: emit("development_quality", **fields),
            )
            quality.update(
                blocks=len(inputs),
                input_tokens=len(inputs) * length,
                coverage="first fixed development training blocks",
            )
            write_json(destination / "development-quality.json", quality)
            if not all(quality["pass"].values()):
                raise RuntimeError("Fixed development logit/loss gate failed")

            def sink(sample):
                check()
                with (destination / "timing-samples.jsonl").open(
                    "a", encoding="utf-8"
                ) as stream:
                    stream.write(json.dumps(sample, allow_nan=False) + "\n")

            samples = base.paired_probe(
                runners,
                inputs,
                passes=args.passes,
                seed=cfg["inputs"]["timing_seed"],
                sink=sink,
            )
            timing = timing_summary(samples, reference="native")
            write_json(
                destination / "timing.json",
                {
                    "summary": timing,
                    "optimization_ratio_k006_over_k001": (
                        timing["k006"]["paired_geomean_speedup"]
                        / timing["k001"]["paired_geomean_speedup"]
                    ),
                    "raw_samples": "timing-samples.jsonl",
                    "qualification": "development fixed gates only",
                },
            )
            emit("timed", summary=timing)
            if args.full_validation:
                tokens = np.memmap(
                    verify_record(manifest["validation"]), dtype=np.int32, mode="r"
                )
                verify_record(manifest["validation_metadata"])
                blocks, tail = divmod(len(tokens), length)
                if blocks != 338 or tail != 1444:
                    raise ValueError("Complete validation coverage mismatch")
                validation_inputs = (
                    torch.tensor(
                        np.array(tokens[index * length : (index + 1) * length]),
                        device="cuda",
                        dtype=torch.long,
                    )[None]
                    for index in range(blocks)
                )
                validation = base.compare_inputs(
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
                write_json(destination / "full-validation.json", validation)
                if not all(validation["pass"].values()):
                    raise RuntimeError("Complete-validation fixed gate failed")
        for item in sources:
            verify_record(item)
        emit(
            "complete",
            condition=args.condition,
            selected_sites=sorted(sites),
            full_validation=args.full_validation,
        )
    except BaseException as error:
        emit("failed_or_incomplete", error=str(error))
        (destination / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        raise


if __name__ == "__main__":
    main()
