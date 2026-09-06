"""Native BF16 occupancy on the first four declared development train blocks.

Diagnostic only: no candidate kernel, optimizer, attention rewrite or tuning.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time
import traceback

import numpy as np
import torch
import transformers
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
RUN, ROOT = HERE.parent, HERE.parent.parents[1]
sys.path[:0] = [str(RUN), str(ROOT / "src")]
from run025_common import config, inside, read_json, record, sha256, verify_record, write_json
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata

BLOCKS = 4
TILE_SHAPES = ((16, 32), (32, 32), (16, 64))


def development_condition(manifest, identifier):
    row = next((row for row in manifest["checkpoints"] if row["id"] == identifier), None)
    if row is None or row.get("partition") != "development":
        raise ValueError("Occupancy requires a declared development endpoint; held-out interiors are excluded")
    return row


def tensor_counts(value, *, attention=False):
    """Count actual nonzero rows/tiles; never infer layout from marginal sparsity."""
    if value.dtype != torch.bfloat16 or value.ndim < 2:
        raise ValueError("Expected BF16 activation with at least two dimensions")
    if attention and value.ndim != 4:
        raise ValueError("Attention taps must expose [B,H,T,d]")
    flat = value.detach().reshape(-1, value.shape[-1])
    rows, width = flat.shape
    if not rows or not width:
        raise ValueError("Empty activation")
    nonzero = flat != 0
    tensors = [torch.isfinite(flat).sum(dtype=torch.int64),
               (~nonzero).sum(dtype=torch.int64),
               (~nonzero.any(dim=1)).sum(dtype=torch.int64)]
    tiles = {}
    if not attention:
        for tm, tk in TILE_SHAPES:
            if rows % tm or width % tk:
                raise ValueError("Linear activation must fully cover declared 2D tiles; no padded estimand")
            active = nonzero.reshape(rows // tm, tm, width // tk, tk).any(dim=3).any(dim=1)
            tensors.append((~active).sum(dtype=torch.int64))
            tiles[f"m{tm}_k{tk}"] = {"shape": [tm, tk], "tiles": (rows // tm) * (width // tk)}
    if attention:
        head_nonzero = nonzero.reshape(value.shape[0], value.shape[1], -1).any(dim=2)
        tensors.append((~head_nonzero).sum(dtype=torch.int64))
    # One host synchronization per hook, not one per individual statistic.
    counts = torch.stack(tensors).cpu().tolist()
    result = {"kind": "attention" if attention else "linear", "width": width,
              "calls": 1, "elements": flat.numel(), "finite_count": counts[0],
              "zero_count": counts[1], "rows": rows, "zero_rows": counts[2], "tiles": tiles}
    for index, tile in enumerate(tiles.values()):
        tile["zero_tiles"] = counts[3 + index]
    if attention:
        result.update(head_sequences=value.shape[0] * value.shape[1], zero_head_sequences=counts[-1])
    return result


def add_counts(destination, source):
    if not destination:
        destination.update(json.loads(json.dumps(source)))
        return
    if destination["kind"] != source["kind"] or destination["width"] != source["width"]:
        raise ValueError("Cannot pool mismatched activation shapes or kinds")
    for key in ("calls", "elements", "finite_count", "zero_count", "rows", "zero_rows",
                "head_sequences", "zero_head_sequences"):
        if key in source:
            destination[key] += source[key]
    if destination["tiles"].keys() != source["tiles"].keys():
        raise ValueError("Cannot pool mismatched tile definitions")
    for key, tile in source["tiles"].items():
        if destination["tiles"][key]["shape"] != tile["shape"]:
            raise ValueError("Cannot pool mismatched tiles")
        destination["tiles"][key]["tiles"] += tile["tiles"]
        destination["tiles"][key]["zero_tiles"] += tile["zero_tiles"]


def with_fractions(counts):
    result = json.loads(json.dumps(counts))
    result["zero_fraction"] = result["zero_count"] / result["elements"]
    result["all_zero_row_fraction"] = result["zero_rows"] / result["rows"]
    for tile in result["tiles"].values():
        tile["all_zero_tile_fraction"] = tile["zero_tiles"] / tile["tiles"]
    if "head_sequences" in result:
        result["all_zero_head_sequence_fraction"] = result["zero_head_sequences"] / result["head_sequences"]
    return result


class NativeOccupancy:
    """Observe true Linear inputs and already-installed canonical attention taps."""
    def __init__(self, model):
        self.model, self.handles, self.per_layer, self.metadata = model, [], {}, []
        self.missing_attention_taps = []

    def _collect(self, name, value, attention=False):
        row = tensor_counts(value, attention=attention)
        add_counts(self.per_layer.setdefault(name, {}), row)

    def __enter__(self):
        for index, layer in enumerate(self.model.gpt_neox.layers):
            for site, module, path in (
                ("a", layer.attention.query_key_value, "attention.query_key_value"),
                ("m", layer.mlp.dense_h_to_4h, "mlp.dense_h_to_4h"),
                ("h", layer.mlp.dense_4h_to_h, "mlp.dense_4h_to_h"),
                ("z", layer.attention.dense, "attention.dense")):
                name = f"{site}.layer_{index}"
                def pre_hook(_module, inputs, name=name):
                    self._collect(name, inputs[0])
                self.handles.append(module.register_forward_pre_hook(pre_hook))
                self.metadata.append({"name": name, "hook": "native Linear input",
                                      "module": f"gpt_neox.layers.{index}.{path}"})
            for site in ("q_post", "k_post", "v"):
                name = f"{site}.layer_{index}"
                tap = getattr(layer.attention, f"{site}_site", None)
                if tap is None:
                    self.missing_attention_taps.append(name)
                    continue
                def post_hook(_module, _inputs, output, name=name):
                    self._collect(name, output, attention=True)
                self.handles.append(tap.register_forward_hook(post_hook))
                self.metadata.append({"name": name, "hook": "existing canonical post-gate tap",
                                      "module": f"gpt_neox.layers.{index}.attention.{site}_site"})
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self):
        pooled = {}
        for name, row in self.per_layer.items():
            add_counts(pooled.setdefault(name.split(".layer_")[0], {}), row)
        return {"per_layer": {name: with_fractions(row) for name, row in self.per_layer.items()},
                "pooled_by_site": {name: with_fractions(row) for name, row in pooled.items()},
                "missing_attention_taps": self.missing_attention_taps,
                "attention_policy": "observe existing taps only; never install ports or rewrite forward",
                "integer_pooling": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--seconds", type=int, default=60)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    directory = inside(RUN / "artifacts", args.attempt)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def check():
        if time.monotonic() - started >= args.seconds:
            raise TimeoutError("Development occupancy deadline; external watchdog must bound stuck CUDA calls")

    def emit(stage, **fields):
        row = {"stage": stage, "condition": args.condition,
               "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               "elapsed_seconds": time.monotonic() - started, **fields}
        write_json(directory / "status.json", row)
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for the declared native BF16 diagnostic")
        cfg = config()
        for name, actual in (("torch", torch.__version__.split("+")[0]), ("numpy", np.__version__),
                             ("transformers", transformers.__version__)):
            if actual != cfg["runtime"][name]:
                raise RuntimeError(f"Pinned {name} mismatch")
        manifest = read_json(RUN / "prelaunch/input_manifest.json")
        if manifest["config_sha256"] != sha256(RUN / "config.json"):
            raise ValueError("Input manifest/config mismatch")
        endpoint = development_condition(manifest, args.condition)
        for item in endpoint["files"] + endpoint["provenance"]:
            verify_record(item)
        length = cfg["inputs"]["sequence_length"]
        development = np.memmap(verify_record(manifest["development"]), dtype=np.int32, mode="r").reshape(-1, length)
        if length != 2048 or len(development) != 64 or len(manifest["development_source_block_ids"]) != 64:
            raise ValueError("Expected 64 declared length-2048 training blocks")
        source_paths = [Path(__file__), RUN / "run025_common.py", RUN / "config.json"]
        source_paths += sorted((ROOT / "src/sparsity_research").glob("*.py"))
        sources = [record(path) for path in source_paths]
        torch.manual_seed(2503)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        emit("loading")
        model = load_checkpoint_pythia(transformers.AutoModelForCausalLM, ROOT / endpoint["checkpoint"], torch=torch)
        model = model.cuda().bfloat16().eval()
        model.set_attn_implementation("sdpa")
        model.config.use_cache = False
        topology = topology_metadata(model)
        identity = {"condition": endpoint, "input_manifest": record(RUN / "prelaunch/input_manifest.json"),
                    "development": manifest["development"], "selected_source_block_ids": manifest["development_source_block_ids"][:BLOCKS],
                    "selected_bytes_sha256": hashlib.sha256(development[:BLOCKS].tobytes()).hexdigest(),
                    "blocks": BLOCKS, "input_tokens": BLOCKS * length, "prediction_tokens": BLOCKS * (length - 1),
                    "selection": "first four of the fixed 64 train-split development blocks", "sources": sources,
                    "topology": topology, "workload": "native BF16 B1 T2048; unchanged gates and SDPA; no candidate kernel",
                    "tile_shapes": [list(shape) for shape in TILE_SHAPES],
                    "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                    "gpu": torch.cuda.get_device_name(), "max_seconds": args.seconds,
                    "scope": "diagnostic occupancy, not canonical validation R_model or benchmark timing"}
        write_json(directory / "manifest.json", identity)
        loss_sum, predictions = 0., 0
        with NativeOccupancy(model) as capture, torch.inference_mode():
            write_json(directory / "hook-coverage.json", {"hooks": capture.metadata,
                       "missing_attention_taps": capture.missing_attention_taps})
            for index in range(BLOCKS):
                check()
                ids = torch.tensor(development[index].copy(), device="cuda", dtype=torch.long)[None]
                logits = model(input_ids=ids, use_cache=False, output_attentions=False).logits
                block_loss = F.cross_entropy(logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                                            ids[:, 1:].reshape(-1), reduction="sum")
                loss_sum += float(block_loss)
                predictions += ids.shape[1] - 1
                summary = capture.summary()
                write_json(directory / "occupancy.json", {"completed_blocks": index + 1, "complete": False, **summary})
                if any(row["finite_count"] != row["elements"] for row in summary["per_layer"].values()):
                    raise RuntimeError("Nonfinite native activation detected; counts preserved")
                emit("block", completed=index + 1, total=BLOCKS,
                     native_development_loss=loss_sum / predictions,
                     input_tokens_per_second=(index + 1) * length / (time.monotonic() - started))
                del logits, block_loss
        for item in sources:
            verify_record(item)
        if set(summary["per_layer"]) != {item["name"] for item in capture.metadata}:
            raise RuntimeError("A registered native diagnostic hook was never invoked")
        if any(row["calls"] != BLOCKS for row in summary["per_layer"].values()):
            raise RuntimeError("Expected one hook invocation per site/layer per selected block")
        write_json(directory / "occupancy.json", {"completed_blocks": BLOCKS, "complete": True,
                   "native_development_loss": loss_sum / predictions, **summary})
        emit("complete", completed=BLOCKS, total=BLOCKS, native_development_loss=loss_sum / predictions)
        return 0
    except Exception as error:
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        emit("failed", error_type=type(error).__name__, error=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
