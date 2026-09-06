"""Development-only temporal-tile occupancy at existing q/k/v gate taps.

This is an append-only companion to ``collect_development_occupancy.py``.  The
earlier diagnostic counted attention rows and full head sequences but did not
measure the consecutive query-row tiles consumed by K002.  This file leaves
that executed source unchanged and measures the missing estimand directly.
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
sys.path[:0] = [str(HERE), str(RUN), str(ROOT / "src")]
from collect_development_occupancy import BLOCKS, development_condition
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

TEMPORAL_TILE_ROWS = (1, 2, 4, 8, 16, 32, 64)


def attention_counts(value):
    """Return integer occupancy counts for a native [B,H,T,d] BF16 tap."""
    if value.dtype != torch.bfloat16 or value.ndim != 4:
        raise ValueError("Attention taps must expose BF16 [B,H,T,d]")
    batch, heads, tokens, width = value.shape
    if not all((batch, heads, tokens, width)):
        raise ValueError("Empty attention activation")
    if any(tokens % rows for rows in TEMPORAL_TILE_ROWS):
        raise ValueError("Sequence length must exactly cover every temporal tile")

    nonzero = value.detach() != 0
    active_rows = nonzero.any(dim=-1)
    tensors = [
        torch.isfinite(value).sum(dtype=torch.int64),
        (~nonzero).sum(dtype=torch.int64),
        (~active_rows).sum(dtype=torch.int64),
        (~active_rows.reshape(batch, heads, -1).any(dim=-1)).sum(dtype=torch.int64),
    ]
    temporal_tiles = {}
    for tile_rows in TEMPORAL_TILE_ROWS:
        active_tiles = active_rows.reshape(
            batch, heads, tokens // tile_rows, tile_rows
        ).any(dim=-1)
        tensors.append((~active_tiles).sum(dtype=torch.int64))
        temporal_tiles[f"t{tile_rows}"] = {
            "rows": tile_rows,
            "tiles": batch * heads * (tokens // tile_rows),
        }

    # One synchronization per tap invocation, independent of statistic count.
    counts = torch.stack(tensors).cpu().tolist()
    for index, tile in enumerate(temporal_tiles.values()):
        tile["zero_tiles"] = counts[4 + index]
    return {
        "calls": 1,
        "elements": value.numel(),
        "finite_count": counts[0],
        "zero_count": counts[1],
        "rows": batch * heads * tokens,
        "zero_rows": counts[2],
        "head_sequences": batch * heads,
        "zero_head_sequences": counts[3],
        "tokens": tokens,
        "width": width,
        "temporal_tiles": temporal_tiles,
    }


def add_counts(destination, source):
    if not destination:
        destination.update(json.loads(json.dumps(source)))
        return
    if (destination["tokens"], destination["width"]) != (
        source["tokens"],
        source["width"],
    ):
        raise ValueError("Cannot pool mismatched attention shapes")
    for key in (
        "calls",
        "elements",
        "finite_count",
        "zero_count",
        "rows",
        "zero_rows",
        "head_sequences",
        "zero_head_sequences",
    ):
        destination[key] += source[key]
    if destination["temporal_tiles"].keys() != source["temporal_tiles"].keys():
        raise ValueError("Cannot pool mismatched temporal tile definitions")
    for key, tile in source["temporal_tiles"].items():
        destination["temporal_tiles"][key]["tiles"] += tile["tiles"]
        destination["temporal_tiles"][key]["zero_tiles"] += tile["zero_tiles"]


def with_fractions(counts):
    result = json.loads(json.dumps(counts))
    result["zero_fraction"] = result["zero_count"] / result["elements"]
    result["zero_row_fraction"] = result["zero_rows"] / result["rows"]
    result["zero_head_sequence_fraction"] = (
        result["zero_head_sequences"] / result["head_sequences"]
    )
    for tile in result["temporal_tiles"].values():
        tile["zero_tile_fraction"] = tile["zero_tiles"] / tile["tiles"]
    return result


class AttentionTileOccupancy:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.per_layer = {}
        self.metadata = []
        self.missing = []

    def _collect(self, name, value):
        add_counts(self.per_layer.setdefault(name, {}), attention_counts(value))

    def __enter__(self):
        for layer_index, layer in enumerate(self.model.gpt_neox.layers):
            for site in ("q_post", "k_post", "v"):
                name = f"{site}.layer_{layer_index}"
                tap = getattr(layer.attention, f"{site}_site", None)
                if tap is None:
                    self.missing.append(name)
                    continue

                def hook(_module, _inputs, output, name=name):
                    self._collect(name, output)

                self.handles.append(tap.register_forward_hook(hook))
                self.metadata.append(
                    {
                        "name": name,
                        "hook": "existing canonical post-gate tap",
                        "module": (
                            f"gpt_neox.layers.{layer_index}.attention.{site}_site"
                        ),
                    }
                )
        return self

    def __exit__(self, *_args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def summary(self):
        pooled = {}
        for name, row in self.per_layer.items():
            add_counts(pooled.setdefault(name.split(".layer_")[0], {}), row)
        return {
            "per_layer": {
                name: with_fractions(row) for name, row in self.per_layer.items()
            },
            "pooled_by_site": {
                name: with_fractions(row) for name, row in pooled.items()
            },
            "missing_attention_taps": self.missing,
            "integer_pooling": True,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--seconds", type=int, default=120)
    args = parser.parse_args()
    if args.seconds <= 0:
        parser.error("--seconds must be positive")
    directory = inside(RUN / "artifacts", args.attempt)
    directory.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def check():
        if time.monotonic() - started >= args.seconds:
            raise TimeoutError("Attention occupancy diagnostic deadline")

    def emit(stage, **fields):
        row = {
            "stage": stage,
            "condition": args.condition,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": time.monotonic() - started,
            **fields,
        }
        write_json(directory / "status.json", row)
        with (directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(row, allow_nan=False), flush=True)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required for native BF16 occupancy")
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
            raise ValueError("Input manifest/config mismatch")
        endpoint = development_condition(manifest, args.condition)
        for item in endpoint["files"] + endpoint["provenance"]:
            verify_record(item)
        length = cfg["inputs"]["sequence_length"]
        development = np.memmap(
            verify_record(manifest["development"]), dtype=np.int32, mode="r"
        ).reshape(-1, length)
        if length != 2048 or len(development) != 64:
            raise ValueError("Expected 64 declared length-2048 development blocks")
        sources = [
            record(Path(__file__)),
            record(HERE / "collect_development_occupancy.py"),
            record(RUN / "run025_common.py"),
            record(RUN / "config.json"),
        ]
        torch.manual_seed(2503)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        emit("loading")
        model = load_checkpoint_pythia(
            transformers.AutoModelForCausalLM,
            ROOT / endpoint["checkpoint"],
            torch=torch,
        )
        model = model.cuda().bfloat16().eval()
        model.set_attn_implementation("sdpa")
        model.config.use_cache = False
        write_json(
            directory / "manifest.json",
            {
                "condition": endpoint,
                "input_manifest": record(RUN / "prelaunch/input_manifest.json"),
                "development": manifest["development"],
                "selected_source_block_ids": manifest[
                    "development_source_block_ids"
                ][:BLOCKS],
                "selected_bytes_sha256": hashlib.sha256(
                    development[:BLOCKS].tobytes()
                ).hexdigest(),
                "blocks": BLOCKS,
                "input_tokens": BLOCKS * length,
                "prediction_tokens": BLOCKS * (length - 1),
                "selection": "first four fixed train-split development blocks",
                "sources": sources,
                "topology": topology_metadata(model),
                "workload": "native BF16 B1 T2048; unchanged gates and SDPA",
                "temporal_tile_rows": list(TEMPORAL_TILE_ROWS),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(),
                "max_seconds": args.seconds,
                "scope": "development diagnostic; not canonical R_model or timing",
            },
        )
        loss_sum = 0.0
        predictions = 0
        with AttentionTileOccupancy(model) as capture, torch.inference_mode():
            write_json(
                directory / "hook-coverage.json",
                {"hooks": capture.metadata, "missing_attention_taps": capture.missing},
            )
            if capture.missing:
                raise RuntimeError("Declared condition lacks canonical attention taps")
            for index in range(BLOCKS):
                check()
                ids = torch.tensor(
                    development[index].copy(), device="cuda", dtype=torch.long
                )[None]
                logits = model(
                    input_ids=ids, use_cache=False, output_attentions=False
                ).logits
                block_loss = F.cross_entropy(
                    logits[:, :-1].float().reshape(-1, logits.shape[-1]),
                    ids[:, 1:].reshape(-1),
                    reduction="sum",
                )
                loss_sum += float(block_loss)
                predictions += ids.shape[1] - 1
                summary = capture.summary()
                write_json(
                    directory / "attention-tile-occupancy.json",
                    {"completed_blocks": index + 1, "complete": False, **summary},
                )
                if any(
                    row["finite_count"] != row["elements"]
                    for row in summary["per_layer"].values()
                ):
                    raise RuntimeError("Nonfinite native attention activation")
                emit(
                    "block",
                    completed=index + 1,
                    total=BLOCKS,
                    native_development_loss=loss_sum / predictions,
                )
                del logits, block_loss
        if set(summary["per_layer"]) != {
            item["name"] for item in capture.metadata
        }:
            raise RuntimeError("A registered attention tap was never invoked")
        if any(
            row["calls"] != BLOCKS for row in summary["per_layer"].values()
        ):
            raise RuntimeError("Expected one attention-tap call per layer and block")
        for item in sources:
            verify_record(item)
        write_json(
            directory / "attention-tile-occupancy.json",
            {
                "completed_blocks": BLOCKS,
                "complete": True,
                "native_development_loss": loss_sum / predictions,
                **summary,
            },
        )
        emit(
            "complete",
            completed=BLOCKS,
            total=BLOCKS,
            native_development_loss=loss_sum / predictions,
        )
        return 0
    except Exception as error:
        (directory / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        emit("failed", error_type=type(error).__name__, error=str(error))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
