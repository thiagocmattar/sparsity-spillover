"""Emit compact read-only phase-2 diagnostics from the live RunPod workspace."""

from __future__ import annotations

import json
import math
from pathlib import Path
import statistics


RUN = Path(
    "/workspace/run025-autoresearch-rtxpro4500-001/sparsity-spillover/"
    "runs/025-2026-09-05-pythia-agentic-sparse-kernel-search"
)


def read(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


result = {"occupancy": {}}
for condition in ("a1h", "a4-0p5", "a7-0p5"):
    path = RUN / f"artifacts/occupancy-14m-{condition}-rtxpro4500-001/occupancy.json"
    data = read(path)
    if data is None:
        continue
    sites = {}
    for site, row in data["pooled_by_site"].items():
        sites[site] = {
            "scalar_zero": round(row["zero_fraction"], 8),
            "zero_rows": round(row["all_zero_row_fraction"], 8),
            "zero_tiles": {
                key: round(tile["all_zero_tile_fraction"], 8)
                for key, tile in row["tiles"].items()
            },
        }
        if "all_zero_head_sequence_fraction" in row:
            sites[site]["zero_head_sequences"] = round(
                row["all_zero_head_sequence_fraction"], 8
            )
    result["occupancy"][condition] = {
        "blocks": data.get("blocks"),
        "loss": data.get("native_development_loss"),
        "missing_attention_taps": data["missing_attention_taps"],
        "sites": sites,
    }

attention_path = (
    RUN
    / "artifacts/attention-tiles-14m-a7-0p5-rtxpro4500-001/attention-tile-occupancy.json"
)
attention = read(attention_path)
result["attention_tiles"] = None
if attention is not None:
    result["attention_tiles"] = {
        "complete": attention["complete"],
        "completed_blocks": attention["completed_blocks"],
        "loss": attention.get("native_development_loss"),
        "missing_attention_taps": attention["missing_attention_taps"],
        "sites": {
            site: {
                "zero_rows": round(row["zero_row_fraction"], 8),
                "zero_head_sequences": round(
                    row["zero_head_sequence_fraction"], 8
                ),
                "zero_temporal_tiles": {
                    key: round(tile["zero_tile_fraction"], 8)
                    for key, tile in row["temporal_tiles"].items()
                },
            }
            for site, row in attention["pooled_by_site"].items()
        },
    }

k002 = RUN / "autoresearch/candidates/k002/artifacts/rtxpro4500-contiguous-001"
k002_patterns = ["all_zero_q", "alternating_zero_tiles", "single_active_row", "signed_dense"]
result["k002_pattern_timing"] = {}
for path in sorted(k002.glob("timing-*.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines() if line]
    shape = {}
    for input_index, pattern in enumerate(k002_patterns):
        selected = [row for row in rows if row["input_index"] == input_index]
        by_mode = {
            mode: {
                (row["repeat"], row["input_index"]): row["host_ms"]
                for row in selected
                if row["mode"] == mode
            }
            for mode in ("native_flash", "k002")
        }
        keys = sorted(by_mode["native_flash"])
        shape[pattern] = {
            "native_flash_median_ms": round(
                statistics.median(by_mode["native_flash"].values()), 6
            ),
            "k002_median_ms": round(
                statistics.median(by_mode["k002"].values()), 6
            ),
            "paired_geomean_speedup": round(
                math.exp(
                    sum(
                        math.log(
                            by_mode["native_flash"][key] / by_mode["k002"][key]
                        )
                        for key in keys
                    )
                    / len(keys)
                ),
                6,
            ),
        }
    result["k002_pattern_timing"][path.stem] = shape

dense = RUN / "autoresearch/dense_probe/artifacts/rtxpro4500-all-dense-001"
result["dense"] = {
    "status": read(dense / "status.json"),
    "setup": read(dense / "setup.json"),
    "quality": read(dense / "development-quality.json"),
    "timing": read(dense / "timing.json"),
}
dense_max = RUN / "autoresearch/dense_probe/artifacts/rtxpro4500-dense-max-autotune-002"
result["dense_max_autotune"] = {
    "status": read(dense_max / "status.json"),
    "setup": read(dense_max / "setup.json"),
    "quality": read(dense_max / "development-quality.json"),
    "timing": read(dense_max / "timing.json"),
}

numeric = RUN / "artifacts/a1h-layer-diagnostic-rtxpro4500-001"
layers = read(numeric / "block-1-layers.json")
propagation = read(numeric / "block-1-propagation.json")
layer_summary = None
if layers is not None:
    layer_summary = {
        "operations": len(layers),
        "p0_failures": sum(not row["p0_vs_native"]["pass"] for row in layers),
        "fp32_fused_failures": sum(
            not row["fp32_fused_vs_native"]["pass"] for row in layers
        ),
        "p0_vs_fp32_failures": sum(
            not row["p0_vs_fp32_fused"]["pass"] for row in layers
        ),
        "by_site": {
            site: {
                "operations": sum(row["site"] == site for row in layers),
                "p0_failures": sum(
                    row["site"] == site and not row["p0_vs_native"]["pass"]
                    for row in layers
                ),
                "max_p0_relative_l2": max(
                    (
                        row["p0_vs_native"]["relative_l2"]
                        for row in layers
                        if row["site"] == site
                    ),
                    default=None,
                ),
            }
            for site in sorted({row["site"] for row in layers})
        },
        "first_p0_failure": next(
            (
                {
                    "ordinal": row["ordinal"],
                    "layer": row["layer"],
                    "site": row["site"],
                    "module": row["module"],
                    "comparison": row["p0_vs_native"],
                }
                for row in layers
                if not row["p0_vs_native"]["pass"]
            ),
            None,
        ),
    }
result["numerics"] = {
    "status": read(numeric / "status.json"),
    "identity": read(numeric / "identity.json"),
    "layer_summary": layer_summary,
    "propagation": propagation,
    "alias": read(numeric / "block-1-alias.json"),
}

print(json.dumps(result, sort_keys=True))
