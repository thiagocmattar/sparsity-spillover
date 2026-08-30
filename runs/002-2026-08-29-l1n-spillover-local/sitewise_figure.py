"""Post-hoc m diagnostic and four-panel Run 002 publication figure."""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import json
import math
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Mapping

from sparsity_research.artifacts import build_transfer_inventory
from sparsity_research.capture import ActivationCapture
from sparsity_research.evaluation import evaluate_complete_blocks
from sparsity_research.metrics import ActivationAccumulator
from sparsity_research.pythia import load_checkpoint_pythia, topology_metadata

from run_config import (
    DEFAULT_CONFIG,
    RUN_DIR,
    cache_identity,
    inventory_content_sha256,
    load_config,
    load_verified_caches,
    mapping,
    require_cuda,
    require_validation_coverage,
    write_json,
)


VERIFICATION_PATH = RUN_DIR / "artifacts" / "verification.json"
M_DIAGNOSTIC_PATH = RUN_DIR / "artifacts" / "posthoc-m-activation-statistics.json"
FIGURE_DATA_PATH = RUN_DIR / "artifacts" / "figure_data_sitewise.json"
FIGURE_PATH = RUN_DIR / "figures" / "02-h-vs-site-near-zero-grid.pdf"
OBSERVATION_PATH = RUN_DIR / "observations" / "O002-h-vs-site-near-zero-grid.md"
COMPLETION_PATH = RUN_DIR / "artifacts" / "sitewise_figure_completion.json"
SITES = ("q_post", "k_post", "v", "m")
EPSILON = 0.001


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pooled_counts(statistics: Mapping[str, Any], site: str, epsilon: float) -> dict[str, Any]:
    matches = [row for row in statistics["pooled_by_site"] if row["name"] == site]
    if len(matches) != 1:
        raise ValueError(f"Expected one pooled row for site {site!r}.")
    row = matches[0]
    key = f"{epsilon:g}"
    hits = int(row["threshold_hits"][key])
    total = int(row["total"])
    if total <= 0 or not 0 <= hits <= total:
        raise ValueError(f"Invalid near-zero counts for site {site!r}.")
    fraction = hits / total
    recorded = float(row["threshold_fractions"][key])
    if not math.isclose(fraction, recorded, rel_tol=0.0, abs_tol=1e-15):
        raise ValueError(f"Stored fraction disagrees with integer counts for {site!r}.")
    return {"hits": hits, "total": total, "fraction": fraction, "percent": 100.0 * fraction}


def _validate_layer_pool(statistics: Mapping[str, Any], site: str, epsilon: float) -> None:
    key = f"{epsilon:g}"
    layers = [row for row in statistics["rows"] if row["name"].startswith(f"{site}.layer_")]
    if len(layers) != 6:
        raise ValueError(f"Expected six layer rows for {site!r}, found {len(layers)}.")
    pooled = _pooled_counts(statistics, site, epsilon)
    if sum(int(row["total"]) for row in layers) != pooled["total"]:
        raise ValueError(f"Layer totals do not pool to the {site!r} total.")
    if sum(int(row["threshold_hits"][key]) for row in layers) != pooled["hits"]:
        raise ValueError(f"Layer hits do not pool to the {site!r} hit count.")


def compute_m_diagnostics(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_path: str | Path = M_DIAGNOSTIC_PATH,
) -> dict[str, Any]:
    """Measure m at the retained final checkpoints without rewriting attempts."""

    output = Path(output_path)
    if output.exists():
        artifact = json.loads(output.read_text(encoding="utf-8"))
        if artifact.get("status") != "verified" or len(artifact.get("conditions", [])) != 10:
            raise ValueError("Existing post-hoc m diagnostic is incomplete.")
        return artifact

    import numpy as np
    import torch
    from transformers import AutoModelForCausalLM

    config = load_config(config_path)
    require_cuda(torch)
    verification = json.loads(VERIFICATION_PATH.read_text(encoding="utf-8"))
    if verification.get("status") != "verified" or verification.get("evidence_label") != "valid":
        raise ValueError("Post-hoc measurement requires valid terminal Run 002 evidence.")
    train_tokens, validation_tokens, train_metadata, validation_metadata, cache_seconds = (
        load_verified_caches(config, np=np)
    )
    del train_tokens, train_metadata

    records = []
    started = perf_counter()
    for source in verification["conditions"]:
        attempt_id = str(source["attempt_id"])
        attempt_dir = RUN_DIR / "artifacts" / "attempts" / attempt_id
        manifest_path = attempt_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"Attempt {attempt_id} is not terminally completed.")
        if manifest["condition"]["id"] != source["condition"]["id"]:
            raise ValueError(f"Condition identity mismatch for {attempt_id}.")

        checkpoint_dir = attempt_dir / manifest["checkpoint"]["path"]
        inventory = build_transfer_inventory(checkpoint_dir)
        checkpoint_hash = inventory_content_sha256(inventory)
        if checkpoint_hash != manifest["checkpoint"]["content_sha256"]:
            raise ValueError(f"Checkpoint content hash mismatch for {attempt_id}.")
        if checkpoint_hash != source["checkpoint_content_sha256"]:
            raise ValueError(f"Verification/checkpoint hash mismatch for {attempt_id}.")

        condition_started = perf_counter()
        model = load_checkpoint_pythia(AutoModelForCausalLM, checkpoint_dir, torch=torch)
        model.config.use_cache = False
        model.to(device=torch.device("cuda"), dtype=torch.float32)
        if topology_metadata(model) != manifest["topology"]:
            raise ValueError(f"Reloaded topology differs for {attempt_id}.")

        accumulator = ActivationAccumulator((0.0, EPSILON, 0.01))
        torch.cuda.synchronize()
        evaluation_started = perf_counter()
        with ActivationCapture(model, ["m"], torch=torch) as capture:

            def observe(_output: Any, _batch_sequences: int) -> None:
                accumulator.update(capture.activations, torch=torch)
                capture.clear()

            coverage = evaluate_complete_blocks(
                model=model,
                tokens=validation_tokens,
                block_size=int(mapping(config, "data")["sequence_length"]),
                batch_size=int(mapping(config, "validation")["batch_size"]),
                device=torch.device("cuda"),
                torch=torch,
                np=np,
                autocast_dtype=torch.bfloat16,
                after_batch=observe,
            )
        torch.cuda.synchronize()
        evaluation_seconds = perf_counter() - evaluation_started
        require_validation_coverage(coverage, config)
        if not math.isclose(
            float(coverage["loss"]),
            float(source["final_validation_loss"]),
            rel_tol=0.0,
            abs_tol=1e-5,
        ):
            raise ValueError(f"Post-hoc validation loss changed for {attempt_id}.")

        statistics = {
            "rows": accumulator.rows(),
            "pooled_by_site": accumulator.pooled_by_site(),
        }
        _validate_layer_pool(statistics, "m", EPSILON)
        original_statistics = attempt_dir / "diagnostics" / "activation_statistics.json"
        records.append(
            {
                "attempt_id": attempt_id,
                "condition": source["condition"],
                "checkpoint_content_sha256": checkpoint_hash,
                "original_activation_statistics": {
                    "path": original_statistics.relative_to(RUN_DIR).as_posix(),
                    "sha256": _sha256(original_statistics),
                },
                "coverage": coverage,
                "statistics": statistics,
                "evaluation_seconds": evaluation_seconds,
                "total_seconds": perf_counter() - condition_started,
            }
        )
        del model
        gc.collect()
        torch.cuda.empty_cache()

    artifact = {
        "schema_version": 1,
        "status": "verified",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "site": "m",
        "site_definition": "MLP-branch LayerNorm/gate output feeding the W1/up projection",
        "epsilon": EPSILON,
        "source_verification": {
            "path": VERIFICATION_PATH.relative_to(RUN_DIR).as_posix(),
            "sha256": _sha256(VERIFICATION_PATH),
            "evidence_label": verification["evidence_label"],
        },
        "validation_cache": cache_identity(validation_metadata),
        "coverage": {
            "documents": int(mapping(config, "validation")["documents"]),
            "sequences": int(mapping(config, "validation")["complete_sequences"]),
            "input_tokens": int(mapping(config, "validation")["input_tokens"]),
            "excluded_tail_tokens": int(mapping(config, "validation")["excluded_tail_tokens"]),
        },
        "cache_verification_seconds": cache_seconds,
        "wall_seconds": perf_counter() - started,
        "conditions": records,
        "source_code": {
            "06_plot_sitewise.py": _sha256(RUN_DIR / "06_plot_sitewise.py"),
            "sitewise_figure.py": _sha256(Path(__file__)),
        },
    }
    write_json(output, artifact)
    return artifact


def reduce_sitewise_points(
    verification: Mapping[str, Any],
    m_artifact: Mapping[str, Any],
    statistics_loader: Callable[[str], Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Reduce every plotted coordinate from integer hits and denominators."""

    if verification.get("status") != "verified" or verification.get("evidence_label") != "valid":
        raise ValueError("Figure input must be valid terminal verification evidence.")
    if m_artifact.get("status") != "verified" or float(m_artifact["epsilon"]) != EPSILON:
        raise ValueError("Figure input requires the verified epsilon-1e-3 m diagnostic.")
    m_by_attempt = {row["attempt_id"]: row for row in m_artifact["conditions"]}
    grouped: dict[str, list[dict[str, Any]]] = {"gelu": [], "relu": []}
    for source in verification["conditions"]:
        attempt_id = source["attempt_id"]
        condition = source["condition"]
        activation = condition["activation"]
        if activation not in grouped or attempt_id not in m_by_attempt:
            raise ValueError(f"Incomplete sitewise input for {attempt_id}.")
        original = statistics_loader(attempt_id)
        m_record = m_by_attempt[attempt_id]
        sites = {
            site: _pooled_counts(
                m_record["statistics"] if site == "m" else original,
                site,
                EPSILON,
            )
            for site in ("h", *SITES)
        }
        point = {
            "attempt_id": attempt_id,
            "condition_id": condition["id"],
            "activation": activation,
            "order": int(condition["order"]),
            "label": "control" if condition["is_control"] else f"lambda={condition['pressure_weight']:g}",
            "pressure_weight": float(condition["pressure_weight"]),
            "is_control": bool(condition["is_control"]),
            "near_zero": sites,
            "final_validation_loss": float(source["final_validation_loss"]),
        }
        if not all(math.isfinite(row["percent"]) for row in sites.values()):
            raise ValueError(f"Nonfinite plotted value for {attempt_id}.")
        grouped[activation].append(point)

    expected = ["control", "lambda=0.1", "lambda=0.5", "lambda=1", "lambda=5"]
    for activation, points in grouped.items():
        points.sort(key=lambda point: point["order"])
        if [point["label"] for point in points] != expected:
            raise ValueError(f"Incomplete or misordered {activation} series.")
    return grouped


def generate_sitewise_figure() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    verification = json.loads(VERIFICATION_PATH.read_text(encoding="utf-8"))
    m_artifact = compute_m_diagnostics()

    def load_statistics(attempt_id: str) -> Mapping[str, Any]:
        record = next(row for row in m_artifact["conditions"] if row["attempt_id"] == attempt_id)
        path = RUN_DIR / record["original_activation_statistics"]["path"]
        if _sha256(path) != record["original_activation_statistics"]["sha256"]:
            raise ValueError(f"Original activation diagnostic changed for {attempt_id}.")
        return json.loads(path.read_text(encoding="utf-8"))

    grouped = reduce_sitewise_points(verification, m_artifact, load_statistics)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
        }
    )
    styles = {
        "gelu": {"display": "GeLU", "color": "#0072B2", "marker": "o", "linestyle": "-"},
        "relu": {"display": "ReLU", "color": "#D55E00", "marker": "s", "linestyle": "--"},
    }
    label_offsets = {
        "gelu": [(4, -10), (4, 7), (0, 7), (0, 7), (0, 7)],
        "relu": [(0, -11), (0, 7), (-2, -11), (-2, 7), (0, -11)],
    }
    all_points = grouped["gelu"] + grouped["relu"]
    x_max = max(point["near_zero"]["h"]["percent"] for point in all_points)
    y_max = max(point["near_zero"][site]["percent"] for point in all_points for site in SITES)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.975, top=0.86, bottom=0.14, hspace=0.28, wspace=0.20)
    legend_handles = []
    for ax, site in zip(axes.flat, SITES, strict=True):
        for activation in ("gelu", "relu"):
            points = grouped[activation]
            style = styles[activation]
            line = ax.plot(
                [point["near_zero"]["h"]["percent"] for point in points],
                [point["near_zero"][site]["percent"] for point in points],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                markersize=5.7,
                markerfacecolor="white",
                markeredgewidth=1.6,
                label=style["display"],
            )[0]
            if site == SITES[0]:
                legend_handles.append(line)
            for index, point in enumerate(points):
                dx, dy = label_offsets[activation][index]
                ax.annotate(
                    "ctrl" if point["is_control"] else f"{point['pressure_weight']:g}",
                    (point["near_zero"]["h"]["percent"], point["near_zero"][site]["percent"]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left" if dx > 0 else "center",
                    va="bottom" if dy > 0 else "top",
                    fontsize=7.6,
                    color="#202020",
                )
        ax.set_title(site)
        ax.set_xlim(0.0, max(1.0, x_max * 1.10))
        ax.set_ylim(0.0, max(0.01, y_max * 1.25))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.grid(axis="both", color="#D9D9D9", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Near-zero mass at h (%)")
    for ax, site in zip(axes[:, 0], ("q_post", "v"), strict=True):
        ax.set_ylabel(f"Near-zero mass at {site} (%)")
    for ax, site in zip(axes[:, 1], ("k_post", "m"), strict=True):
        ax.set_ylabel(f"Near-zero mass at {site} (%)")
    fig.suptitle("Pythia-14M L1N sitewise near-zero mass", y=0.955, fontsize=15)
    fig.legend(
        legend_handles,
        [styles[activation]["display"] for activation in ("gelu", "relu")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.035,
        "Seed 0; |x| <= 1e-3; 451 updates per condition; all 338 validation blocks; labels are control or lambda",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#404040",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURE_PATH, format="pdf", bbox_inches="tight")
    plt.close(fig)

    figure_data = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "epsilon": EPSILON,
        "x_site": "h",
        "y_sites": list(SITES),
        "estimand": "each coordinate is integer near-zero hits pooled across all validation batches and six layers, divided by its pooled element count",
        "source_verification": {
            "path": VERIFICATION_PATH.relative_to(RUN_DIR).as_posix(),
            "sha256": _sha256(VERIFICATION_PATH),
        },
        "source_m_diagnostic": {
            "path": M_DIAGNOSTIC_PATH.relative_to(RUN_DIR).as_posix(),
            "sha256": _sha256(M_DIAGNOSTIC_PATH),
        },
        "series": grouped,
        "shared_axes": {"x_starts_at_zero": True, "y_starts_at_zero": True, "shared_y_scale": True},
        "figure": {
            "path": FIGURE_PATH.relative_to(RUN_DIR).as_posix(),
            "bytes": FIGURE_PATH.stat().st_size,
            "sha256": _sha256(FIGURE_PATH),
        },
        "source_code": {
            "06_plot_sitewise.py": _sha256(RUN_DIR / "06_plot_sitewise.py"),
            "sitewise_figure.py": _sha256(Path(__file__)),
        },
    }
    write_json(FIGURE_DATA_PATH, figure_data)
    _write_observation(grouped, verification["evidence_label"])
    completion = {
        "schema_version": 1,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "figure": figure_data["figure"],
        "figure_data": FIGURE_DATA_PATH.relative_to(RUN_DIR).as_posix(),
        "observation": OBSERVATION_PATH.relative_to(RUN_DIR).as_posix(),
        "m_diagnostic": M_DIAGNOSTIC_PATH.relative_to(RUN_DIR).as_posix(),
    }
    write_json(COMPLETION_PATH, completion)
    return FIGURE_PATH


def _write_observation(
    grouped: Mapping[str, list[Mapping[str, Any]]], evidence_label: str
) -> None:
    rows = []
    for activation in ("gelu", "relu"):
        for point in grouped[activation]:
            values = point["near_zero"]
            rows.append(
                "| {activation} | {label} | {h:.6f} | {q:.6f} | {k:.6f} | {v:.6f} | {m:.6f} |".format(
                    activation="GeLU" if activation == "gelu" else "ReLU",
                    label=point["label"],
                    h=values["h"]["percent"],
                    q=values["q_post"]["percent"],
                    k=values["k_post"]["percent"],
                    v=values["v"]["percent"],
                    m=values["m"]["percent"],
                )
            )
    text = f"""# O002 - Sitewise L1N spillover trajectories

## Question

How do the individual `q_post`, `k_post`, `v`, and `m` near-zero responses vary
with near-zero mass at pressured `h`, and how do those trajectories differ for
GeLU and ReLU?

## Method and coverage

The four panels use the ten matched final Run 002 checkpoints. Every coordinate
is measured over all 500 MiniPile validation documents, 338 complete 2,048-token
blocks, 692,224 input tokens, all six layers, and the declared 1,444-token
excluded tail. The threshold is `abs(x) <= 1e-3`. Integer hits and denominators
are pooled across validation batches and layers before division; there is no
condition, site, or seed averaging.

The original terminal diagnostic supplies `h`, `q_post`, `k_post`, and `v`.
Because `m` was not selected in the approved pre-launch diagnostic set, it was
measured post hoc by reloading each retained, hash-verified final checkpoint and
rerunning the same complete validation pass. Operational `m` is the MLP-branch
LayerNorm/gate output feeding W1. Every post-hoc loss reproduced its stored final
validation loss within `1e-5`, and all ten `m` records contain six layer rows
whose integer counts reconcile to the pooled row.

## Values

| Activation | Pressure label | h (%) | q_post (%) | k_post (%) | v (%) | m (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Figure caption and encoding

**Figure 02. Sitewise near-zero mass versus pressured FFN `h` for local
Pythia-14M L1N pretraining.** The 2x2 panels show `q_post`, `k_post`, `v`, and
`m`. Blue circles/solid lines denote GeLU; orange squares/dashed lines denote
ReLU. Point labels are the no-pressure control (`ctrl`) or naive-L1 weight.
Lines connect control through increasing pressure strength as visual guides,
not fitted relationships. All panels use the same zero-based x and y scales and
report percent at epsilon `1e-3`.

## Interpretation and limits

The panels separate the attention-site average shown in Figure 01 and add the
MLP input branch. They are descriptive trajectories from one seed and a short
training horizon. GeLU-versus-ReLU differences include the activation operator
change. Near-zero activation mass is not an exact-zero product count, removable
compute, measured speedup, a causal route, or a long-horizon optimum. The source
attempt evidence is `{evidence_label}`; the additional `m` measurement is a
checkpoint-reconstructible post-hoc diagnostic, not a training-time gradient
measurement.

## Provenance

- Terminal source: `../artifacts/verification.json`
- Original terminal activations: `../artifacts/attempts/*/diagnostics/activation_statistics.json`
- Post-hoc `m`: `../artifacts/posthoc-m-activation-statistics.json`
- Numerical reduction: `../artifacts/figure_data_sitewise.json`
- Source script: `../06_plot_sitewise.py` and `../sitewise_figure.py`
- Output: `../figures/02-h-vs-site-near-zero-grid.pdf`
"""
    OBSERVATION_PATH.write_text(text, encoding="utf-8", newline="\n")
