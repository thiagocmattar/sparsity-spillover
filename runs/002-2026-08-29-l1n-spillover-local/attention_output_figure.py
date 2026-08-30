"""Post-hoc W_o-output diagnostic and Run 002 publication figure."""

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
DIAGNOSTIC_PATH = (
    RUN_DIR / "artifacts" / "posthoc-attention-output-activation-statistics.json"
)
FIGURE_DATA_PATH = RUN_DIR / "artifacts" / "figure_data_attention_output.json"
FIGURE_PATH = RUN_DIR / "figures" / "03-h-vs-attention-output-near-zero.pdf"
OBSERVATION_PATH = RUN_DIR / "observations" / "O003-h-vs-attention-output-near-zero.md"
COMPLETION_PATH = RUN_DIR / "artifacts" / "attention_output_figure_completion.json"
SITE = "attention_output"
EPSILON = 0.001


class AttentionOutputCapture:
    """Capture both sides of zero-dropout between W_o and the residual sum."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self.after_wo: dict[str, Any] = {}
        self.before_residual: dict[str, Any] = {}
        self._handles: list[Any] = []

    def __enter__(self) -> "AttentionOutputCapture":
        layers = getattr(getattr(self.model, "gpt_neox", None), "layers", None)
        if layers is None or len(layers) != 6:
            raise ValueError("Attention-output capture requires six GPT-NeoX layers.")
        for index, layer in enumerate(layers):
            dense = getattr(getattr(layer, "attention", None), "dense", None)
            dropout = getattr(layer, "post_attention_dropout", None)
            if dense is None or dropout is None:
                raise ValueError(f"Could not resolve W_o/residual boundary in layer {index}.")
            name = f"{SITE}.layer_{index}"
            self._handles.append(dense.register_forward_hook(self._capture(self.after_wo, name)))
            self._handles.append(
                dropout.register_forward_hook(self._capture(self.before_residual, name))
            )
        return self

    def __exit__(self, *_args: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def clear(self) -> None:
        self.after_wo.clear()
        self.before_residual.clear()

    @staticmethod
    def _capture(target: dict[str, Any], name: str) -> Callable[[Any, Any, Any], None]:
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            if not hasattr(output, "detach"):
                raise TypeError(f"Expected tensor output at {name}.")
            target[name] = output

        return hook


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
    if not math.isclose(
        fraction,
        float(row["threshold_fractions"][key]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"Stored fraction disagrees with integer counts for {site!r}.")
    return {"hits": hits, "total": total, "fraction": fraction, "percent": 100.0 * fraction}


def _validate_layer_pool(statistics: Mapping[str, Any]) -> None:
    key = f"{EPSILON:g}"
    layers = [row for row in statistics["rows"] if row["name"].startswith(f"{SITE}.layer_")]
    if len(layers) != 6:
        raise ValueError(f"Expected six layer rows, found {len(layers)}.")
    pooled = _pooled_counts(statistics, SITE, EPSILON)
    if sum(int(row["total"]) for row in layers) != pooled["total"]:
        raise ValueError("Layer totals do not reconcile to the pooled attention-output total.")
    if sum(int(row["threshold_hits"][key]) for row in layers) != pooled["hits"]:
        raise ValueError("Layer hits do not reconcile to the pooled attention-output hits.")


def compute_attention_output_diagnostics(
    config_path: str | Path = DEFAULT_CONFIG,
    *,
    output_path: str | Path = DIAGNOSTIC_PATH,
) -> dict[str, Any]:
    """Measure the post-W_o/pre-residual tensor without rewriting attempts."""

    output = Path(output_path)
    if output.exists():
        artifact = json.loads(output.read_text(encoding="utf-8"))
        if artifact.get("status") != "verified" or len(artifact.get("conditions", [])) != 10:
            raise ValueError("Existing attention-output diagnostic is incomplete.")
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
        manifest = json.loads((attempt_dir / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            raise ValueError(f"Attempt {attempt_id} is not terminally completed.")
        if manifest["condition"]["id"] != source["condition"]["id"]:
            raise ValueError(f"Condition identity mismatch for {attempt_id}.")

        checkpoint_dir = attempt_dir / manifest["checkpoint"]["path"]
        checkpoint_hash = inventory_content_sha256(build_transfer_inventory(checkpoint_dir))
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
        dropout_probabilities = [
            float(layer.post_attention_dropout.p) for layer in model.gpt_neox.layers
        ]
        if dropout_probabilities != [0.0] * 6:
            raise ValueError(f"Attention dropout is not zero for {attempt_id}.")

        accumulator = ActivationAccumulator((0.0, EPSILON, 0.01))
        equality_checks = 0
        maximum_absolute_difference = 0.0
        torch.cuda.synchronize()
        evaluation_started = perf_counter()
        with AttentionOutputCapture(model) as capture:

            def observe(_output: Any, _batch_sequences: int) -> None:
                nonlocal equality_checks, maximum_absolute_difference
                if set(capture.after_wo) != set(capture.before_residual):
                    raise ValueError("W_o and residual-boundary captures differ by layer.")
                for name in sorted(capture.after_wo):
                    after_wo = capture.after_wo[name].detach()
                    before_residual = capture.before_residual[name].detach()
                    if after_wo.shape != before_residual.shape:
                        raise ValueError(f"Shape changed before residual addition at {name}.")
                    difference = float((after_wo.float() - before_residual.float()).abs().max().cpu())
                    maximum_absolute_difference = max(maximum_absolute_difference, difference)
                    equality_checks += 1
                accumulator.update(capture.before_residual, torch=torch)
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
        if equality_checks != int(coverage["batches"]) * 6 or maximum_absolute_difference != 0.0:
            raise ValueError(f"Post-W_o and pre-residual tensors differ for {attempt_id}.")
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
        _validate_layer_pool(statistics)
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
                "boundary_equivalence": {
                    "attention_dropout_probabilities": dropout_probabilities,
                    "comparisons": equality_checks,
                    "maximum_absolute_difference": maximum_absolute_difference,
                    "exactly_equal": True,
                },
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
        "site": SITE,
        "site_definition": "output of attention.dense (W_o), equal under zero dropout to post_attention_dropout output immediately before residual addition",
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
            "08_plot_attention_output.py": _sha256(RUN_DIR / "08_plot_attention_output.py"),
            "attention_output_figure.py": _sha256(Path(__file__)),
        },
    }
    write_json(output, artifact)
    return artifact


def reduce_attention_output_points(
    verification: Mapping[str, Any],
    diagnostic: Mapping[str, Any],
    statistics_loader: Callable[[str], Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Reduce both axes from their pooled integer hits and denominators."""

    if verification.get("status") != "verified" or verification.get("evidence_label") != "valid":
        raise ValueError("Figure input must be valid terminal verification evidence.")
    if diagnostic.get("status") != "verified" or float(diagnostic["epsilon"]) != EPSILON:
        raise ValueError("Figure input requires the verified epsilon-1e-3 diagnostic.")
    by_attempt = {row["attempt_id"]: row for row in diagnostic["conditions"]}
    grouped: dict[str, list[dict[str, Any]]] = {"gelu": [], "relu": []}
    for source in verification["conditions"]:
        attempt_id = source["attempt_id"]
        condition = source["condition"]
        activation = condition["activation"]
        if activation not in grouped or attempt_id not in by_attempt:
            raise ValueError(f"Incomplete attention-output input for {attempt_id}.")
        original = statistics_loader(attempt_id)
        record = by_attempt[attempt_id]
        point = {
            "attempt_id": attempt_id,
            "condition_id": condition["id"],
            "activation": activation,
            "order": int(condition["order"]),
            "label": "control" if condition["is_control"] else f"lambda={condition['pressure_weight']:g}",
            "pressure_weight": float(condition["pressure_weight"]),
            "is_control": bool(condition["is_control"]),
            "h_near_zero": _pooled_counts(original, "h", EPSILON),
            "attention_output_near_zero": _pooled_counts(record["statistics"], SITE, EPSILON),
            "final_validation_loss": float(source["final_validation_loss"]),
        }
        grouped[activation].append(point)

    expected = ["control", "lambda=0.1", "lambda=0.5", "lambda=1", "lambda=5"]
    for activation, points in grouped.items():
        points.sort(key=lambda point: point["order"])
        if [point["label"] for point in points] != expected:
            raise ValueError(f"Incomplete or misordered {activation} series.")
    return grouped


def generate_attention_output_figure() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    verification = json.loads(VERIFICATION_PATH.read_text(encoding="utf-8"))
    diagnostic = compute_attention_output_diagnostics()

    def load_statistics(attempt_id: str) -> Mapping[str, Any]:
        record = next(row for row in diagnostic["conditions"] if row["attempt_id"] == attempt_id)
        path = RUN_DIR / record["original_activation_statistics"]["path"]
        if _sha256(path) != record["original_activation_statistics"]["sha256"]:
            raise ValueError(f"Original activation diagnostic changed for {attempt_id}.")
        return json.loads(path.read_text(encoding="utf-8"))

    grouped = reduce_attention_output_points(verification, diagnostic, load_statistics)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
        }
    )
    styles = {
        "gelu": {"display": "GeLU", "color": "#0072B2", "marker": "o", "linestyle": "-"},
        "relu": {"display": "ReLU", "color": "#D55E00", "marker": "s", "linestyle": "--"},
    }
    offsets = {
        "gelu": [(5, -11), (0, 19), (0, 7), (9, 19), (0, 7)],
        "relu": [(0, -11), (0, 7), (-2, -11), (-2, 7), (0, -11)],
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.subplots_adjust(left=0.14, right=0.97, top=0.86, bottom=0.22)
    for activation in ("gelu", "relu"):
        points = grouped[activation]
        style = styles[activation]
        ax.plot(
            [point["h_near_zero"]["percent"] for point in points],
            [point["attention_output_near_zero"]["percent"] for point in points],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.8,
            label=style["display"],
        )
        for index, point in enumerate(points):
            dx, dy = offsets[activation][index]
            ax.annotate(
                point["label"],
                (point["h_near_zero"]["percent"], point["attention_output_near_zero"]["percent"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 5 else "center",
                va="bottom" if dy > 0 else "top",
                fontsize=8.2,
                color="#202020",
            )

    all_points = grouped["gelu"] + grouped["relu"]
    x_max = max(point["h_near_zero"]["percent"] for point in all_points)
    y_max = max(point["attention_output_near_zero"]["percent"] for point in all_points)
    ax.set_xlim(0.0, max(1.0, x_max * 1.10))
    ax.set_ylim(0.0, max(0.01, y_max * 1.25))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.set_xlabel("Near-zero mass at h, |x| <= 1e-3 (%)", labelpad=8)
    ax.set_ylabel("Near-zero mass after W_o, before residual addition (%)", labelpad=8)
    ax.set_title("Pythia-14M L1N and attention-output near-zero mass", pad=12)
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.text(
        0.5,
        0.035,
        "Seed 0; 451 updates per condition; all 338 validation blocks; lines follow control to increasing lambda",
        ha="center",
        va="bottom",
        fontsize=8.2,
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
        "y_site": SITE,
        "y_site_definition": diagnostic["site_definition"],
        "estimand": "each coordinate is integer near-zero hits pooled across all validation batches and six layers, divided by its pooled element count",
        "source_verification": {
            "path": VERIFICATION_PATH.relative_to(RUN_DIR).as_posix(),
            "sha256": _sha256(VERIFICATION_PATH),
        },
        "source_diagnostic": {
            "path": DIAGNOSTIC_PATH.relative_to(RUN_DIR).as_posix(),
            "sha256": _sha256(DIAGNOSTIC_PATH),
        },
        "series": grouped,
        "axes": {"x_starts_at_zero": True, "y_starts_at_zero": True},
        "figure": {
            "path": FIGURE_PATH.relative_to(RUN_DIR).as_posix(),
            "bytes": FIGURE_PATH.stat().st_size,
            "sha256": _sha256(FIGURE_PATH),
        },
        "source_code": {
            "08_plot_attention_output.py": _sha256(RUN_DIR / "08_plot_attention_output.py"),
            "attention_output_figure.py": _sha256(Path(__file__)),
        },
    }
    write_json(FIGURE_DATA_PATH, figure_data)
    _write_observation(grouped, verification["evidence_label"])
    write_json(
        COMPLETION_PATH,
        {
            "schema_version": 1,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "figure": figure_data["figure"],
            "figure_data": FIGURE_DATA_PATH.relative_to(RUN_DIR).as_posix(),
            "observation": OBSERVATION_PATH.relative_to(RUN_DIR).as_posix(),
            "diagnostic": DIAGNOSTIC_PATH.relative_to(RUN_DIR).as_posix(),
        },
    )
    return FIGURE_PATH


def _write_observation(
    grouped: Mapping[str, list[Mapping[str, Any]]], evidence_label: str
) -> None:
    rows = []
    for activation in ("gelu", "relu"):
        for point in grouped[activation]:
            rows.append(
                "| {activation} | {label} | {h:.6f} | {output:.6f} |".format(
                    activation="GeLU" if activation == "gelu" else "ReLU",
                    label=point["label"],
                    h=point["h_near_zero"]["percent"],
                    output=point["attention_output_near_zero"]["percent"],
                )
            )
    text = f"""# O003 - L1N pressure and attention-output near-zero mass

## Question

How does near-zero mass at the attention-branch output immediately after the
output projection `W_o` and before residual addition vary with near-zero mass at
pressured `h`, and how do those trajectories differ for GeLU and ReLU?

## Method and coverage

The figure uses the ten matched final Run 002 checkpoints. Every coordinate is
measured over all 500 MiniPile validation documents, 338 complete 2,048-token
blocks, 692,224 input tokens, all six layers, and the declared 1,444-token
excluded tail. The threshold is `abs(x) <= 1e-3`. Integer hits and denominators
are pooled across validation batches and layers before division; there is no
condition or seed averaging.

The x-coordinate comes from the original terminal `h` diagnostic. The output
metric was not selected before launch, so it was measured post hoc from all ten
retained, hash-verified final checkpoints. Hooks capture both the output of each
`attention.dense` (`W_o`) module and the corresponding
`post_attention_dropout` output immediately before the parallel residual sum.
Attention dropout is zero in every layer, and all captured tensors were exactly
equal. Every post-hoc pass reproduced its stored final validation loss within
`1e-5`; six layer rows reconcile to each pooled output row.

## Values

| Activation | Pressure label | h near-zero (%) | Attention output near-zero (%) |
| --- | --- | ---: | ---: |
{chr(10).join(rows)}

## Figure caption and encoding

**Figure 03. Near-zero mass after attention output projection versus pressured
FFN `h` for local Pythia-14M L1N pretraining.** Blue circles/solid lines denote
GeLU; orange squares/dashed lines denote ReLU. Labels identify the no-pressure
control and naive-L1 weight. Lines connect control through increasing pressure
strength as visual guides, not fitted relationships. Both axes begin at zero
and report percent at epsilon `1e-3`.

## Interpretation and limits

This is a checkpoint-reconstructible post-hoc activation diagnostic, not a
training-time gradient measurement. The trajectories are descriptive results
from one seed and a short training horizon. GeLU-versus-ReLU differences include
the activation operator change. Near-zero activation mass is not an exact-zero
product count, removable compute, measured speedup, a causal route, or a
long-horizon optimum. The source attempt evidence is `{evidence_label}`.

## Provenance

- Terminal source: `../artifacts/verification.json`
- Original `h`: `../artifacts/attempts/*/diagnostics/activation_statistics.json`
- Post-hoc attention output: `../artifacts/posthoc-attention-output-activation-statistics.json`
- Numerical reduction: `../artifacts/figure_data_attention_output.json`
- Source script: `../08_plot_attention_output.py` and `../attention_output_figure.py`
- Output: `../figures/03-h-vs-attention-output-near-zero.pdf`
"""
    OBSERVATION_PATH.write_text(text, encoding="utf-8", newline="\n")
