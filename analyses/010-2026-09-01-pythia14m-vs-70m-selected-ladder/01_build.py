"""Build the matched Pythia-14M versus 70M selected-ladder comparison."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from statistics import correlation
from typing import Any, Iterable


ANALYSIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = ANALYSIS_DIR.parent.parent
RUN_014 = REPO_ROOT / "runs" / "014-2026-08-31-pythia14m-full-pass-a7-ol1"
RUN_015 = REPO_ROOT / "runs" / "015-2026-08-31-pythia14m-corrected-a4-ol1"
RUN_018 = REPO_ROOT / "runs" / "018-2026-09-01-pythia70m-selected-ladder-canonical-init"
TEAL_14 = (
    REPO_ROOT
    / "analyses"
    / "005-2026-08-30-run004-controls-teal-posthoc"
    / "teal_frontier.json"
)
TEAL_70 = RUN_018 / "artifacts" / "teal" / "teal_frontiers.json"
FIGURE_DATA = ANALYSIS_DIR / "figure_data.json"
TABLES = ANALYSIS_DIR / "tables.md"
FIGURE = ANALYSIS_DIR / "figures" / "01-pythia14m-vs-70m-selected-ladder.pdf"

KAPPAS = (0.0, 0.01, 0.05, 0.1, 0.5)
TARGETS = tuple(index / 10 for index in range(10))
SITES = ("a", "m", "h", "q_post", "k_post", "v", "z", "attention_output")
COVERAGE = {
    "sequences": 338,
    "input_tokens": 692_224,
    "excluded_tail_tokens": 1_444,
    "complete_block_coverage": True,
}
TRAINING_STEPS = 712
TRAINING_TOKENS = 1_493_172_224


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _repo_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _close(actual: float, expected: float, tolerance: float = 1e-12) -> None:
    if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(f"Numeric mismatch: {actual!r} != {expected!r}")


def _require_coverage(row: dict[str, Any], source: str) -> None:
    for key, expected in COVERAGE.items():
        if row.get(key) != expected:
            raise ValueError(f"Coverage mismatch for {source}/{key}")


def _event_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _matches_family(condition_id: str, scale: str, family: str) -> bool:
    if scale == "14M" and family == "A4-OL1":
        return condition_id.startswith("a4z-ol1-kappa-")
    if scale == "14M" and family == "A7-OL1":
        return condition_id.startswith("a7-ol1-kappa-")
    if scale == "70M" and family == "A4-OL1":
        return condition_id.startswith("a4-ol1-kappa-")
    if scale == "70M" and family == "A7-OL1":
        return condition_id.startswith("a7-ol1-kappa-")
    return False


def _trained_rows(run_dir: Path, scale: str, family: str) -> list[dict[str, Any]]:
    rows = []
    for attempt in sorted((run_dir / "artifacts" / "attempts").iterdir()):
        if not attempt.is_dir():
            continue
        metrics_path = attempt / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = _read_json(metrics_path)
        condition = metrics["condition"]
        condition_id = condition["id"]
        if not _matches_family(condition_id, scale, family):
            continue

        manifest_path = attempt / "manifest.json"
        logical_path = attempt / "diagnostics" / "logical_products.json"
        activation_path = attempt / "diagnostics" / "activation_statistics.json"
        events_path = attempt / "events.jsonl"
        manifest = _read_json(manifest_path)
        logical = _read_json(logical_path)
        activation = _read_json(activation_path)
        events = _event_rows(events_path)
        training = [event for event in events if event.get("event") == "train"]

        if manifest.get("status") != "completed" or manifest["condition"]["id"] != condition_id:
            raise ValueError(f"Incomplete or mismatched manifest: {attempt.name}")
        if metrics["training"]["completed_steps"] != TRAINING_STEPS:
            raise ValueError(f"Incomplete training steps: {attempt.name}")
        if metrics["training"]["input_tokens"] != TRAINING_TOKENS:
            raise ValueError(f"Training-token mismatch: {attempt.name}")
        if len(training) != TRAINING_STEPS:
            raise ValueError(f"Incomplete boundary log: {attempt.name}")
        if any(event["gradient_overflow"] or event["optimizer_step_skipped"] for event in training):
            raise ValueError(f"Overflow or skipped update: {attempt.name}")
        if any(not math.isfinite(float(event["task_loss"])) for event in training):
            raise ValueError(f"Non-finite task loss: {attempt.name}")

        _require_coverage(metrics["validation"]["final"], f"{attempt.name}/final")
        _require_coverage(logical["coverage"], f"{attempt.name}/logical")
        measured = logical["measured"]
        zero_count = int(measured["block_zero_product_count"])
        product_count = int(measured["model_product_count"])
        r_model = float(measured["R_model"])
        _close(zero_count / product_count, r_model, tolerance=1e-16)
        if sum(int(value["zero_product_count"]) for value in measured["per_operation"].values()) != int(
            measured["block_zero_product_count"]
        ):
            raise ValueError(f"Logical zero counts do not reconcile: {attempt.name}")

        pooled = {row["name"]: row for row in activation["pooled_by_site"]}
        if set(pooled) != set(SITES):
            raise ValueError(f"Per-site activation rows are incomplete: {attempt.name}")
        site_exact_zero = {}
        for site in SITES:
            site_row = pooled[site]
            count = int(site_row["exact_zero_count"])
            total = int(site_row["total"])
            fraction = float(site_row["exact_zero_fraction"])
            _close(count / total, fraction, tolerance=1e-15)
            site_exact_zero[site] = {
                "exact_zero_count": count,
                "total_count": total,
                "exact_zero_fraction": fraction,
            }

        rows.append(
            {
                "scale": scale,
                "family": family,
                "condition_id": condition_id,
                "attempt_id": attempt.name,
                "kappa": float(condition["gate_threshold"]),
                "validation_loss": float(metrics["validation"]["final"]["loss"]),
                "R_model": r_model,
                "R_model_max": float(logical["architecture_maximum"]["R_model_max_fraction"]),
                "logical_counts": {
                    "zero_product_count": zero_count,
                    "model_product_count": product_count,
                },
                "site_exact_zero": site_exact_zero,
                "conflict_steps": sum(bool(event["gradient_conflict"]) for event in training),
                "projection_steps": sum(bool(event["projection_applied"]) for event in training),
                "median_tokens_per_second": float(metrics["training"]["median_tokens_per_second"]),
                "source_files": {
                    _repo_path(path): _sha256(path)
                    for path in (manifest_path, metrics_path, logical_path, activation_path, events_path)
                },
            }
        )

    rows.sort(key=lambda row: row["kappa"])
    if tuple(row["kappa"] for row in rows) != KAPPAS:
        raise ValueError(f"Incomplete {scale} {family} kappa grid")
    return rows


def _teal_rows(path: Path, scale: str) -> list[dict[str, Any]]:
    payload = _read_json(path)
    raw_points = payload["conditions"] if scale == "14M" else payload["points"]
    id_map = {
        "14M": {"gelu-control": "A0", "relu-control": "A1-H"},
        "70M": {"a0-gelu": "A0", "a1h-relu": "A1-H"},
    }[scale]
    rows = []
    for point in raw_points:
        condition_id = point["condition_id"]
        if condition_id not in id_map:
            continue
        _require_coverage(point["validation"], f"{scale}/{condition_id}/{point['target_sparsity']}")
        logical = point["logical_products"]
        zero_count = int(logical["block_zero_product_count"])
        product_count = int(logical["model_product_count"])
        r_model = float(logical["R_model"])
        _close(zero_count / product_count, r_model, tolerance=1e-16)
        rows.append(
            {
                "scale": scale,
                "control": id_map[condition_id],
                "condition_id": condition_id,
                "target_sparsity": float(point["target_sparsity"]),
                "validation_loss": float(point["validation"]["loss"]),
                "R_model": r_model,
                "logical_counts": {
                    "zero_product_count": zero_count,
                    "model_product_count": product_count,
                },
            }
        )
    rows.sort(key=lambda row: (row["control"], row["target_sparsity"]))
    for control in ("A0", "A1-H"):
        grid = tuple(row["target_sparsity"] for row in rows if row["control"] == control)
        if grid != TARGETS:
            raise ValueError(f"Incomplete {scale} {control} TEAL grid")
    return rows


def _ranks(values: Iterable[float]) -> list[float]:
    values = list(values)
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    for rank, index in enumerate(order, start=1):
        ranks[index] = float(rank)
    return ranks


def _series_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first, last = rows[0], rows[-1]
    r_values = [row["R_model"] for row in rows]
    losses = [row["validation_loss"] for row in rows]
    return {
        "scale": first["scale"],
        "family": first["family"],
        "delta_R_model_percentage_points_kappa_0_to_0p5": 100.0 * (last["R_model"] - first["R_model"]),
        "delta_validation_loss_kappa_0_to_0p5": last["validation_loss"] - first["validation_loss"],
        "pearson_R_model_vs_loss": correlation(r_values, losses),
        "spearman_R_model_vs_loss": correlation(_ranks(r_values), _ranks(losses)),
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = left["validation_loss"] <= right["validation_loss"] and left["R_model"] >= right["R_model"]
    strict = left["validation_loss"] < right["validation_loss"] or left["R_model"] > right["R_model"]
    return no_worse and strict


def build_figure_data() -> dict[str, Any]:
    series = []
    for run_dir, scale, family in (
        (RUN_015, "14M", "A4-OL1"),
        (RUN_014, "14M", "A7-OL1"),
        (RUN_018, "70M", "A4-OL1"),
        (RUN_018, "70M", "A7-OL1"),
    ):
        series.extend(_trained_rows(run_dir, scale, family))
    teal = _teal_rows(TEAL_14, "14M") + _teal_rows(TEAL_70, "70M")

    grouped = {
        (scale, family): [row for row in series if row["scale"] == scale and row["family"] == family]
        for scale in ("14M", "70M")
        for family in ("A4-OL1", "A7-OL1")
    }
    summaries = [_series_summary(grouped[key]) for key in sorted(grouped)]
    persistence = {}
    for scale in ("14M", "70M"):
        a4 = grouped[(scale, "A4-OL1")]
        a7 = grouped[(scale, "A7-OL1")]
        persistence[scale] = {
            "A4_dominates_A7_at_kappa_0": _dominates(a4[0], a7[0]),
            "A7_dominates_A4_at_kappa_0p5": _dominates(a7[-1], a4[-1]),
            "A7_kappa_0p1_delta_R_model_percentage_points_from_kappa_0": 100.0
            * (a7[3]["R_model"] - a7[0]["R_model"]),
            "A7_kappa_0p1_delta_validation_loss_from_kappa_0": a7[3]["validation_loss"]
            - a7[0]["validation_loss"],
        }

    return {
        "schema_version": 1,
        "status": "complete_verified_analysis",
        "question": "Does the selected R_model versus validation-loss tradeoff persist from Pythia-14M to 70M?",
        "coverage": {"documents": 500, **COVERAGE, "seed_count": 1},
        "trained_endpoints": sorted(series, key=lambda row: (row["scale"], row["family"], row["kappa"])),
        "teal_points": sorted(teal, key=lambda row: (row["scale"], row["control"], row["target_sparsity"])),
        "series_summaries": summaries,
        "persistence_checks": persistence,
        "sources": {
            _repo_path(path): _sha256(path)
            for path in (
                RUN_014 / "artifacts" / "verification.json",
                RUN_015 / "artifacts" / "verification.json",
                RUN_018 / "artifacts" / "verification.json",
                TEAL_14,
                TEAL_70,
            )
        },
        "interpretation": {
            "R_model": "logical zero-product opportunity, not runtime speedup",
            "comparison": "descriptive one-seed scale persistence, not a scaling law",
            "lines": "dose-order guides, not fitted response curves",
        },
    }


def table_markdown(data: dict[str, Any]) -> str:
    lines = [
        "# Pythia-14M versus 70M selected ladder",
        "",
        "## Trained OL1 endpoints",
        "",
        "| Scale | Family | kappa | Validation loss | R_model (%) | a zero (%) | m zero (%) | h zero (%) | q_post zero (%) | k_post zero (%) | v zero (%) | z zero (%) | Conflicts | Projections |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["trained_endpoints"]:
        zeros = row["site_exact_zero"]
        lines.append(
            f"| {row['scale']} | {row['family']} | {row['kappa']:g} | {row['validation_loss']:.6f} | "
            f"{100 * row['R_model']:.4f} | {100 * zeros['a']['exact_zero_fraction']:.4f} | "
            f"{100 * zeros['m']['exact_zero_fraction']:.4f} | {100 * zeros['h']['exact_zero_fraction']:.4f} | "
            f"{100 * zeros['q_post']['exact_zero_fraction']:.4f} | "
            f"{100 * zeros['k_post']['exact_zero_fraction']:.4f} | {100 * zeros['v']['exact_zero_fraction']:.4f} | "
            f"{100 * zeros['z']['exact_zero_fraction']:.4f} | {row['conflict_steps']} | {row['projection_steps']} |"
        )
    lines.extend(
        [
            "",
            "## Endpoint-span summaries",
            "",
            "| Scale | Family | Delta R_model, kappa 0 to 0.5 (pp) | Delta loss | Pearson r | Spearman rho |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in data["series_summaries"]:
        lines.append(
            f"| {row['scale']} | {row['family']} | "
            f"{row['delta_R_model_percentage_points_kappa_0_to_0p5']:.4f} | "
            f"{row['delta_validation_loss_kappa_0_to_0p5']:+.6f} | "
            f"{row['pearson_R_model_vs_loss']:.4f} | {row['spearman_R_model_vs_loss']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Selected post-hoc TEAL points",
            "",
            "| Scale | Control | Target sparsity | Validation loss | R_model (%) |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in data["teal_points"]:
        if row["target_sparsity"] in {0.0, 0.1, 0.3, 0.5}:
            lines.append(
                f"| {row['scale']} | {row['control']} | {row['target_sparsity']:.1f} | "
                f"{row['validation_loss']:.6f} | {100 * row['R_model']:.4f} |"
            )
    lines.extend(
        [
            "",
            "All fractions are recomputed from pooled integer counts over the complete validation workload.",
            "R_model is a logical-product opportunity, not measured speedup.",
            "",
        ]
    )
    return "\n".join(lines)


def render_figure(data: dict[str, Any]) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "pdf.compression": 9,
        }
    )
    colors = {"A4-OL1": "#6F4C9B", "A7-OL1": "#0072B2", "A0": "#777777", "A1-H": "#D55E00"}
    scale_style = {"14M": ("--", "o"), "70M": ("-", "s")}
    label_offsets = {
        ("14M", "A4-OL1", 0.0): (4, -12),
        ("14M", "A7-OL1", 0.0): (-5, 9),
        ("14M", "A4-OL1", 0.1): (4, 4),
        ("14M", "A7-OL1", 0.1): (4, -12),
        ("70M", "A4-OL1", 0.1): (4, -12),
        ("70M", "A7-OL1", 0.1): (4, 5),
    }
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)

    control_endpoints: list[dict[str, Any]] = []
    for scale in ("14M", "70M"):
        for control in ("A0", "A1-H"):
            rows = [
                row
                for row in data["teal_points"]
                if row["scale"] == scale and row["control"] == control
            ]
            linestyle, marker = scale_style[scale]
            ax.plot(
                [100 * row["R_model"] for row in rows],
                [row["validation_loss"] for row in rows],
                color=colors[control],
                linestyle=linestyle,
                marker=marker,
                linewidth=1.55,
                markersize=4.5,
                alpha=0.95,
                zorder=2,
            )
            endpoint = rows[0]
            control_endpoints.append(endpoint)
            ax.scatter(
                [100 * endpoint["R_model"]],
                [endpoint["validation_loss"]],
                color=colors[control],
                marker=marker,
                s=58,
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
            )

    for scale in ("14M", "70M"):
        for family in ("A4-OL1", "A7-OL1"):
            rows = [row for row in data["trained_endpoints"] if row["scale"] == scale and row["family"] == family]
            linestyle, marker = scale_style[scale]
            ax.plot(
                [100 * row["R_model"] for row in rows],
                [row["validation_loss"] for row in rows],
                color=colors[family],
                linestyle=linestyle,
                marker=marker,
                linewidth=1.7,
                markersize=5.2,
                zorder=3,
            )
            for row in rows:
                if row["kappa"] in {0.0, 0.1, 0.5}:
                    offset = label_offsets.get((scale, family, row["kappa"]), (3, 3))
                    ax.annotate(
                        f"k={row['kappa']:g}",
                        (100 * row["R_model"], row["validation_loss"]),
                        xytext=offset,
                        textcoords="offset points",
                        fontsize=7.2,
                        color=colors[family],
                        horizontalalignment="right" if offset[0] < 0 else "left",
                    )

    path_handles = [
        Line2D([0], [0], color=colors[name], linewidth=2, label=label)
        for name, label in (
            ("A0", "A0 + post-hoc TEAL"),
            ("A1-H", "A1-H + post-hoc TEAL"),
            ("A4-OL1", "A4-OL1 trained ladder"),
            ("A7-OL1", "A7-OL1 trained ladder"),
        )
    ]
    path_legend = ax.legend(
        handles=path_handles,
        title="Path",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
    )
    ax.add_artist(path_legend)
    scale_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=scale_style[scale][0],
            marker=scale_style[scale][1],
            linewidth=1.6,
            markersize=5,
            label=scale,
        )
        for scale in ("14M", "70M")
    ]
    ax.legend(
        handles=scale_handles,
        title="Model size",
        frameon=False,
        loc="upper left",
        bbox_to_anchor=(1.01, 0.72),
    )

    endpoint_lines = ["Final control checkpoints", "(TEAL target 0; black edge)"]
    for endpoint in control_endpoints:
        r_percent = 100 * endpoint["R_model"]
        r_text = f"{r_percent:.6f}" if r_percent < 0.001 else f"{r_percent:.4f}"
        endpoint_lines.append(
            f"{endpoint['scale']} {endpoint['control']}: "
            f"R={r_text}%, loss={endpoint['validation_loss']:.4f}"
        )
    endpoint_lines.extend(["", "TEAL markers: targets 0.0 to 0.9", "OL1 labels: selected kappa values"])
    ax.text(
        1.02,
        0.48,
        "\n".join(endpoint_lines),
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7.8,
        linespacing=1.35,
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "#F6F6F6", "edgecolor": "#BBBBBB"},
    )

    ax.set_title("Pythia-14M and 70M: validation loss vs. measured R_model")
    ax.set_xlabel("Measured R_model (%)")
    ax.set_ylabel("Complete-validation loss")
    ax.set_xlim(-1.0, 42.0)
    ax.set_ylim(4.0, 9.35)
    ax.grid(alpha=0.22, linewidth=0.6)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        FIGURE,
        format="pdf",
        bbox_inches="tight",
        metadata={"Creator": "Analysis 010", "CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main() -> None:
    data = build_figure_data()
    _write_json(FIGURE_DATA, data)
    _write_text(TABLES, table_markdown(data))
    render_figure(data)
    print(f"wrote {len(data['trained_endpoints'])} trained endpoints and {len(data['teal_points'])} TEAL points")


if __name__ == "__main__":
    main()
