from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / "analyses/011-2026-09-03-pythia14m-70m-410m-selected-ladder/figure_data.json"
RUN021 = ROOT / "runs/021-2026-09-03-pythia410m-a0-learning-rate-screen-resolved-lr/artifacts/selection.json"
OUT_DATA = HERE / "figure_data.json"
OUT_TABLES = HERE / "tables.md"
FIG_DIR = HERE / "figures"
TABLE_DIR = HERE / "tables"

SCALES = ("14M", "70M", "410M")
SCALE_LABELS = {"14M": "Pythia-14M", "70M": "Pythia-70M", "410M": "Pythia-410M"}
PARAMETERS = {"14M": 14_067_712, "70M": 70_426_624, "410M": 405_334_016}
TRAINING_TOKENS = 1_493_172_224
SITES = ("a", "m", "h", "z", "q_post", "k_post", "v")
SITE_LABELS = {"a": "a", "m": "m", "h": "h", "z": "z", "q_post": "q", "k_post": "k", "v": "v"}
OPS = (
    "qkv_projection",
    "qk_scores",
    "probability_value",
    "attention_output_projection",
    "mlp_w1",
    "mlp_w2",
)
OP_LABELS = {
    "qkv_projection": "QKV",
    "qk_scores": "QK",
    "probability_value": "PV",
    "attention_output_projection": "$W_o$",
    "mlp_w1": "$W_1$",
    "mlp_w2": "$W_2$",
}
FAMILY_STYLE = {
    "A0": ("#b2182b", "o"),
    "A1-H": ("#333333", "s"),
    "A4-OL1": ("#6a3d9a", "^"),
    "A7-OL1": ("#1f9ac0", "D"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def close(actual: float, expected: float, tolerance: float = 1e-10) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise AssertionError(f"{actual} != {expected}")


def source_path(row: dict[str, Any], suffix: str) -> tuple[Path, str]:
    matches = [(path, value) for path, value in row["source_files"].items() if path.endswith(suffix)]
    if len(matches) != 1:
        raise AssertionError(f"expected one {suffix} source for {row['condition_id']}")
    relative, expected_hash = matches[0]
    path = ROOT / relative
    if sha256(path) != expected_hash:
        raise AssertionError(f"source hash mismatch: {relative}")
    return path, expected_hash


def select(rows: list[dict[str, Any]], **fields: Any) -> dict[str, Any]:
    matches = [row for row in rows if all(row[key] == value for key, value in fields.items())]
    if len(matches) != 1:
        raise AssertionError(f"selection {fields} produced {len(matches)} rows")
    return matches[0]


def verify_analysis011(data: dict[str, Any]) -> None:
    if data["status"] != "complete_verified_analysis":
        raise AssertionError("Analysis 011 is not complete_verified_analysis")
    coverage = data["coverage"]
    expected = {
        "documents": 500,
        "sequences": 338,
        "input_tokens": 692_224,
        "excluded_tail_tokens": 1_444,
        "complete_block_coverage": True,
        "seed_count_per_scale": 1,
    }
    for key, value in expected.items():
        if coverage[key] != value:
            raise AssertionError(f"coverage mismatch for {key}")
    if len(data["trained_endpoints"]) != 30 or len(data["teal_points"]) != 60:
        raise AssertionError("incomplete selected-ladder grid")
    for scale in SCALES:
        for family in ("A4-OL1", "A7-OL1"):
            rows = [r for r in data["trained_endpoints"] if r["scale"] == scale and r["family"] == family]
            if [r["kappa"] for r in rows] != [0.0, 0.01, 0.05, 0.1, 0.5]:
                raise AssertionError(f"incomplete trained grid: {scale} {family}")
        for control in ("A0", "A1-H"):
            rows = [r for r in data["teal_points"] if r["scale"] == scale and r["control"] == control]
            if [r["target_sparsity"] for r in rows] != [i / 10 for i in range(10)]:
                raise AssertionError(f"incomplete TEAL grid: {scale} {control}")


def late_train_mean(data: dict[str, Any], scale: str) -> float:
    rows = [r for r in data["a0_gradient_norms"] if r["scale"] == scale and 649 <= r["step"] <= 712]
    if len(rows) != 64:
        raise AssertionError(f"expected 64 late A0 boundaries for {scale}")
    return sum(r["task_loss"] for r in rows) / len(rows)


def baseline_table(data: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for scale in SCALES:
        a0 = select(data["teal_points"], scale=scale, control="A0", target_sparsity=0.0)
        summary = select(data["a0_gradient_summaries"], scale=scale)
        rows.append(
            {
                "scale": scale,
                "parameters": PARAMETERS[scale],
                "training_tokens": TRAINING_TOKENS,
                "tokens_per_parameter": TRAINING_TOKENS / PARAMETERS[scale],
                "peak_lr": summary["peak_learning_rate"],
                "late_train_loss": late_train_mean(data, scale),
                "validation_loss": a0["validation_loss"],
                "clipped_boundaries": summary["clipped_boundaries"],
            }
        )
    return rows


def delta_rows(data: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for scale in SCALES:
        for family in ("A4-OL1", "A7-OL1"):
            rows = [r for r in data["trained_endpoints"] if r["scale"] == scale and r["family"] == family]
            base = select(rows, kappa=0.0)
            for row in rows:
                output.append(
                    {
                        "kind": "trained",
                        "scale": scale,
                        "family": family,
                        "dose": row["kappa"],
                        "delta_loss": row["validation_loss"] - base["validation_loss"],
                        "delta_R_model_pp": 100 * (row["R_model"] - base["R_model"]),
                    }
                )
        for control in ("A0", "A1-H"):
            rows = [r for r in data["teal_points"] if r["scale"] == scale and r["control"] == control]
            base = select(rows, target_sparsity=0.0)
            for row in rows:
                output.append(
                    {
                        "kind": "posthoc",
                        "scale": scale,
                        "family": control,
                        "dose": row["target_sparsity"],
                        "delta_loss": row["validation_loss"] - base["validation_loss"],
                        "delta_R_model_pp": 100 * (row["R_model"] - base["R_model"]),
                    }
                )
    return output


def endpoint_microstructure(data: dict[str, Any]) -> list[dict[str, Any]]:
    output = []
    for scale in SCALES:
        for family in ("A4-OL1", "A7-OL1"):
            for kappa in (0.0, 0.5):
                row = select(data["trained_endpoints"], scale=scale, family=family, kappa=kappa)
                activation_path, activation_hash = source_path(row, "diagnostics/activation_statistics.json")
                logical_path, logical_hash = source_path(row, "diagnostics/logical_products.json")
                activation = load_json(activation_path)
                logical = load_json(logical_path)
                pooled = {item["name"]: item for item in activation["pooled_by_site"]}
                site_zeros = {}
                for site in SITES:
                    item = pooled[site]
                    close(item["exact_zero_fraction"], item["exact_zero_count"] / item["total"])
                    site_zeros[site] = item["exact_zero_fraction"]
                measured = logical["measured"]
                close(measured["R_model"], measured["block_zero_product_count"] / measured["model_product_count"])
                close(measured["R_model"], row["R_model"])
                operation_contributions = {}
                for operation in OPS:
                    item = measured["per_operation"][operation]
                    close(item["zero_product_fraction"], item["zero_product_count"] / item["product_count"])
                    operation_contributions[operation] = item["zero_product_count"] / measured["model_product_count"]
                close(sum(operation_contributions.values()), row["R_model"])
                output.append(
                    {
                        "scale": scale,
                        "family": family,
                        "kappa": kappa,
                        "validation_loss": row["validation_loss"],
                        "R_model": row["R_model"],
                        "R_model_max": row["R_model_max"],
                        "site_exact_zero": site_zeros,
                        "operation_contributions": operation_contributions,
                        "sources": {
                            str(activation_path.relative_to(ROOT)).replace("\\", "/"): activation_hash,
                            str(logical_path.relative_to(ROOT)).replace("\\", "/"): logical_hash,
                        },
                    }
                )
    return output


def verify_run021() -> dict[str, Any]:
    selection = load_json(RUN021)
    selected = select(selection["arms"], condition_id=selection["selected_condition_id"])
    if selection["status"] != "selected" or selected["peak_learning_rate"] != 0.0003:
        raise AssertionError("Run 021 did not retain the 3e-4 baseline")
    return {
        "path": str(RUN021.relative_to(ROOT)).replace("\\", "/"),
        "sha256": sha256(RUN021),
        "status": selection["status"],
        "selected_peak_learning_rate": selected["peak_learning_rate"],
    }


def build_data() -> dict[str, Any]:
    data = load_json(SOURCE)
    verify_analysis011(data)
    result = {
        "schema_version": 1,
        "status": "complete_verified",
        "question": "Paper-facing cross-scale and microstructure synthesis through Pythia-410M",
        "coverage": data["coverage"],
        "source_analysis": {
            "path": str(SOURCE.relative_to(ROOT)).replace("\\", "/"),
            "sha256": sha256(SOURCE),
        },
        "run021_selection": verify_run021(),
        "baseline_exposure": baseline_table(data),
        "trained_endpoints": data["trained_endpoints"],
        "teal_points": data["teal_points"],
        "within_family_deltas": delta_rows(data),
        "endpoint_microstructure": endpoint_microstructure(data),
        "interpretation": {
            "r_model": "logical zero-product opportunity, not removed FLOPs or measured speedup",
            "scale": "one seed per scale under equal total tokens, not a scaling law",
            "recipe_contrast": "A4-OL1 versus A7-OL1 changes both gate topology and pressure sites",
        },
    }
    return result


def style_axis(axis: Any) -> None:
    axis.grid(True, color="#dddddd", linewidth=0.55, alpha=0.75)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def add_upper_exit(axis: Any, xs: list[float], ys: list[float], color: str, limit: float) -> None:
    for index in range(1, len(xs)):
        if ys[index - 1] <= limit < ys[index]:
            fraction = (limit - ys[index - 1]) / (ys[index] - ys[index - 1])
            crossing = xs[index - 1] + fraction * (xs[index] - xs[index - 1])
            axis.scatter([crossing], [limit - 0.015], marker="^", s=25, color=color, clip_on=True, zorder=5)


def render_absolute(data: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.35), sharex=True, sharey=True)
    for axis, scale in zip(axes, SCALES, strict=True):
        for family in ("A0", "A1-H"):
            rows = [r for r in data["teal_points"] if r["scale"] == scale and r["control"] == family]
            color, marker = FAMILY_STYLE[family]
            xs = [100 * r["R_model"] for r in rows]
            ys = [r["validation_loss"] for r in rows]
            axis.plot(xs, ys, color=color, marker=marker, markersize=3.2, linewidth=1.0, label=f"{family} post-hoc")
            add_upper_exit(axis, xs, ys, color, 6.15)
        for family in ("A4-OL1", "A7-OL1"):
            rows = [r for r in data["trained_endpoints"] if r["scale"] == scale and r["family"] == family]
            color, marker = FAMILY_STYLE[family]
            axis.plot([100 * r["R_model"] for r in rows], [r["validation_loss"] for r in rows], color=color, marker=marker, markersize=4.0, linewidth=1.35, label=f"{family} trained")
            for row in rows:
                if row["kappa"] in (0.0, 0.1, 0.5) and row["validation_loss"] <= 6.15:
                    axis.annotate(f"{row['kappa']:g}", (100 * row["R_model"], row["validation_loss"]), xytext=(2, 3), textcoords="offset points", fontsize=6.2, color=color)
        exposure = next(r for r in data["baseline_exposure"] if r["scale"] == scale)
        axis.set_title(f"{SCALE_LABELS[scale]}\n{exposure['tokens_per_parameter']:.1f} tokens/parameter", fontsize=9)
        axis.set_xlim(-2, 90)
        axis.set_ylim(3.85, 6.15)
        axis.set_xlabel(r"$R_{model}$ (\%)")
        style_axis(axis)
    axes[0].set_ylabel("Validation loss (lower is better)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=7.3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def render_deltas(data: dict[str, Any], output: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 5.35), sharex="row", sharey="row")
    for column, scale in enumerate(SCALES):
        top = axes[0, column]
        bottom = axes[1, column]
        for family in ("A4-OL1", "A7-OL1"):
            rows = [r for r in data["within_family_deltas"] if r["kind"] == "trained" and r["scale"] == scale and r["family"] == family]
            color, marker = FAMILY_STYLE[family]
            top.plot([r["delta_R_model_pp"] for r in rows], [r["delta_loss"] for r in rows], color=color, marker=marker, markersize=4, linewidth=1.25, label=family)
            for row in rows:
                if row["dose"] in (0.1, 0.5):
                    top.annotate(f"{row['dose']:g}", (row["delta_R_model_pp"], row["delta_loss"]), xytext=(2, 3), textcoords="offset points", fontsize=6.2, color=color)
        for family in ("A0", "A1-H"):
            rows = [r for r in data["within_family_deltas"] if r["kind"] == "posthoc" and r["scale"] == scale and r["family"] == family]
            color, marker = FAMILY_STYLE[family]
            bottom.plot([r["delta_R_model_pp"] for r in rows], [r["delta_loss"] for r in rows], color=color, marker=marker, markersize=3.2, linewidth=1.0, label=family)
        top.axhline(0, color="#777777", linewidth=0.65)
        bottom.axhline(0, color="#777777", linewidth=0.65)
        top.set_title(SCALE_LABELS[scale], fontsize=9)
        top.set_ylim(-0.62, 0.68)
        bottom.set_ylim(-0.15, 5.25)
        bottom.set_xlabel(r"$\Delta R_{model}$ (percentage points)")
        style_axis(top)
        style_axis(bottom)
    axes[0, 0].set_ylabel("Trained " + r"$\Delta$" + " loss\nrelative to " + r"$\kappa=0$")
    axes[1, 0].set_ylabel("Post-hoc " + r"$\Delta$" + " loss\nrelative to $p=0$")
    h1, l1 = axes[0, 0].get_legend_handles_labels()
    h2, l2 = axes[1, 0].get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + [f"{label} post-hoc" for label in l2], loc="upper center", ncol=4, frameon=False, fontsize=7.3, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def render_site_heatmap(data: dict[str, Any], output: Path) -> None:
    columns = (("A4-OL1", 0.0), ("A4-OL1", 0.5), ("A7-OL1", 0.0), ("A7-OL1", 0.5))
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.7), sharey=True)
    image = None
    for axis, scale in zip(axes, SCALES, strict=True):
        matrix = np.array([[100 * select(data["endpoint_microstructure"], scale=scale, family=family, kappa=kappa)["site_exact_zero"][site] for family, kappa in columns] for site in SITES])
        image = axis.imshow(matrix, vmin=0, vmax=100, cmap="viridis", aspect="auto")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                text_color = "white" if value < 38 or value > 82 else "black"
                axis.text(j, i, "<.01" if value < 0.01 else f"{value:.1f}", ha="center", va="center", fontsize=6.6, color=text_color)
        axis.set_title(SCALE_LABELS[scale], fontsize=9)
        axis.set_xticks(range(4), ["A4\n0", "A4\n.5", "A7\n0", "A7\n.5"], fontsize=7)
        axis.set_yticks(range(len(SITES)), [SITE_LABELS[s] for s in SITES], fontsize=8)
        axis.set_xlabel(r"Recipe and $\kappa$")
    axes[0].set_ylabel("Activation site")
    if image is None:
        raise AssertionError("heatmap not rendered")
    fig.subplots_adjust(left=0.08, right=0.88, bottom=0.17, top=0.9, wspace=0.18)
    color_axis = fig.add_axes((0.9, 0.2, 0.015, 0.64))
    colorbar = fig.colorbar(image, cax=color_axis)
    colorbar.set_label(r"Exact-zero mass (\%)", fontsize=8)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def render_operation_contributions(data: dict[str, Any], output: Path) -> None:
    columns = (("A4-OL1", 0.0), ("A4-OL1", 0.5), ("A7-OL1", 0.0), ("A7-OL1", 0.5))
    colors = ("#4c78a8", "#f58518", "#e45756", "#72b7b2", "#54a24b", "#b279a2")
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), sharey=True)
    x = np.arange(len(columns))
    for axis, scale in zip(axes, SCALES, strict=True):
        bottoms = np.zeros(len(columns))
        for operation, color in zip(OPS, colors, strict=True):
            values = np.array([100 * select(data["endpoint_microstructure"], scale=scale, family=family, kappa=kappa)["operation_contributions"][operation] for family, kappa in columns])
            axis.bar(x, values, bottom=bottoms, color=color, width=0.72, label=OP_LABELS[operation])
            bottoms += values
        for index, total in enumerate(bottoms):
            axis.text(index, total + 1.1, f"{total:.1f}", ha="center", va="bottom", fontsize=6.8)
        axis.set_title(SCALE_LABELS[scale], fontsize=9)
        axis.set_xticks(x, ["A4\n0", "A4\n.5", "A7\n0", "A7\n.5"], fontsize=7)
        axis.set_xlabel(r"Recipe and $\kappa$")
        axis.set_ylim(0, 88)
        style_axis(axis)
    axes[0].set_ylabel("Contribution to $R_{model}$ (percentage points)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, frameon=False, fontsize=7.3, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)


def table_outputs(data: dict[str, Any]) -> None:
    baseline_lines = [
        "# Paper-facing tables",
        "",
        "All losses are paired with the eager logical-product pass unless the column explicitly names training loss.",
        "",
        "## Baseline exposure and optimization context",
        "",
        "| Model | Parameters | Training tokens | Tokens / parameter | Peak LR | Mean A0 train loss, steps 649--712 | A0 validation loss | Clipped A0 boundaries |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in data["baseline_exposure"]:
        baseline_lines.append(f"| Pythia-{row['scale']} | {row['parameters']:,} | {row['training_tokens']:,} | {row['tokens_per_parameter']:.3f} | {row['peak_lr']:.0e} | {row['late_train_loss']:.6f} | {row['validation_loss']:.6f} | {row['clipped_boundaries']} / 712 |")
    baseline_lines += [
        "",
        "## Matched trained endpoint boundary",
        "",
        "| Model | A4-OL1 k=0 loss / R_model | A4-OL1 k=.5 loss / R_model | A7-OL1 k=0 loss / R_model | A7-OL1 k=.5 loss / R_model |",
        "|---|---:|---:|---:|---:|",
    ]
    for scale in SCALES:
        cells = []
        for family, kappa in (("A4-OL1", 0.0), ("A4-OL1", 0.5), ("A7-OL1", 0.0), ("A7-OL1", 0.5)):
            row = select(data["trained_endpoints"], scale=scale, family=family, kappa=kappa)
            cells.append(f"{row['validation_loss']:.6f} / {100 * row['R_model']:.4f}%")
        baseline_lines.append(f"| Pythia-{scale} | " + " | ".join(cells) + " |")
    baseline_lines += [
        "",
        "The A4-OL1/A7-OL1 columns compare complete recipes: A7 adds post-RoPE q/k/v gates and pressures those sites. They do not isolate gate placement.",
    ]
    write_text(OUT_TABLES, "\n".join(baseline_lines) + "\n")

    tex = [
        "% Generated by analyses/012-2026-09-04-paper-synthesis/01_build.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Matched one-pass exposure and A0 optimization context. Every model sees the same total tokens, not the same tokens per parameter. Validation loss is paired with the eager logical-product pass.}",
        "\\label{tab:baseline-exposure}",
        "\\small",
        "\\begin{tabular}{lrrrrrr}",
        "\\toprule",
        "Model & Params & Tok./param. & Peak LR & Late train & Val. & Clip \\\\",
        "\\midrule",
    ]
    for row in data["baseline_exposure"]:
        tex.append(f"{row['scale']} & {row['parameters']/1e6:.1f}M & {row['tokens_per_parameter']:.1f} & {row['peak_lr']:.0e} & {row['late_train_loss']:.3f} & {row['validation_loss']:.3f} & {row['clipped_boundaries']}/712 \\\\")
    tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write_text(TABLE_DIR / "baseline-exposure.tex", "\n".join(tex) + "\n")

    endpoint_tex = [
        "% Generated by analyses/012-2026-09-04-paper-synthesis/01_build.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Low- and high-dose trained endpoints. Each cell is validation loss / $R_{model}$ (\\%). A4-OL1 and A7-OL1 are complete recipes, not an isolated topology contrast.}",
        "\\label{tab:cross-scale-endpoints}",
        "\\small",
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "& \\multicolumn{2}{c}{A4-OL1} & \\multicolumn{2}{c}{A7-OL1} \\\\",
        "\\cmidrule(lr){2-3}\\cmidrule(l){4-5}",
        "Model & $\\kappa=0$ & $\\kappa=.5$ & $\\kappa=0$ & $\\kappa=.5$ \\\\",
        "\\midrule",
    ]
    for scale in SCALES:
        cells = []
        for family, kappa in (("A4-OL1", 0.0), ("A4-OL1", 0.5), ("A7-OL1", 0.0), ("A7-OL1", 0.5)):
            row = select(data["trained_endpoints"], scale=scale, family=family, kappa=kappa)
            cells.append(f"{row['validation_loss']:.3f} / {100 * row['R_model']:.1f}")
        endpoint_tex.append(scale + " & " + " & ".join(cells) + " \\\\")
    endpoint_tex += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    write_text(TABLE_DIR / "cross-scale-endpoints.tex", "\n".join(endpoint_tex) + "\n")


def main() -> None:
    data = build_data()
    write_text(OUT_DATA, json.dumps(data, indent=2, sort_keys=True) + "\n")
    table_outputs(data)
    render_absolute(data, FIG_DIR / "01-absolute-frontiers-by-scale.pdf")
    render_deltas(data, FIG_DIR / "02-within-scale-deltas.pdf")
    render_site_heatmap(data, FIG_DIR / "03-sitewise-zero-structure.pdf")
    render_operation_contributions(data, FIG_DIR / "04-operation-contributions.pdf")


if __name__ == "__main__":
    main()
