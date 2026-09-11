"""K050 sparsity/speedup and matched ablations for the 30-checkpoint cohort."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "analyses/018-2026-09-08-results-materials/figure_data.json"
RETROSPECTIVE = ROOT / "runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/results/matched-retrospective-001.json"
CANDIDATES = ("k050-no-skip", "k050-attention-dense", "k050")
ROW_LABELS = (("Fusion only", "(all sparse skips off)"),
              ("+ projection skipping", "(attention dense)"),
              ("+ attention skipping", "(full K050)"))
STYLE = {"Baseline / local": ("#687570", "o"),
         "4-site": ("#2878B5", "D"), "7-site": ("#C96024", "^")}


def geometric_mean(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def regression(points):
    """Unweighted OLS with intercept; x is in sparsity percentage points."""
    x = np.array([p["sparsity_percent"] for p in points])
    y = np.array([p["speedup"] for p in points])
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - intercept - slope * x) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"n": len(x), "slope_per_percentage_point": float(slope),
            "intercept": float(intercept), "r2": float(r2)}


def read_evidence():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    raw = json.loads(RETROSPECTIVE.read_text(encoding="utf-8"))
    raw_hash = hashlib.sha256(RETROSPECTIVE.read_bytes()).hexdigest()
    assert raw_hash == source["sources"][RETROSPECTIVE.relative_to(ROOT).as_posix()]
    raw_points = {(p["condition"], p["candidate"]): p for p in raw["points"]
                  if p["phase"] == "final"}
    trained = {p["id"]: p for p in source["trained"] if p["scale"] == "14M"}
    runtime = source["runtime"]
    conditions = {f"c{i:02d}" for i in range(1, 31)}
    points = []
    for candidate in CANDIDATES:
        selected = sorted((p for p in runtime["points"] if p["candidate"] == candidate),
                          key=lambda p: p["condition"])
        assert len(selected) == 30 and {p["condition"] for p in selected} == conditions
        for point in selected:
            assert point["phase"] == "final" and point["qualified"]
            assert len(point["replicates"]) == 3
            assert all(r["qualified"] and r["status"] == "complete" for r in point["replicates"])
            original = raw_points[point["condition"], candidate]
            for key in ("speedup", "canonical_counts", "replicates", "R_model"):
                assert point[key] == original[key]
            assert math.isclose(point["speedup"], geometric_mean(
                [r["speedup"] for r in point["replicates"]]), rel_tol=1e-12)
            counts = point["canonical_counts"]
            zeros = counts["block_zero_product_count"]
            assert zeros == sum(op["zero_product_count"] for op in counts["per_operation"].values())
            assert counts["model_product_count"] == counts["block_product_count"] + counts["lm_head_product_count"]
            sparsity = zeros / counts["model_product_count"]
            assert math.isclose(sparsity, point["R_model"], abs_tol=1e-15)
            checkpoint = trained[point["evidence_id"]]
            assert checkpoint["family"] == point["family"]
            assert math.isclose(checkpoint["R_model"], sparsity, abs_tol=1e-10)
            family = ("4-site" if point["family"].startswith("A4") else
                      "7-site" if point["family"].startswith("A7") else "Baseline / local")
            points.append({**point, "sparsity_percent": 100 * sparsity,
                           "dose": checkpoint["dose"], "visual_family": family})
    groups = {candidate: [p for p in points if p["candidate"] == candidate]
              for candidate in CANDIDATES}
    summaries = []
    for candidate, (label, detail) in zip(CANDIDATES, ROW_LABELS):
        gm = geometric_mean([p["speedup"] for p in groups[candidate]])
        assert math.isclose(gm, runtime["final_candidates"][candidate]["geomean"], rel_tol=1e-12)
        summaries.append({"candidate": candidate, "label": label, "detail": detail,
                          "n": len(groups[candidate]), "geomean_speedup": gm})
    increments = []
    for reference, treatment in zip(CANDIDATES[:-1], CANDIDATES[1:]):
        pairs = []
        for ref, trt in zip(groups[reference], groups[treatment]):
            assert ref["condition"] == trt["condition"] and ref["evidence_id"] == trt["evidence_id"]
            pairs.append({"condition": ref["condition"], "ratio": trt["speedup"] / ref["speedup"]})
        ratio = geometric_mean([p["ratio"] for p in pairs])
        increments.append({"reference": reference, "treatment": treatment,
                           "pairs": pairs, "geomean_ratio": ratio,
                           "relative_change_percent": 100 * (ratio - 1)})
    fit = regression(groups["k050"])
    assert math.isclose(fit["r2"], runtime["k050_regression"]["r2"], abs_tol=1e-12)
    return {
        "question": "How does model-wide sparsity relate to K050 speedup, and what do matched skip ablations contribute?",
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (SOURCE, RETROSPECTIVE)},
        "coverage": source["coverage"], "timing_block_indices": runtime["timing_block_indices"],
        "timing_metric": runtime["metric"], "physical_gpu_uuid": runtime["physical_gpu_uuid"],
        "protocol": {"gpu": "RTX5090", "execution_dtype": "BF16", "sparsity_dtype": "FP16",
                     "batch_size": 1, "sequence_length": 2048, "output_vocabulary": 50304,
                     "timing_inputs": 64, "passes_per_process": 7, "processes": 3,
                     "baseline": "same-checkpoint native PyTorch/SDPA CUDA graph",
                     "ablation_comparison": "ratios of separately measured native-normalized speedups; equal checkpoint weighting"},
        "points": points, "regression": fit, "ablations": summaries, "increments": increments,
        "highlight_condition": "c30",
        "interpretation": "Descriptive in-sample fit across distinct trained checkpoints; no causal or held-out prediction claim",
    }


def make_figure(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 9, "axes.titlesize": 8,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
    fig, (ax, ab) = plt.subplots(1, 2, figsize=(7.4, 3.4),
                                gridspec_kw={"width_ratios": (65, 35)})
    fig.subplots_adjust(left=.085, right=.985, bottom=.22, top=.88, wspace=.32)
    ax.set_title("(a) Model-wide sparsity predicts realized speedup", loc="left", pad=11)
    ab.set_title("(b) Matched kernel ablations", loc="left", pad=11)
    selected = [p for p in data["points"] if p["candidate"] == "k050"]
    for family, (color, marker) in STYLE.items():
        rows = [p for p in selected if p["visual_family"] == family]
        ax.scatter([p["sparsity_percent"] for p in rows], [p["speedup"] for p in rows],
                   s=27, c=color, marker=marker, linewidths=.35, edgecolors="white",
                   zorder=3, gid=family)
    fit = data["regression"]
    xx = np.linspace(min(p["sparsity_percent"] for p in selected), max(p["sparsity_percent"] for p in selected), 100)
    ax.plot(xx, fit["intercept"] + fit["slope_per_percentage_point"] * xx,
            color="#4E5355", ls="--", lw=1.1, zorder=2, gid="OLS")
    ax.axhline(1, color=".6", ls=":", lw=.8, zorder=1)
    ax.text(12.4, 1.52, rf'$R^2 = {fit["r2"]:.3f}$', fontsize=9,
            color="#363A3C", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    endpoint, = [p for p in selected if p["condition"] == data["highlight_condition"]]
    ax.annotate('7-site + OL1, $\\kappa = 0.5$\n' + f'{endpoint["speedup"]:.2f}' + r'$\times$',
                xy=(endpoint["sparsity_percent"], endpoint["speedup"]),
                xytext=(-6, -15), textcoords="offset points", ha="right", va="top",
                color=STYLE["7-site"][0], fontsize=7.6,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})
    ax.set(xlim=(-.5, 30), ylim=(.8, 1.85),
           xlabel=r'Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)',
           ylabel="Full-model speedup (×)")
    ax.set_xticks(range(0, 31, 5))
    ax.set_yticks(np.arange(.8, 1.81, .2))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=".93", lw=.5)
    handles = [Line2D([], [], color=color, marker=marker, ls="none", markersize=4.5, label=family)
               for family, (color, marker) in STYLE.items()]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, -.23),
              ncol=3, frameon=False, fontsize=7, handletextpad=.3, columnspacing=1.0)

    ab.set(xlim=(.995, 1.29), ylim=(-.55, 2.65), yticks=[],
           xlabel="Geometric-mean speedup (×)")
    ab.set_xticks([1.00, 1.10, 1.20])
    ab.spines["left"].set_visible(False)
    ab.tick_params(axis="y", left=False)
    ab.axvline(1, color=".55", ls=":", lw=.8)
    ab.text(1.008, 2.61, "1× native", fontsize=6.8, color=".4", va="top")
    values = [r["geomean_speedup"] for r in data["ablations"]]
    for y, row, value in zip((2, 1, 0), data["ablations"], values):
        ab.text(1.008, y + .36, row["label"], fontsize=8, va="center")
        ab.text(1.008, y + .16, row["detail"], fontsize=6.8, color=".42", va="center")
        ab.plot([1, value], [y, y], color=".76", lw=1.2, zorder=1)
        ab.scatter([value], [y], s=33, color="#41484C", zorder=3, gid=row["candidate"])
        ab.annotate(f"{value:.3f}×", xy=(value, y), xytext=(0, -10),
                    textcoords="offset points", ha="center", va="top", fontsize=8,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .5})
    for y, first, second, increment in zip((2, 1), values[:-1], values[1:], data["increments"]):
        ab.annotate("", xy=(second, y - .75), xytext=(first, y - .32),
                    arrowprops={"arrowstyle": "->", "color": ".45", "lw": .8,
                                "mutation_scale": 7, "shrinkA": 0, "shrinkB": 0})
        change = f'{increment["relative_change_percent"]:+.1f}%'.replace("-", "−")
        ab.text(min(first, second) - .013, y - .43, change,
                ha="right", va="center", fontsize=8, color="#41484C")
    return fig


def main():
    data = read_evidence()
    (HERE / "data/kernel-realization.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    fig = make_figure(data)
    fig.savefig(HERE / "figures/07-kernel-realization.pdf",
                metadata={"Title": "Model-wide sparsity, K050 speedup and matched kernel ablations",
                          "Creator": "Analysis 021 / 07_kernel_realization.py", "CreationDate": None})
    plt.close(fig)
    print(f'30 K050 checkpoints; R²={data["regression"]["r2"]:.6f}')
    for row in data["ablations"]:
        print(f'{row["label"]}: {row["geomean_speedup"]:.6f}×')


if __name__ == "__main__":
    main()
