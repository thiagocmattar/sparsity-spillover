"""K050 native-relative speedup and projection bypass for 30 checkpoints."""

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
INVESTIGATION = HERE / "investigation/data/checkpoints.json"
INVESTIGATION_FITS = HERE / "investigation/data/analysis.json"
CANDIDATES = ("k050-no-skip", "k050-attention-dense", "k050")
STYLE = {"Baseline / local": ("#777777", "o"),
         "4-site": ("#2878B5", "D"), "7-site": ("#C96024", "^")}


def geometric_mean(values):
    return math.exp(math.fsum(math.log(v) for v in values) / len(values))


def regression(points, x_key="sparsity_percent", y_key="speedup"):
    """Unweighted OLS with intercept; both plotted x variables are percentages."""
    x = np.array([p[x_key] for p in points])
    y = np.array([p[y_key] for p in points])
    slope, intercept = np.polyfit(x, y, 1)
    r2 = 1 - np.sum((y - intercept - slope * x) ** 2) / np.sum((y - y.mean()) ** 2)
    return {"n": len(x), "pearson_r": float(np.corrcoef(x, y)[0, 1]), "slope_per_percentage_point": float(slope),
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
    investigated = json.loads(INVESTIGATION.read_text(encoding="utf-8"))
    assert len(investigated) == 30 and {p["condition"] for p in investigated} == conditions
    by_condition = {p["condition"]: p for p in investigated}
    projection_points = []
    for point in groups["k050"]:
        record = by_condition[point["condition"]]
        assert record["checkpoint_evidence_id"] == point["evidence_id"]
        assert record["recipe"] == point["family"]
        assert math.isclose(record["s_model_percent"], point["sparsity_percent"], abs_tol=1e-12)
        assert math.isclose(record["full_speedup"], point["speedup"], rel_tol=1e-12)
        for prefix, candidate in (("all_skips_off", "k050-no-skip"), ("projection_on", "k050-attention-dense")):
            original = next(p for p in groups[candidate] if p["condition"] == point["condition"])
            assert math.isclose(record[f"{prefix}_paired_speedup"], original["speedup"], rel_tol=1e-12)
        bypass = record["projection_mma_bypassed"] / record["projection_mma_potential"]
        gain = record["all_skips_off_candidate_gm_ms"] / record["projection_on_candidate_gm_ms"]
        assert math.isclose(bypass, record["projection_mma_bypass_fraction"], rel_tol=1e-12)
        assert math.isclose(gain, record["projection_sparse_gain"], rel_tol=1e-12)
        projection_points.append({"condition": point["condition"], "evidence_id": point["evidence_id"],
                                  "visual_family": point["visual_family"], "bypass_percent": 100 * bypass,
                                  "projection_sparse_gain": gain,
                                  **{key: record[key] for key in ("projection_mma_bypassed", "projection_mma_potential",
                                      "projection_simt_products", "all_skips_off_candidate_gm_ms",
                                      "projection_on_candidate_gm_ms", "projection_mma_counter_method", "diagnostic_source")}})
    fit = regression(groups["k050"])
    assert math.isclose(fit["r2"], runtime["k050_regression"]["r2"], abs_tol=1e-12)
    projection_fit = regression(projection_points, "bypass_percent", "projection_sparse_gain")
    saved_fit, = [r for r in json.loads(INVESTIGATION_FITS.read_text())["associations"]
                  if r["family"] == "all" and r["predictor"] == "projection_mma_bypass_fraction"
                  and r["outcome"] == "projection_sparse_gain"]
    assert math.isclose(projection_fit["r2"], saved_fit["ols_r2"], abs_tol=1e-12)
    assert math.isclose(projection_fit["pearson_r"], saved_fit["pearson_r"], abs_tol=1e-12)
    return {
        "question": "How does model-wide sparsity relate to native-relative K050 speedup, and how does projection MMA bypass relate to projection-path gain?",
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in (SOURCE, RETROSPECTIVE, INVESTIGATION, INVESTIGATION_FITS)},
        "coverage": source["coverage"], "timing_block_indices": runtime["timing_block_indices"],
        "timing_metric": runtime["metric"], "physical_gpu_uuid": runtime["physical_gpu_uuid"],
        "protocol": {"gpu": "RTX5090", "execution_dtype": "BF16", "sparsity_dtype": "FP16",
                     "batch_size": 1, "sequence_length": 2048, "output_vocabulary": 50304,
                     "timing_inputs": 64, "passes_per_process": 7, "processes": 3,
                     "baseline": "same-checkpoint native PyTorch/SDPA CUDA graph",
                     "projection_comparison": "GM host latency of all-skips-off / GM host latency of attention-dense; same checkpoint and matched timing protocol",
                     "counter_coverage": "338 validation blocks, actual BF16 operands; includes h/z padding and SIMT substitution"},
        "points": points, "regression": fit,
        "projection_points": projection_points, "projection_regression": projection_fit,
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
                                gridspec_kw={"width_ratios": (55, 45)})
    fig.subplots_adjust(left=.085, right=.985, bottom=.23, top=.86, wspace=.28)
    ax.set_title("(a) Model-wide sparsity and realized speedup", loc="left", pad=11)
    ab.set_title("(b) Kernel-exploitable projection sparsity", loc="left", pad=11)
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
    ax.axhline(1, color=".6", ls=":", lw=.8, zorder=1, gid="reference")
    ax.text(29, 1.012, "1× native", ha="right", va="bottom", fontsize=6.8, color=".45")
    ax.text(.06, .87, rf'$R^2 = {fit["r2"]:.3f}$', transform=ax.transAxes, fontsize=9,
            color="#363A3C", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    endpoint, = [p for p in selected if p["condition"] == data["highlight_condition"]]
    ax.annotate('7-site + OL1, $\\kappa = 0.5$\n' + f'{endpoint["speedup"]:.2f}' + r'$\times$',
                xy=(endpoint["sparsity_percent"], endpoint["speedup"]),
                xytext=(28.5, 1.47), textcoords="data", ha="right", va="top",
                color=STYLE["7-site"][0], fontsize=7.6,
                arrowprops={"arrowstyle": "-", "lw": .6, "color": STYLE["7-site"][0], "shrinkA": 2, "shrinkB": 5},
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2})
    ax.set(xlim=(-.5, 30), ylim=(.95, 1.85),
           xlabel=r'Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)',
           ylabel="Full-model speedup (×)")
    ax.set_xticks(range(0, 31, 5))
    ax.set_yticks(np.arange(1., 1.81, .2))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=".93", lw=.5)
    handles = [Line2D([], [], color=color, marker=marker, ls="none", markersize=4.5, label=family)
               for family, (color, marker) in STYLE.items()]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .025),
               ncol=3, frameon=False, fontsize=7.5, handletextpad=.4, columnspacing=1.8)

    for family, (color, marker) in STYLE.items():
        rows = [p for p in data["projection_points"] if p["visual_family"] == family]
        ab.scatter([p["bypass_percent"] for p in rows], [p["projection_sparse_gain"] for p in rows],
                   s=27, c=color, marker=marker, linewidths=.35, edgecolors="white", zorder=3, gid=family)
    projection_fit = data["projection_regression"]
    xx = np.linspace(min(p["bypass_percent"] for p in data["projection_points"]),
                     max(p["bypass_percent"] for p in data["projection_points"]), 100)
    ab.plot(xx, projection_fit["intercept"] + projection_fit["slope_per_percentage_point"] * xx,
            color="#4E5355", ls="--", lw=1.1, zorder=2, gid="OLS")
    ab.axhline(1, color=".6", ls=":", lw=.8, zorder=1, gid="reference")
    ab.text(83, 1.008, "no sparse-path benefit", ha="right", va="bottom", fontsize=6.5, color=".45")
    ab.text(.06, .90, rf'$R^2 = {projection_fit["r2"]:.3f}$', transform=ab.transAxes, fontsize=9,
            color="#363A3C", bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    ab.set(xlim=(-1.5, 85), ylim=(.95, 1.40),
           xlabel="Projection MMA bypass (%)", ylabel="Projection-path gain (×)")
    ab.set_xticks([0, 20, 40, 60, 80])
    ab.set_yticks([1., 1.1, 1.2, 1.3, 1.4])
    ab.set_axisbelow(True)
    ab.grid(axis="y", color=".93", lw=.5)
    return fig


def main():
    data = read_evidence()
    (HERE / "data/kernel-realization.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    fig = make_figure(data)
    fig.savefig(HERE / "figures/07-kernel-realization.pdf",
                metadata={"Title": "K050 native-relative speedup and kernel-exploitable projection sparsity",
                          "Creator": "Analysis 021 / 07_kernel_realization.py", "CreationDate": None})
    plt.close(fig)
    print(f'30 K050 checkpoints; R²={data["regression"]["r2"]:.6f}')
    print(f'30 matched projection points; R²={data["projection_regression"]["r2"]:.6f}')


if __name__ == "__main__":
    main()
