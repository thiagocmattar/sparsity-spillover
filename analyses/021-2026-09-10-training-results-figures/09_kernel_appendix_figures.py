"""Three focused kernel appendix figures from the retained investigation."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
import numpy as np

HERE = Path(__file__).resolve().parent
STYLE = {"baseline/local": ("#777777", "o"),
         "4-site": ("#2878B5", "D"), "7-site": ("#C96024", "^")}
LABELS = {"baseline/local": "Baseline / local", "4-site": "4-site", "7-site": "7-site"}
KAPPAS = (0, .01, .05, .1, .5)
SMODEL = r"Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)"


def read_evidence():
    paths = [HERE / "investigation/data/checkpoints.json", HERE / "investigation/data/analysis.json"]
    rows, analysis = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    assert [row["condition"] for row in rows] == [f"c{i:02d}" for i in range(1, 31)]
    assert all(sum(row["family"] == family for row in rows) == 10 for family in STYLE)
    sources = {path.relative_to(HERE).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in paths}
    return rows, analysis, sources


def setup_style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
                         "axes.labelsize": 8.5, "axes.titlesize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans"})


def fit_for(analysis, predictor, outcome, family="all"):
    fit, = [entry for entry in analysis["associations"]
            if (entry["predictor"], entry["outcome"], entry["family"]) == (predictor, outcome, family)]
    return fit


def scatter(ax, rows, predictor, outcome, scale, fit, *, pearson=False):
    for family, (color, marker) in STYLE.items():
        selected = [row for row in rows if row["family"] == family]
        if selected:
            ax.scatter([scale * row[predictor] for row in selected], [row[outcome] for row in selected],
                       s=27, color=color, marker=marker, edgecolors="white", linewidths=.35,
                       zorder=3, gid=family)
    xx = np.linspace(min(row[predictor] for row in rows), max(row[predictor] for row in rows), 100)
    ax.plot(scale * xx, fit["intercept"] + fit["slope"] * xx,
            color=".4", ls="--", lw=.85, zorder=2, gid="OLS")
    label = (rf'$r = {fit["pearson_r"]:.3f}$' + "\n") if pearson else ""
    label += rf'$R^2 = {fit["ols_r2"]:.3f}$'
    ax.text(.045, .95, label, transform=ax.transAxes, ha="left", va="top", fontsize=8,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1}, gid="fit-label")
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=".93", lw=.5)


def shared_legend(fig, families, y):
    handles = [Line2D([], [], color=STYLE[family][0], marker=STYLE[family][1],
                      ls="none", markersize=4.5, label=LABELS[family]) for family in families]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, y), ncol=len(handles),
               fontsize=8, frameon=False, handletextpad=.4, columnspacing=1.8)


def within_family(rows, analysis):
    fig, axes = plt.subplots(1, 3, figsize=(8.7, 3.3))
    fig.subplots_adjust(left=.09, right=.985, bottom=.29, top=.86, wspace=.47)
    for letter, ax, family in zip("abc", axes, STYLE):
        selected = [row for row in rows if row["family"] == family]
        fit = fit_for(analysis, "s_model_percent", "full_speedup", family)
        scatter(ax, selected, "s_model_percent", "full_speedup", 1, fit, pearson=True)
        ax.set_title(f"({letter}) {LABELS[family]} ($n=10$)", loc="left", pad=10)
        ax.set(xlabel=SMODEL, ylabel="Native-relative full-model\nspeedup (×)")
        ax.margins(x=.09, y=.13)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    centered = analysis["family_summary"]["family_centered_smodel_vs_speedup"]["ols_r2"]
    fig.text(.5, .07, rf'Within-family centered: $R^2 = {centered:.3f}$', ha="center", fontsize=8.5)
    return fig


def scalar_and_kernel(rows, analysis):
    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.7), sharey="row")
    fig.subplots_adjust(left=.11, right=.98, bottom=.20, top=.92, hspace=.52, wspace=.34)
    specs = [
        ("projection_scalar_zero_fraction", "projection_sparse_gain", "Projection scalar zeros (%)",
         "(a) Scalar projection sparsity", (-2, 102)),
        ("projection_mma_bypass_fraction", "projection_sparse_gain", "Projection MMA bypass (%)",
         "(b) Projection kernel-skippable work", (-2, 87)),
        ("attention_scalar_zero_fraction", "attention_sparse_gain", "Attention scalar zeros (%)",
         "(c) Scalar attention sparsity", (-2, 102)),
        ("attention_mma_skip_fraction", "attention_sparse_gain", "Attention MMA skipped (%)",
         "(d) Attention kernel-skippable work", (-2, 70)),
    ]
    for i, (ax, (predictor, outcome, xlabel, title, xlim)) in enumerate(zip(axes.flat, specs)):
        scatter(ax, rows, predictor, outcome, 100, fit_for(analysis, predictor, outcome))
        projection = i < 2
        ax.set(xlabel=xlabel, xlim=xlim, ylim=(.95, 1.40) if projection else (.982, 1.001))
        ax.set_title(title, loc="left", pad=9)
        ax.set_ylabel("Projection-skipping gain (×)" if projection else "Attention-skipping gain (×)")
        ax.set_yticks([1., 1.1, 1.2, 1.3, 1.4] if projection else [.985, .990, .995, 1.000])
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f" if projection else "%.3f"))
        ax.tick_params(labelleft=True)
        ax.xaxis.set_major_locator(MaxNLocator(5))
        ax.axhline(1, color=".55", ls=":", lw=.75, zorder=1, gid="reference")
        if i % 2 == 0:
            ax.annotate("1×: no sparse-path benefit", xy=(.98, 1), xycoords=("axes fraction", "data"),
                        xytext=(0, 3 if projection else -4), textcoords="offset points",
                        ha="right", va="bottom" if projection else "top", fontsize=6.6, color=".45",
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": .6})
    shared_legend(fig, tuple(STYLE), .068)
    fig.text(.5, .025, "MMA bypass = matrix-multiply instructions skipped", ha="center", fontsize=7.5)
    return fig


def absolute_latency(rows):
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 5.8), sharey="col")
    fig.subplots_adjust(left=.125, right=.985, bottom=.16, top=.84, hspace=.43, wspace=.48)
    metrics = (("native_baseline_gm_ms", "Native SDPA"),
               ("all_skips_off_candidate_gm_ms", "K050: all sparse skips off"),
               ("full_candidate_gm_ms", "Full K050"))
    for i, pressure in enumerate(("none", "orthogonal_l1")):
        for j, (metric, title) in enumerate(metrics):
            ax = axes[i, j]
            for family in ("4-site", "7-site"):
                selected = sorted([row for row in rows if row["family"] == family
                                   and row["pressure"] == pressure], key=lambda row: row["kappa"])
                assert [row["kappa"] for row in selected] == list(KAPPAS)
                color, marker = STYLE[family]
                ax.plot(range(5), [row[metric] for row in selected], color=color, marker=marker,
                        ms=4.6, lw=.85, markeredgecolor="white", markeredgewidth=.35, gid=family)
            ax.set(xlabel=r"Threshold $\kappa$", ylabel="Geometric-mean latency (ms)", xlim=(-.18, 4.18))
            ax.set_xticks(range(5), ["0", "0.01", "0.05", "0.1", "0.5"])
            values = [row[metric] for row in rows if row["family"] in ("4-site", "7-site")]
            pad = .12 * (max(values) - min(values))
            ax.set_ylim(min(values) - pad, max(values) + pad)
            ax.yaxis.set_major_locator(MaxNLocator(4))
            ax.yaxis.set_major_formatter(FormatStrFormatter("%.4f" if j == 1 else "%.3f"))
            ax.tick_params(labelleft=True)
            ax.set_axisbelow(True)
            ax.grid(axis="y", color=".93", lw=.5)
            if i == 0:
                ax.set_title(title, pad=11)
        position = axes[i, 0].get_position()
        fig.text(.018, (position.y0 + position.y1) / 2, "No OL1" if i == 0 else "+ OL1",
                 rotation=90, va="center", ha="center", fontsize=9)
    fig.suptitle("Native-relative speedup can reflect differences in the native baseline", fontsize=10, y=.96)
    shared_legend(fig, ("4-site", "7-site"), .04)
    return fig


def main():
    setup_style()
    rows, analysis, sources = read_evidence()
    output = HERE / "figures/appendix"
    output.mkdir(exist_ok=True)
    figures = [
        ("01-kernel-within-family.pdf", within_family(rows, analysis)),
        ("02-kernel-scalar-vs-skippable.pdf", scalar_and_kernel(rows, analysis)),
        ("03-kernel-absolute-latency.pdf", absolute_latency(rows)),
    ]
    for filename, figure in figures:
        figure.savefig(output / filename, metadata={"Title": filename.removesuffix(".pdf"),
                                                   "Creator": "Analysis 021 / 09_kernel_appendix_figures.py",
                                                   "CreationDate": None})
        plt.close(figure)
    manifest = {"source_sha256": sources, "conditions": [row["condition"] for row in rows],
                "figure_order": [filename for filename, _ in figures],
                "scope": "retained 30-checkpoint investigation; no new measurements; manuscript unchanged",
                "mma_interpretation": "Instruction bypass includes scalar substitution and padding; not eliminated arithmetic"}
    (HERE / "data/kernel-appendix-figures.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n")
    print("Wrote three appendix PDFs using the retained 30-checkpoint cohort.")


if __name__ == "__main__":
    main()
