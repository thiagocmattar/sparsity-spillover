"""Five-page diagnostic PDF from the retained 30-checkpoint investigation."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
STYLE = {"baseline/local": ("#687570", "o"), "4-site": ("#2878B5", "D"), "7-site": ("#C96024", "^")}
KAPPAS = (0, .01, .05, .1, .5)


def polish(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", color=".92", lw=.5)


def legend(fig, families=tuple(STYLE)):
    handles = [Line2D([], [], color=STYLE[f][0], marker=STYLE[f][1], label=f,
                      lw=.8, markersize=4) for f in families]
    fig.legend(handles=handles, ncol=len(handles), frameon=False,
               loc="lower center", bbox_to_anchor=(.5, .047), fontsize=9)


def scatter(ax, rows, x, y, factor, stats):
    for family, (color, marker) in STYLE.items():
        group = [r for r in rows if r["family"] == family]
        ax.scatter([factor * r[x] for r in group], [r[y] for r in group],
                   color=color, marker=marker, s=26, linewidths=.35, edgecolors="white", zorder=3)
    if stats["ols_r2"] is not None:
        xx = np.linspace(min(r[x] for r in rows), max(r[x] for r in rows), 100)
        ax.plot(xx * factor, stats["intercept"] + stats["slope"] * xx, "--", color=".45", lw=.9)
        ax.text(.04, .97, f'r = {stats["pearson_r"]:.3f}\n$R^2$ = {stats["ols_r2"]:.3f}',
                transform=ax.transAxes, va="top", fontsize=8,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5})
    polish(ax)


def association_page(rows, analysis):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.5))
    fig.subplots_adjust(left=.085, right=.985, top=.90, bottom=.17, hspace=.47, wspace=.40)
    spec = [
        ("s_model_percent", "full_speedup", 1, "Model-wide sparsity (%)", "Full-model speedup (×)", "(a) Whole-model relationship"),
        ("projection_scalar_zero_fraction", "projection_sparse_gain", 100, "Projection scalar zeros (%)", "Projection-sparse gain (×)", "(b) Scalar projection sparsity"),
        ("projection_mma_bypass_fraction", "projection_sparse_gain", 100, "Projection MMA bypass (%)", "Projection-sparse gain (×)", "(c) Projection kernel behavior"),
        ("attention_scalar_zero_fraction", "attention_sparse_gain", 100, "Attention scalar zeros (%)", "Attention-sparse gain (×)", "(d) Scalar attention sparsity"),
        ("attention_mma_skip_fraction", "attention_sparse_gain", 100, "Attention MMA skipped (%)", "Attention-sparse gain (×)", "(e) Attention kernel behavior"),
        ("s_model_percent", "full_candidate_gm_ms", 1, "Model-wide sparsity (%)", "K050 latency (ms)", "(f) Absolute optimized latency"),
    ]
    for ax, (x, y, factor, xlabel, ylabel, title) in zip(axes.flat, spec):
        stats = next(r for r in analysis["associations"] if (r["predictor"], r["outcome"], r["family"]) == (x, y, "all"))
        scatter(ax, rows, x, y, factor, stats)
        ax.set(xlabel=xlabel, ylabel=ylabel, title=title)
        if y == "attention_sparse_gain":
            ax.set_ylim(.981, 1.001)
            ax.axhline(1, color=".5", ls=":", lw=.7)
    fig.suptitle("K050: scalar sparsity, kernel counters and runtime gains", fontsize=12)
    legend(fig)
    fig.text(.5, .015, "30 checkpoints. Counters: 338-block BF16 diagnostic; timings: 64-input subset. MMA bypass includes SIMT substitution and padding.",
             ha="center", fontsize=7.6)
    return fig


def family_page(rows, analysis):
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.4))
    fig.subplots_adjust(left=.075, right=.985, top=.78, bottom=.25, wspace=.40)
    for ax, family in zip(axes, STYLE):
        group = [r for r in rows if r["family"] == family]
        stats = next(r for r in analysis["associations"] if r["predictor"] == "s_model_percent" and r["outcome"] == "full_speedup" and r["family"] == family)
        scatter(ax, group, "s_model_percent", "full_speedup", 1, stats)
        ax.set(xlabel="Model-wide sparsity (%)", ylabel="Full-model speedup (×)", title=f"{family} (n = 10)")
    fig.suptitle("The sparsity-speedup relationship also exists within recipe families", fontsize=11, y=.98)
    summary = analysis["family_summary"]
    fig.text(.5, .08, f'Family means alone: $R^2$ = {summary["family_only_r2"]:.3f}; after within-family centering: $R^2$ = {summary["family_centered_smodel_vs_speedup"]["ols_r2"]:.3f}.', ha="center", fontsize=9)
    fig.text(.5, .02, "Axis ranges vary to expose within-family variation. Descriptive fits across different trained checkpoints; no causal interpretation.", ha="center", fontsize=8)
    return fig


def matched_page(rows, pressure):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.2))
    fig.subplots_adjust(left=.08, right=.985, top=.88, bottom=.17, hspace=.42, wspace=.42)
    specs = [
        ("projection_model_contribution_pp", 1, "Projection contribution (pp)", "(a) Projection scalar contribution"),
        ("projection_mma_bypass_fraction", 100, "Projection MMA bypass (%)", "(b) Projection kernel behavior"),
        ("attention_mma_skip_fraction", 100, "Attention MMA skipped (%)", "(c) Attention kernel behavior"),
        ("projection_sparse_gain", 1, "Projection-sparse gain (×)", "(d) Projection skipping benefit"),
        ("attention_sparse_gain", 1, "Attention-sparse gain (×)", "(e) Attention skipping cost"),
        ("full_speedup", 1, "Full-model speedup (×)", "(f) Native-relative speedup"),
    ]
    for ax, (metric, factor, ylabel, title) in zip(axes.flat, specs):
        for family in ("4-site", "7-site"):
            group = sorted([r for r in rows if r["family"] == family and r["pressure"] == pressure], key=lambda r: r["kappa"])
            assert [r["kappa"] for r in group] == list(KAPPAS)
            color, marker = STYLE[family]
            ax.plot(range(5), [factor * r[metric] for r in group], color=color, marker=marker, ms=4.5, lw=.9)
        ax.set(xlabel=r"Threshold $\kappa$", ylabel=ylabel, title=title)
        ax.set_xticks(range(5), ["0", "0.01", "0.05", "0.1", "0.5"])
        if metric in ("projection_sparse_gain", "attention_sparse_gain"):
            ax.axhline(1, color=".5", ls=":", lw=.7)
        polish(ax)
    label = "without pressure" if pressure == "none" else "with OL1"
    fig.suptitle(f"Matched 4-site versus 7-site recipes {label}", fontsize=12)
    legend(fig, ("4-site", "7-site"))
    fig.text(.5, .016, "Same threshold and pressure method. Projection gain = no-skip / attention-dense latency; attention gain = attention-dense / full latency.", ha="center", fontsize=7.5)
    return fig


def latency_page(rows):
    fig, axes = plt.subplots(2, 3, figsize=(9, 6.0))
    fig.subplots_adjust(left=.085, right=.985, top=.88, bottom=.17, hspace=.44, wspace=.36)
    metrics = (("native_baseline_gm_ms", "Native SDPA"), ("all_skips_off_candidate_gm_ms", "K050: all skips off"),
               ("full_candidate_gm_ms", "Full K050"))
    for i, pressure in enumerate(("none", "orthogonal_l1")):
        for j, (metric, title) in enumerate(metrics):
            ax = axes[i, j]
            for family in ("4-site", "7-site"):
                group = sorted([r for r in rows if r["family"] == family and r["pressure"] == pressure], key=lambda r: r["kappa"])
                color, marker = STYLE[family]
                ax.plot(range(5), [r[metric] for r in group], color=color, marker=marker, ms=4.5, lw=.9)
            ax.set(xlabel=r"Threshold $\kappa$", ylabel="Geometric-mean latency (ms)",
                   title=title + (" · no pressure" if i == 0 else " · OL1"))
            ax.set_xticks(range(5), ["0", "0.01", "0.05", "0.1", "0.5"])
            all_values = [r[metric] for r in rows if r["family"] in ("4-site", "7-site")]
            margin = .12 * (max(all_values) - min(all_values))
            ax.set_ylim(min(all_values) - margin, max(all_values) + margin)
            polish(ax)
    fig.suptitle("Higher native-relative speedup does not mean lower K050 latency", fontsize=12)
    legend(fig, ("4-site", "7-site"))
    fig.text(.5, .016, "Same 64 inputs × 7 passes × 3 processes per implementation; columns use different latency ranges, shared between pressure settings.", ha="center", fontsize=7.7)
    return fig


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 9,
                         "axes.titlesize": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .6, "pdf.fonttype": 42})
    rows = json.loads((HERE / "data/checkpoints.json").read_text())
    analysis = json.loads((HERE / "data/analysis.json").read_text())
    (HERE / "figures").mkdir(exist_ok=True)
    with PdfPages(HERE / "figures/kernel-investigation.pdf", metadata={"Title": "K050 structured sparsity and matched runtime investigation", "CreationDate": None}) as output:
        for fig in (association_page(rows, analysis), family_page(rows, analysis),
                    matched_page(rows, "none"), matched_page(rows, "orthogonal_l1"), latency_page(rows)):
            output.savefig(fig)
            plt.close(fig)
    print("Wrote five-page kernel-investigation.pdf")


if __name__ == "__main__":
    main()
