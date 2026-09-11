"""Matched 14M OL1 additions at each threshold, from retained validation data."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "analyses/018-2026-09-08-results-materials/figure_data.json"
KAPPAS = (0, 0.01, 0.05, 0.1, 0.5)
# Match the quality-sparsity overview's recipe colors and marker identities.
STYLE = {"A4": ("4-site", "#2878B5", "D"), "A7": ("7-site", "#C96024", "^")}
SITES = {"A4": {"a", "m", "h", "z"}, "A7": {"a", "m", "h", "z", "q_post", "k_post", "v"}}


def paired_effects(trained):
    selected = [r for r in trained if r["scale"] == "14M"
                and r["family"] in {"A4", "A4-OL1", "A7", "A7-OL1"}]
    index = {(r["family"], r["dose"]): r for r in selected}
    expected = {(f + suffix, k) for f in STYLE for suffix in ("", "-OL1") for k in KAPPAS}
    assert len(index) == len(selected) and set(index) == expected, "Incomplete or duplicate matched cohort"
    pairs = []
    for family in STYLE:
        for kappa in KAPPAS:
            off, on = (index[(family + suffix, kappa)] for suffix in ("", "-OL1"))
            for key in ("initial_parameter_sha256", "schedule_sha256", "seeds",
                        "training_tokens", "validation_sha256"):
                assert off["identity"][key] == on["identity"][key], f"Unmatched {key}"
            # The OL1-only overflow-policy field is not a changed training schedule.
            for key, value in off["identity"]["training"].items():
                assert value == on["identity"]["training"][key], f"Unmatched training {key}"
            pressure = on["identity"]["pressure"]
            assert off["identity"]["pressure"]["method"] == "none"
            assert pressure["method"] == "orthogonal_l1"
            assert pressure["weight"] == pressure["step_budget"] == 1
            assert set(pressure["sites"]) == SITES[family]
            for row in (off, on):
                counts = row["counts"]
                assert counts["block_zero_product_count"] == sum(
                    op["zero_product_count"] for op in counts["per_operation"].values())
                assert counts["model_product_count"] == (
                    sum(op["product_count"] for op in counts["per_operation"].values())
                    + counts["lm_head_product_count"])
                assert row["R_model"] == counts["block_zero_product_count"] / counts["model_product_count"]
            assert off["counts"]["model_product_count"] == on["counts"]["model_product_count"]
            # Subtract pooled integer numerators before converting to percentage points.
            delta_s = 100 * (on["counts"]["block_zero_product_count"]
                             - off["counts"]["block_zero_product_count"]) / on["counts"]["model_product_count"]
            pairs.append({"family": family, "kappa": kappa, "reference": off["id"],
                          "treatment": on["id"], "delta_loss": on["loss"] - off["loss"],
                          "delta_sparsity_pp": delta_s})
    return selected, pairs


def read_evidence():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    selected, pairs = paired_effects(source["trained"])
    for pair in pairs:
        previous = next(r for r in source["contrasts"]
                        if r["reference"] == pair["reference"] and r["treatment"] == pair["treatment"])
        assert pair["delta_loss"] == previous["delta_loss"]
        assert math.isclose(pair["delta_sparsity_pp"], previous["delta_R_pp"], abs_tol=1e-12)
    return {
        "question": "At each threshold, what changes when OL1 is added to four-site or seven-site recipes?",
        "source_sha256": {SOURCE.relative_to(ROOT).as_posix(): hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "coverage": source["coverage"],
        "contrast": "pressure-on minus pressure-off at matched kappa; A4-OL1 - A4 and A7-OL1 - A7",
        "sparsity_unit": "percentage points of model-wide sparsity; pooled block zero products / (block + dense output projection products)",
        "loss_source": "same eager full-validation pass as the logical counters, not terminal training-log loss",
        "x_spacing": "categorical; connecting lines order trained endpoints and do not interpolate models",
        "pressure_weight": 1, "norm_budget": 1,
        "endpoints": [{key: row[key] for key in ("id", "family", "dose", "source", "loss", "R_model", "counts", "identity")}
                      for row in selected],
        "pairs": pairs,
    }


def make_figure(data):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(5.5, 3.5))
    fig.subplots_adjust(left=.135, right=.865, bottom=.2, top=.88, hspace=.5)
    fig.text(.5, .992, r"$\Delta = (+\mathrm{OL1}) - (\mathrm{no\ pressure}),$ at matched $\kappa$",
             ha="center", va="top", fontsize=8)
    panels = (("delta_loss", "(a) Effect of adding OL1 on validation loss",
               r"$\Delta$ validation loss", (-.05, .42)),
              ("delta_sparsity_pp", "(b) Effect of adding OL1 on model-wide sparsity",
               r"$\Delta\mathcal{S}_{\mathrm{model}}$ (pp)", (-1, 13)))
    for ax, (key, title, ylabel, limits) in zip(axes, panels):
        ax.set(ylabel=ylabel, ylim=limits, xlim=(-.15, 4.2))
        ax.set_title(title, loc="left", pad=6)
        ax.axhline(0, color="#80858B", linestyle=":", linewidth=.75, zorder=0)
        ax.set_axisbelow(True)
        ax.grid(color="#E8E9EC", linewidth=.45)
        ax.tick_params(length=3, width=.6)
        for family, (label, color, marker) in STYLE.items():
            rows = [r for r in data["pairs"] if r["family"] == family]
            ax.plot(range(5), [r[key] for r in rows], color=to_rgba(color, .72), linewidth=.9,
                    marker=marker, markersize=4.5, markerfacecolor=color, markeredgecolor=color,
                    markeredgewidth=.6, zorder=3, label=label, gid=f"{key}:{family}")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center", bbox_to_anchor=(.5, .009),
               ncol=2, frameon=False, fontsize=7.3, handlelength=1.4, markerscale=.85,
               columnspacing=1.4, handletextpad=.5, borderaxespad=0)
    axes[0].set_yticks([0, .1, .2, .3, .4], ["0", "+0.1", "+0.2", "+0.3", "+0.4"])
    axes[1].set_yticks([0, 4, 8, 12])
    axes[0].tick_params(axis="x", bottom=True, labelbottom=True)
    axes[1].set_xticks(range(5), [f"{k:g}" for k in KAPPAS])
    axes[1].set_xlabel(r"Threshold $\kappa$", labelpad=4)
    for ax, family, key in ((axes[0], "A4", "delta_loss"), (axes[1], "A7", "delta_sparsity_pp")):
        endpoint = next(r for r in data["pairs"] if r["family"] == family and r["kappa"] == .5)
        _, color, _ = STYLE[family]
        ax.annotate(f"{endpoint['delta_sparsity_pp']:+.2f} pp\n{endpoint['delta_loss']:+.2f} loss",
                    (4, endpoint[key]), xytext=(9, -1), textcoords="offset points",
                    fontsize=7.5, color=color, va="center", annotation_clip=False, linespacing=1.25)
    fig.align_ylabels(axes)
    return fig, axes


def main():
    data = read_evidence()
    (HERE / "data/14m-paired-interventions.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    fig, _ = make_figure(data)
    output = HERE / "figures/03-14m-paired-interventions.pdf"
    fig.savefig(output, metadata={"Title": "Pythia-14M: matched effects of adding OL1", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(f"Saved {output.relative_to(ROOT)}: {len(data['pairs'])} matched pairs")


if __name__ == "__main__":
    main()
