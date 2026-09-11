"""Connect 14M activation structure to measured operation contributions."""

from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ENDPOINTS = ROOT / "analyses/018-2026-09-08-results-materials/figure_data.json"
DENSITIES = ROOT / "runs/031-2026-09-08-signed-activation-density/results"
FAMILIES = ("A4-OL1", "A7-OL1")
LABELS = {"A4-OL1": "4-site + OL1", "A7-OL1": "7-site + OL1"}
COLORS = ("#2878B5", "#C96024")
GROUPS = {"FFN activations": ("h", "m"),
          "Attention activations": ("q_post", "k_post", "v")}
XLIMS = ((0., 3.), (-4., 4.))
REBIN = 10
# Retain the operation colors/order from Analysis 018. QK/PV sit together on top.
OPERATIONS = (
    ("qkv_projection", "QKV", "#b3cde3"),
    ("mlp_w1", "FFN up", "#6497c5"),
    ("mlp_w2", "FFN down", "#275e91"),
    ("attention_output_projection", "Attention output", "#8ac3a3"),
    ("qk_scores", "QK", "#f4b66c"),
    ("probability_value", "PV", "#c96d4c"),
)


def rebin_density(group, grid):
    """Conserve integer mass; normalize by all elements and actual bin widths."""
    counts = np.asarray(group["histogram"], dtype=np.int64)
    assert len(counts) == grid["bins"] and len(counts) % REBIN == 0
    assert group["nonfinite"] == 0
    assert int(counts.sum()) + group["underflow"] + group["overflow"] + group["exact_zero_count"] == group["total"]
    edges = np.linspace(grid["lower"], grid["upper"], grid["bins"] + 1,
                        dtype=np.float32)[::REBIN].astype(np.float64)
    pooled = counts.reshape(-1, REBIN).sum(axis=1)
    density = pooled / (group["total"] * np.diff(edges))
    return edges, pooled, density


def operation_contributions(counts):
    operations = counts["per_operation"]
    assert set(operations) == {key for key, _, _ in OPERATIONS}
    assert sum(op["zero_product_count"] for op in operations.values()) == counts["block_zero_product_count"]
    assert sum(op["product_count"] for op in operations.values()) == counts["block_product_count"]
    assert counts["block_product_count"] + counts["lm_head_product_count"] == counts["model_product_count"]
    return {key: 100 * operations[key]["zero_product_count"] / counts["model_product_count"]
            for key, _, _ in OPERATIONS}


def read_evidence():
    evidence = json.loads(ENDPOINTS.read_text(encoding="utf-8"))
    manifest = json.loads((DENSITIES / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "completed" and not manifest["smoke"]
    assert manifest["blocks_per_checkpoint"] == 338
    paths = [ENDPOINTS, DENSITIES / "manifest.json"]
    records = []
    for family in FAMILIES:
        matches = [row for row in evidence["trained"] if row["scale"] == "14M"
                   and row["family"] == family and row["dose"] == .5]
        assert len(matches) == 1
        endpoint = matches[0]
        path = DENSITIES / f"histograms/14M_{family}_0.5.json.gz"
        paths.append(path)
        captured = json.loads(gzip.decompress(path.read_bytes()))
        source, coverage = captured["source"], captured["coverage"]
        assert source["id"] == endpoint["id"] and source["source"] == endpoint["source"]
        assert source["identity"] == endpoint["identity"]
        assert source["training_parameter"] == .5 and source["scale"] == "14M"
        assert source["source_loss"] == endpoint["loss"]
        assert source["source_R_model"] == endpoint["R_model"]
        assert coverage["complete_block_coverage"] and coverage["sequences"] == 338
        assert coverage["input_tokens"] == 692224 and coverage["excluded_tail_tokens"] == 1444
        assert abs(coverage["loss"] - endpoint["loss"]) < .0005
        assert len(captured["hooks"]) == len(captured["rows"]) == 30
        groups = {}
        for (name, sites), limits in zip(GROUPS.items(), XLIMS):
            group = captured["groups"][name]
            assert group["sites"] == list(sites)
            expected_total = 2658140160 if name == "FFN activations" else 1594884096
            assert group["total"] == expected_total
            for field in ("total", "exact_zero_count", "underflow", "overflow"):
                assert group[field] == sum(captured["sites"][s][field] for s in sites)
            np.testing.assert_array_equal(group["histogram"],
                np.sum([captured["sites"][s]["histogram"] for s in sites], axis=0))
            edges, bins, density = rebin_density(group, captured["grid"])
            outside = int(bins[(edges[:-1] < limits[0]) | (edges[1:] > limits[1])].sum())
            outside += group["underflow"] + group["overflow"]
            groups[name] = {key: group[key] for key in
                           ("total", "exact_zero_count", "underflow", "overflow", "site_element_weights")}
            groups[name].update(bin_counts=bins.tolist(),
                exact_zero_percent=100 * group["exact_zero_count"] / group["total"],
                outside_view_percent=100 * outside / group["total"],
                maximum_density=float(density.max()))
        contributions = operation_contributions(endpoint["counts"])
        assert np.isclose(sum(contributions.values()), 100 * endpoint["R_model"], rtol=0, atol=1e-12)
        records.append({"id": endpoint["id"], "family": family,
            "checkpoint": source["checkpoint"], "checkpoint_content_sha256": source["checkpoint_content_sha256"],
            "identity": endpoint["identity"], "loss": endpoint["loss"],
            "density_evaluation_loss": coverage["loss"], "counts": endpoint["counts"],
            "S_model_percent": sum(contributions.values()), "contributions_pp": contributions,
            "attention_contribution_pp": contributions["qk_scores"] + contributions["probability_value"],
            "groups": groups})
    for field in ("initial_parameter_sha256", "schedule_sha256", "seeds", "training", "training_tokens", "validation_sha256"):
        assert records[0]["identity"][field] == records[1]["identity"][field]
    return {
        "sources": {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "coverage": evidence["coverage"], "scale": "14M", "kappa": .5,
        "normalization": "Nonzero count / (all activation elements * actual bin width); zeros excluded, no renormalization.",
        "native_grid": captured["grid"], "display_bin_edges": edges.tolist(),
        "display_xlims": {name: list(limits) for name, limits in zip(GROUPS, XLIMS)},
        "density_scale": {"scale": "symlog", "linear_threshold": .01},
        "operation_denominator": "All six block operation families plus the dense final output projection; future-masked pairs excluded.",
        "records": records,
    }


def make_figure(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "axes.axisbelow": True, "pdf.fonttype": 42})
    fig = plt.figure(figsize=(7.1, 3.35))
    ffn = fig.add_axes((.075, .24, .183, .54))
    attention = fig.add_axes((.297, .24, .183, .54), sharey=ffn)
    bars = fig.add_axes((.59, .24, .385, .54))
    fig.text(.075, .975, r"(a) Local sparsity at $\kappa=0.5$ (14M)", va="top", fontsize=9)
    fig.text(.59, .975, "(b) Operation-weighted sparsity accounting", va="top", fontsize=9)
    fig.text(.59, .90, r"More FFN zeros does not imply higher $\mathcal{S}_{\mathrm{model}}$", fontsize=7.2)
    edges = np.asarray(data["display_bin_edges"])
    for ax, (name, sites), limits in zip((ffn, attention), GROUPS.items(), XLIMS):
        for record, color in zip(data["records"], COLORS):
            group = record["groups"][name]
            density = np.asarray(group["bin_counts"]) / (group["total"] * np.diff(edges))
            ax.stairs(density, edges, color=color, linewidth=1.0)
        ax.set_yscale("symlog", linthresh=.01, linscale=.5)
        ax.set_ylim(0, 50)
        ax.set_xlim(limits)
        ax.set_yticks([0, .01, .1, 1, 10], ["0", ".01", ".1", "1", "10"])
        ax.set_xticks([0, 1, 2, 3] if name == "FFN activations" else [-4, -2, 0, 2, 4])
        ax.tick_params(length=3, width=.6)
        ax.set_xlabel(r"Activation $x$", labelpad=4)
        left, _, width, _ = ax.get_position().bounds
        title = r"FFN $(h,m)$" if name == "FFN activations" else r"Attention $(q,k,v)$"
        fig.text(left + width / 2, .90, title, ha="center", fontsize=8)
        for row, (record, color) in enumerate(zip(data["records"], COLORS)):
            zero = record["groups"][name]["exact_zero_percent"]
            fig.text(left, .85 - row * .045,
                     f"{LABELS[record['family']]}: {zero:.2f}% exact zeros",
                     color=color, fontsize=6.3)
    attention.tick_params(labelleft=False)
    ffn.set_ylabel("Nonzero density", labelpad=5)

    x = np.array([0., 1.])
    bottoms = np.zeros(2)
    for key, label, color in OPERATIONS:
        heights = np.array([row["contributions_pp"][key] for row in data["records"]])
        bars.bar(x, heights, bottom=bottoms, width=.48, color=color,
                 edgecolor="white", linewidth=.35, label=label)
        bottoms += heights
    for xx, record in zip(x, data["records"]):
        bars.text(xx, record["S_model_percent"] + .65,
                  f"{record['S_model_percent']:.2f}% total", ha="center", fontsize=8.5, fontweight="bold")
    bars.set_ylim(0, 32)
    bars.set_xlim(-.55, 1.55)
    bars.set_yticks([0, 10, 20, 30])
    bars.set_xticks(x, [LABELS[family] for family in FAMILIES])
    for label, color in zip(bars.get_xticklabels(), COLORS):
        label.set_color(color)
    bars.set_ylabel(r"Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)", labelpad=5)
    bars.grid(axis="y", color=".92", linewidth=.45)
    a4, a7 = (row["attention_contribution_pp"] for row in data["records"])
    fig.text(.59, .825, f"QK + PV: {a4:.2f} → {a7:.2f} pp", fontsize=7.2)
    handles, labels = bars.get_legend_handles_labels()
    # Rows: QKV / FFN up / FFN down, then QK / Attention output / PV.
    order = [0, 4, 1, 3, 2, 5]
    fig.legend([handles[i] for i in order], [labels[i] for i in order],
        bbox_to_anchor=(.7825, .015), loc="lower center", ncol=3, frameon=False,
        fontsize=7, handlelength=1.05, handletextpad=.4, columnspacing=.9, labelspacing=.35)
    return fig


def main():
    data = read_evidence()
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "data/14m-distributions-and-operations.json").write_text(
        json.dumps(data, indent=2) + "\n", encoding="utf-8")
    figure = make_figure(data)
    path = HERE / "figures/04-14m-distributions-and-operations.pdf"
    figure.savefig(path, metadata={"Creator": "Analysis 021; Run 031 distributions and Analysis 018 operation counters",
                                  "CreationDate": None, "ModDate": None})
    plt.close(figure)
    for row in data["records"]:
        print(row["family"], f"S_model={row['S_model_percent']:.6f}%",
              f"QK+PV={row['attention_contribution_pp']:.6f} pp", f"loss={row['loss']:.6f}")
    print(path)


if __name__ == "__main__":
    main()
