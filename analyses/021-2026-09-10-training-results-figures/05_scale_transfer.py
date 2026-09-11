"""Cross-size quality costs and architectural ceilings from retained endpoints."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / "analyses/018-2026-09-08-results-materials/figure_data.json"
SIZES = ("14M", "70M", "410M")
KAPPAS = (0, .01, .05, .1, .5)
STYLE = {"A4-OL1": ("4-site + OL1", "#2878B5", "D"),
         "A7-OL1": ("7-site + OL1", "#C96024", "^")}
YLIM = (4.0, 6.4)
XLIM = {"14M": 32, "70M": 52, "410M": 90}


def coordinates(row, baseline, ceiling, sequences):
    """Pool operation counts; use unrounded same-size A0 and common A7 reach."""
    counts = row["counts"]
    zero = counts["block_zero_product_count"]
    total = counts["model_product_count"]
    assert zero == sum(op["zero_product_count"] for op in counts["per_operation"].values())
    assert total == counts["block_product_count"] + counts["lm_head_product_count"]
    assert total == sequences * ceiling["model_product_count"]
    assert math.isclose(row["R_model"], zero / total, abs_tol=1e-15)
    return {
        "sparsity_percent": 100 * zero / total,
        "delta_loss_vs_A0": row["loss"] - baseline["loss"],
        "A7_ceiling_used_percent": 100 * zero / (sequences * ceiling["reachable_product_count"]),
    }


def read_evidence():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    panels = []
    for size in SIZES:
        baseline, = [r for r in source["trained"] if r["scale"] == size and r["family"] == "A0"]
        ceilings = {r["family"]: r for r in source["ceilings"]
                    if r["scale"] == size and r["family"] in ("A4", "A7")}
        rows = [r for r in source["trained"] if r["scale"] == size and r["family"] in STYLE]
        assert len(rows) == 10 and {(r["family"], r["dose"]) for r in rows} == {
            (f, k) for f in STYLE for k in KAPPAS}
        trained = []
        for row in sorted(rows, key=lambda r: (r["family"], r["dose"])):
            for key in ("initial_parameter_sha256", "schedule_sha256", "seeds",
                        "training_tokens", "validation_sha256"):
                assert row["identity"][key] == baseline["identity"][key], f"Unmatched {size}: {key}"
            pressure = row["identity"]["pressure"]
            assert pressure["method"] == "orthogonal_l1"
            assert pressure["weight"] == pressure["step_budget"] == 1
            trained.append({**row, **coordinates(row, baseline, ceilings["A7"], source["coverage"]["sequences"])})
        clipping = [r for r in source["clipping"] if r["scale"] == size and r["control"] == "A0"]
        assert len(clipping) == 10 and {r["dose"] for r in clipping} == {n / 10 for n in range(10)}
        clipping = [{**r, **coordinates(r, baseline, ceilings["A7"], source["coverage"]["sequences"])}
                    for r in sorted(clipping, key=lambda r: r["dose"])]
        panels.append({"scale": size, "baseline": baseline, "ceilings": ceilings,
                       "trained": trained, "clipping": clipping,
                       "clipping_points_above_display": sum(r["loss"] > YLIM[1] for r in clipping)})
    return {
        "question": "Does high-threshold seven-site OL1 remain favorable across sizes when quality cost and architectural ceiling are explicit?",
        "source_sha256": {SOURCE.relative_to(ROOT).as_posix(): hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "coverage": source["coverage"],
        "delta_loss_reference": "Retained sidecar deltas use full-precision same-size untreated A0, including for measured clipping p=0; figure shows absolute loss",
        "sparsity_unit": "100 * pooled block zero products / (block + dense output projection products)",
        "utilization_reference": "Same-size A7 architectural ceiling, for every recipe; not each recipe's own ceiling",
        "display": {"absolute_loss_limits": list(YLIM), "sparsity_max_percent": XLIM,
                    "clipping": "All coordinates retained; axes clip the off-scale portion of the dotted paths"},
        "panels": panels,
    }


def endpoint_summary(data):
    return [{"scale": p["scale"], **next(r for r in p["trained"]
             if r["family"] == "A7-OL1" and r["dose"] == .5)} for p in data["panels"]]


def table_rows(data):
    return [[r["scale"], f'{r["sparsity_percent"]:.2f}%',
             f'{r["A7_ceiling_used_percent"]:.1f}%', f'{r["delta_loss_vs_A0"]:+.3f}']
            for r in endpoint_summary(data)]


def make_figure(data):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 9, "axes.titlesize": 8,
        "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "pdf.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(7.4, 2.95))
    fig.subplots_adjust(left=.08, right=.986, bottom=.245, top=.84, wspace=.19)
    # Boundary labels identify evaluated settings; no labels on intermediate kappa.
    offsets = {
        "14M": {"A4-OL1": ((3, -14), (5, 7)), "A7-OL1": ((-3, 14), (10, 8))},
        "70M": {"A4-OL1": ((-3, -13), (-9, 3)), "A7-OL1": ((-4, 10), (17, -9))},
        "410M": {"A4-OL1": (None, (-5, 18)), "A7-OL1": ((-4, -13), (12, 7))},
    }
    for index, (ax, panel) in enumerate(zip(axes, data["panels"])):
        size = panel["scale"]
        ax.set(xlim=(0, XLIM[size]), ylim=YLIM, xlabel=r"$\mathcal{S}_{\mathrm{model}}$ (%)")
        ax.set_yticks([4, 4.5, 5, 5.5, 6])
        ax.set_xticks({"14M": [0, 10, 20, 30], "70M": [0, 10, 20, 30, 40, 50],
                       "410M": [0, 30, 60, 90]}[size])
        ax.tick_params(length=3, width=.6)
        ax.set_axisbelow(True)
        ax.grid(axis="y", color="#ECEDEF", linewidth=.45)
        ax.set_title(f'({"abc"[index]}) {size}', loc="left", pad=24)
        for j, family in enumerate(STYLE):
            label, color, marker = STYLE[family]
            ceiling = 100 * panel["ceilings"][family[:2]]["R_model_max_fraction"]
            ax.axvline(ceiling, color=color, alpha=.25, linewidth=.8, zorder=0)
            ax.annotate(f'{ceiling:.2f}%', (ceiling, 1), xycoords=("data", "axes fraction"),
                        xytext=((-9 if j == 0 else 6) if size == "410M" else (-3 if j == 0 else 3), 4),
                        textcoords="offset points",
                        color=color, fontsize=7, ha="right", va="bottom")
            rows = [r for r in panel["trained"] if r["family"] == family]
            ax.plot([r["sparsity_percent"] for r in rows], [r["loss"] for r in rows],
                    color=to_rgba(color, .72), linewidth=.85, marker=marker, markersize=4.5,
                    markerfacecolor=color, markeredgecolor=color, markeredgewidth=.5,
                    zorder=3, gid=f"{size}:{family}")
            for row, offset in zip((rows[0], rows[-1]), offsets[size][family]):
                if offset is None:
                    continue  # Only seven-site kappa=0 is labeled in the 410M cluster.
                ax.annotate(rf'$\kappa = {row["dose"]:g}$',
                            (row["sparsity_percent"], row["loss"]),
                            xytext=offset, textcoords="offset points", fontsize=7, color=color,
                            ha="right" if offset[0] < 0 or family == "A7-OL1" and row["dose"] == .5 else "left",
                            va="center", zorder=5,
                            bbox={"facecolor": "white", "edgecolor": "none", "pad": .6},
                            arrowprops=({"arrowstyle": "-", "color": color, "lw": .45,
                                         "alpha": .5, "shrinkA": 2, "shrinkB": 4}
                                        if size == "14M" and row["dose"] == 0 else None))
        rows = panel["clipping"]
        ax.plot([r["sparsity_percent"] for r in rows], [r["loss"] for r in rows],
                color="#999DA2", alpha=.6, linestyle=":", linewidth=.7, marker="o", markersize=2.7,
                markerfacecolor="white", markeredgewidth=.6, zorder=1, gid=f"{size}:clipping")
        endpoint = next(r for r in panel["trained"] if r["family"] == "A7-OL1" and r["dose"] == .5)
        # The common A7 normalization replaces the old second row.
        ax.annotate(f'{endpoint["A7_ceiling_used_percent"]:.1f}%\nof ceiling',
                    (endpoint["sparsity_percent"], endpoint["loss"]),
                    xytext={"14M": (-18, -16), "70M": (-6, 14), "410M": (-16, -11)}[size],
                    textcoords="offset points", ha="center", multialignment="center",
                    va="bottom" if size == "70M" else "top", linespacing=1.1,
                    fontsize=7, color=STYLE["A7-OL1"][1], zorder=5,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": .6})
    axes[0].set_ylabel("Validation loss")
    handles = [Line2D([], [], color=color, marker=marker, linewidth=.9, markersize=4.3, label=label)
               for label, color, marker in STYLE.values()]
    handles.append(Line2D([], [], color="#999DA2", alpha=.6, linestyle=":", marker="o", markersize=2.7,
                          markerfacecolor="white", linewidth=.7, label="A0 + post-hoc clipping"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.54, .04), ncol=3,
               frameon=False, fontsize=7.5, handlelength=2, columnspacing=1.6)
    return fig


def main():
    data = read_evidence()
    data_path = HERE / "data/scale-transfer.json"
    data_path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    table = "# Seven-site OL1 endpoints at kappa = 0.5\n\n"
    table += "| Size | S_model (%) | Seven-site ceiling used | Delta loss vs. same-size A0 |\n"
    table += "| --- | ---: | ---: | ---: |\n"
    table += "".join("| " + " | ".join(row) + " |\n" for row in table_rows(data))
    table += "\nComputed from unrounded endpoints and integer counts. See [O006](../observations/O006-scale-transfer.md) for sources, coverage and caption.\n"
    (HERE / "tables").mkdir(exist_ok=True)
    (HERE / "tables/scale-transfer-endpoints.md").write_text(table, encoding="utf-8", newline="\n")
    fig = make_figure(data)
    path = HERE / "figures/05-scale-transfer.pdf"
    fig.savefig(path, metadata={"Title": "Quality-sparsity trade-offs across Pythia model sizes",
                              "Subject": "Analysis 021 O006; absolute validation loss and common A7 ceiling",
                              "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(path.relative_to(ROOT).as_posix())
    print(json.dumps(table_rows(data)))


if __name__ == "__main__":
    main()
