"""Introduction-facing 14M overview from retained, full-validation evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAINED_SOURCE = ROOT / "analyses/018-2026-09-08-results-materials/figure_data.json"
CLIPPING_SOURCE = ROOT / "runs/030-2026-09-08-all-models-posthoc-clipping/results/clipping-points.json"
FAMILIES = {
    "A0": "Baseline",
    "A1-H": "ReLU",
    "A1-H-OL1": "ReLU + pressure",
    "A4": "Thresholds",
    "A4-OL1": "Thresholds + pressure",
    "A7": "Thresholds + attention",
    "A7-OL1": "Thresholds + attention + pressure",
}
COLORS = {"baseline": "#60656C", "relu": "#22836D", "projection": "#2878B5", "attention": "#C96024"}
YLIM = (5.03, 6.23)


def read_evidence():
    trained = json.loads(TRAINED_SOURCE.read_text(encoding="utf-8"))
    clipping = json.loads(CLIPPING_SOURCE.read_text(encoding="utf-8"))
    assert clipping["status"] == "complete_verified"
    rows = []
    for collection, selected in (
        (trained["trained"], FAMILIES),
        (clipping["points"], {"A0", "A1-H"}),
    ):
        for row in collection:
            if row["scale"] != "14M" or row["family"] not in selected:
                continue
            record = {key: row[key] for key in ("id", "kind", "family", "dose", "loss", "R_model", "source")}
            record["label"] = FAMILIES[row["family"]] + (" + clipping" if row["kind"] == "clipped" else "")
            record["counts"] = {key: row["counts"][key] for key in ("block_zero_product_count", "model_product_count")}
            record["coverage"] = row.get("coverage", trained["coverage"])
            if row["kind"] == "clipped":
                record["source_checkpoint_id"] = row["source_checkpoint_id"]
                record["clipping_sites"] = row["normalization_sites"]
            rows.append(record)
    return {
        "sources": {
            path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (TRAINED_SOURCE, CLIPPING_SOURCE)
        },
        "coverage": trained["coverage"],
        "label_note": "Thresholds act at FFN and attention projections; + attention adds Q/K/V. Pressure is orthogonal L1.",
        "omitted_training_family": "A1-H-L1",
        "display_loss_limits": list(YLIM),
        "ceilings": [row for row in trained["ceilings"] if row["scale"] == "14M" and row["family"] in {"A4", "A7"}],
        "points": rows,
    }


def series(data, family, kind="trained"):
    return sorted(
        (row for row in data["points"] if row["family"] == family and row["kind"] == kind),
        key=lambda row: -1 if row["dose"] is None else row["dose"],
    )


def xy(row):
    # Recompute the plotted ratio from pooled integer counters.
    counts = row["counts"]
    return 100 * counts["block_zero_product_count"] / counts["model_product_count"], row["loss"]


def make_figure(data):
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(5.5, 5.1))
    fig.subplots_adjust(left=0.125, right=0.97, bottom=0.225, top=0.825)
    fig.suptitle("Quality–sparsity trade-offs", y=0.986, fontsize=12, fontweight="bold")
    fig.text(0.5, 0.935, "Pythia-14M", ha="center", fontsize=9, color="#52565B")
    ax.set(xlim=(-0.65, 31.3), ylim=YLIM,
           xlabel=r"Model-wide sparsity $\mathcal{S}_{\mathrm{model}}$ (%)",
           ylabel="Validation loss (lower is better)")
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_yticks([5.2, 5.4, 5.6, 5.8, 6.0, 6.2])
    ax.grid(axis="y", color="#E8E9EC", linewidth=0.55)
    ax.set_axisbelow(True)

    for ceiling, label, color in zip(data["ceilings"],
                                    ["Projection ceiling", "With attention"],
                                    [COLORS["projection"], COLORS["attention"]]):
        x = 100 * ceiling["reachable_product_count"] / ceiling["model_product_count"]
        ax.axvline(x, color=color, alpha=0.6, linewidth=0.8, linestyle=(0, (2, 3)),
                   gid=f"ceiling:{ceiling['family']}")
        suffix = " ceiling" if ceiling["family"] == "A7" else ""
        ax.text(x, 1.025, f"{label}\n{x:.2f}%{suffix}", transform=ax.get_xaxis_transform(),
                ha="center" if ceiling["family"] == "A4" else "right", va="bottom",
                color=color, fontsize=8, linespacing=1.35)

    for family, color in [("A0", COLORS["baseline"]), ("A1-H", COLORS["relu"])]:
        rows = series(data, family, "clipped")
        ax.plot(*zip(*(xy(row) for row in rows)), color=color, linestyle=(0, (3, 2)),
                marker="o", markersize=3.2, markerfacecolor="white", markeredgewidth=0.8,
                linewidth=1.05, zorder=2, gid=f"clipped:{family}")

    handles = []
    for family in FAMILIES:
        rows = series(data, family)
        pressure = family.endswith("OL1")
        if family == "A0":
            color, marker = COLORS["baseline"], "o"
        elif family.startswith("A1"):
            color, marker = COLORS["relu"], "^" if pressure else "s"
        elif family.startswith("A4"):
            color, marker = COLORS["projection"], "D"
        else:
            color, marker = COLORS["attention"], "^"
        line, = ax.plot(*zip(*(xy(row) for row in rows)), label=FAMILIES[family],
                        color=color, marker=marker, markersize=4.2,
                        markerfacecolor=color if pressure or family in {"A0", "A1-H"} else "white",
                        markeredgewidth=0.9, linestyle="--" if pressure else "-",
                        linewidth=1.25, zorder=3, gid=f"trained:{family}")
        if family.startswith(("A4", "A7")):
            handles.append(line)

    def label(text, row, location, color, **kwargs):
        return ax.annotate(text, xy=xy(row), xytext=location, color=color,
                           fontsize=8, va="center", linespacing=1.25,
                           arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65},
                           bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
                           **kwargs)

    label("Baseline", series(data, "A0")[0], (0.45, 5.085), COLORS["baseline"])
    label("ReLU", series(data, "A1-H")[0], (0.3, 5.405), COLORS["relu"])
    best_local = min(series(data, "A1-H-OL1"), key=lambda row: row["loss"])
    label("ReLU + pressure\nlower loss", best_local, (5.4, 5.145), COLORS["relu"])
    label("Baseline\n+ clipping", series(data, "A0", "clipped")[5], (0.2, 6.035), COLORS["baseline"])
    label("ReLU\n+ clipping", series(data, "A1-H", "clipped")[5], (7.9, 6.155), COLORS["relu"])

    plain = series(data, "A7")[-1]
    pressured = series(data, "A7-OL1")[-1]
    label(f"{xy(plain)[0]:.2f}%", plain, (16.2, 5.66), COLORS["attention"])
    label(f"{xy(pressured)[0]:.2f}% sparsity\nnear 29.95% ceiling", pressured,
          (19.3, 5.42), COLORS["attention"], fontweight="bold")
    ax.text(19.2, 6.075,
            f"Adding pressure\n+{xy(pressured)[0] - xy(plain)[0]:.2f} pp sparsity\n+{pressured['loss'] - plain['loss']:.2f} loss",
            fontsize=8, color=COLORS["attention"], va="center", linespacing=1.35)
    # Connect the two matched endpoints with a light bracket, distinct from dose paths.
    ax.annotate("", xy=xy(plain), xytext=(18.8, 5.945),
                arrowprops={"arrowstyle": "-", "color": COLORS["attention"], "lw": 0.65, "alpha": 0.65})
    ax.annotate("", xy=xy(pressured), xytext=(25.7, 5.945),
                arrowprops={"arrowstyle": "-", "color": COLORS["attention"], "lw": 0.65, "alpha": 0.65})

    fig.legend(handles=handles, ncol=2, loc="lower center", bbox_to_anchor=(0.55, 0.064),
               frameon=False, fontsize=7.5, handlelength=2.2, columnspacing=1.5,
               labelspacing=0.8, handletextpad=0.6)
    fig.text(0.125, 0.022, "Thresholds: FFN and attention projections. + attention: also Q/K/V.",
             fontsize=7.1, color="#52565B")
    return fig, ax


def main():
    data = read_evidence()
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "data/14m-quality-sparsity.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    fig, _ = make_figure(data)
    output = HERE / "figures/01-14m-quality-sparsity.pdf"
    fig.savefig(output, metadata={"Title": "Quality–sparsity trade-offs: Pythia-14M", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
