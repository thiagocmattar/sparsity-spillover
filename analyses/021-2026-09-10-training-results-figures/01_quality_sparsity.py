"""Introduction-facing 14M overview from retained, full-validation evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TRAINED_SOURCE = ROOT / "analyses/018-2026-09-08-results-materials/figure_data.json"
CLIPPING_SOURCE = ROOT / "runs/030-2026-09-08-all-models-posthoc-clipping/results/clipping-points.json"
FAMILIES = {
    "A0": "Baseline",
    "A1-H": "ReLU",
    "A1-H-OL1": "ReLU + local pressure",
    "A4": "4-site thresholds",
    "A4-OL1": "4-site + pressure",
    "A7": "7-site thresholds (+Q/K/V)",
    "A7-OL1": "7-site + pressure",
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
            record["label"] = FAMILIES[row["family"]] + (" + post-hoc clipping" if row["kind"] == "clipped" else "")
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
        "label_note": "4-site thresholds act at a,m,h,z; 7-site thresholds add q_post,k_post,v. Pressure is orthogonal L1.",
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
    fig, ax = plt.subplots(figsize=(5.5, 3.15))
    fig.subplots_adjust(left=0.12, right=0.985, bottom=0.145, top=0.985)
    ax.set(xlim=(-0.65, 31.3), ylim=YLIM,
           xlabel="Model-wide sparsity (%)",
           ylabel="Validation loss (lower is better)")
    ax.set_xticks([0, 5, 10, 15, 20, 25, 30])
    ax.set_yticks([5.2, 5.4, 5.6, 5.8, 6.0, 6.2])
    ax.grid(axis="y", color="#E8E9EC", linewidth=0.55)
    ax.set_axisbelow(True)

    for ceiling, label, color in zip(data["ceilings"],
                                    ["4-site reach", "7-site reach (+Q/K/V)"],
                                    [COLORS["projection"], COLORS["attention"]]):
        x = 100 * ceiling["reachable_product_count"] / ceiling["model_product_count"]
        ax.axvline(x, color=color, alpha=0.5, linewidth=0.75,
                   gid=f"ceiling:{ceiling['family']}")
        ax.text(x + (0.35 if ceiling["family"] == "A4" else -0.35), 6.205,
                f"{label}\n{x:.2f}%", ha="left" if ceiling["family"] == "A4" else "right",
                va="top", color=color, fontsize=7.3, linespacing=1.25,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.7})

    for family, color in [("A0", COLORS["baseline"]), ("A1-H", COLORS["relu"])]:
        rows = series(data, family, "clipped")
        ax.plot(*zip(*(xy(row) for row in rows)), color=to_rgba(color, 0.85), linestyle=":",
                marker="o", markersize=3.4, markerfacecolor="white", markeredgecolor=color,
                markeredgewidth=0.8, linewidth=0.9, zorder=2, gid=f"clipped:{family}")

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
                        color=to_rgba(color, 0.7), marker=marker, markersize=5.3,
                        markerfacecolor=color if pressure or family in {"A0", "A1-H"} else "white",
                        markeredgecolor=color, markeredgewidth=0.9,
                        linestyle="--" if pressure else "-", linewidth=0.78,
                        zorder=3, gid=f"trained:{family}")
        if family.startswith(("A4", "A7")):
            handles.append(line)

    def label(text, target, location, color, fontsize=7.5, **kwargs):
        return ax.annotate(text, xy=target, xytext=location, color=color,
                           fontsize=fontsize, va="center", linespacing=1.2,
                           arrowprops={"arrowstyle": "-", "color": color, "lw": 0.65},
                           bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.2},
                           **kwargs)

    label("Baseline", xy(series(data, "A0")[0]), (0.35, 5.09), COLORS["baseline"])
    label("ReLU", xy(series(data, "A1-H")[0]), (0.3, 5.43), COLORS["relu"])
    best_local = min(series(data, "A1-H-OL1"), key=lambda row: row["loss"])
    label("Local pressure", xy(best_local), (5.4, 5.14), COLORS["relu"])
    for family, color, location in [("A0", COLORS["baseline"], (0.15, 6.015)),
                                    ("A1-H", COLORS["relu"], (8.0, 6.03))]:
        rows = series(data, family, "clipped")
        # Attach to the drawn path between p=.4 and p=.5, not to a measured marker.
        path_target = tuple((a + b) / 2 for a, b in zip(xy(rows[4]), xy(rows[5])))
        label(FAMILIES[family] + " +\npost-hoc\nclipping", path_target, location, color)

    plain = series(data, "A7")[-1]
    pressured = series(data, "A7-OL1")[-1]
    reach = next(row for row in data["ceilings"] if row["family"] == "A7")["R_model_max_fraction"]
    utilization = 100 * pressured["R_model"] / reach
    label(f"{xy(pressured)[0]:.2f}% ({utilization:.1f}% of reach)", xy(pressured),
          (19.3, 5.41), COLORS["attention"], fontsize=7.1)
    ax.text(21.35, 6.005, r"At $\kappa = 0.5$, pressure adds",
            fontsize=7.5, color=COLORS["attention"], ha="center", va="center")
    ax.text(21.35, 5.94,
            f"+{xy(pressured)[0] - xy(plain)[0]:.1f} pp sparsity / +{pressured['loss'] - plain['loss']:.2f} loss",
            fontsize=7.5, fontweight="bold", color=COLORS["attention"], ha="center", va="center")
    # Double heads distinguish the matched comparison from the ordered sweep paths.
    ax.annotate("", xy=xy(pressured), xytext=xy(plain),
                arrowprops={"arrowstyle": "<->", "color": "#9A7060",
                            "lw": 0.65, "mutation_scale": 8, "shrinkA": 4, "shrinkB": 5},
                gid="matched-pressure-arrow")

    ax.legend(handles=handles, ncol=2, loc="lower right", bbox_to_anchor=(1, 0.045),
              frameon=True, facecolor="white", edgecolor="none", framealpha=1,
              fontsize=7.1, handlelength=1.5, columnspacing=0.75, borderpad=0.25,
              labelspacing=0.7, handletextpad=0.4, borderaxespad=0.2)
    return fig, ax


def main():
    data = read_evidence()
    (HERE / "data").mkdir(exist_ok=True)
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "data/14m-quality-sparsity.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    fig, _ = make_figure(data)
    output = HERE / "figures/01-14m-quality-sparsity.pdf"
    fig.savefig(output, metadata={"Title": "Pythia-14M: quality–sparsity trade-offs", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(output.relative_to(ROOT))


if __name__ == "__main__":
    main()
