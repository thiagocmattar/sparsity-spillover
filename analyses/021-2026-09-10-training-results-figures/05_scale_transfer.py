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
YLIM = (-.05, 1.35)
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
                       "clipping_points_above_display": sum(r["delta_loss_vs_A0"] > YLIM[1] for r in clipping)})
    return {
        "question": "Does high-threshold seven-site OL1 remain favorable across sizes when quality cost and architectural ceiling are explicit?",
        "source_sha256": {SOURCE.relative_to(ROOT).as_posix(): hashlib.sha256(SOURCE.read_bytes()).hexdigest()},
        "coverage": source["coverage"],
        "loss_reference": "Full-precision same-size untreated A0 evaluation, including for the measured clipping p=0 point",
        "sparsity_unit": "100 * pooled block zero products / (block + dense output projection products)",
        "utilization_reference": "Same-size A7 architectural ceiling, for every recipe; not each recipe's own ceiling",
        "display": {"delta_loss_limits": list(YLIM), "sparsity_max_percent": XLIM,
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
    fig, axes = plt.subplots(1, 3, sharey=True, figsize=(7.4, 4.2))
    fig.subplots_adjust(left=.08, right=.986, bottom=.405, top=.815, wspace=.19)
    # Boundary labels identify evaluated settings; no labels on intermediate kappa.
    offsets = {
        "14M": {"A4-OL1": ((4, -15), (5, 6)), "A7-OL1": ((-3, 16), None)},
        "70M": {"A4-OL1": ((-7, -15), (-10, -15)), "A7-OL1": ((-10, 10), None)},
        "410M": {"A4-OL1": ((-12, 8), (-8, -10)), "A7-OL1": ((-7, -16), None)},
    }
    for index, (ax, panel) in enumerate(zip(axes, data["panels"])):
        size = panel["scale"]
        ax.set(xlim=(0, XLIM[size]), ylim=YLIM, xlabel=r"$\mathcal{S}_{\mathrm{model}}$ (%)")
        ax.set_yticks([0, .3, .6, .9, 1.2])
        ax.set_xticks({"14M": [0, 10, 20, 30], "70M": [0, 10, 20, 30, 40, 50],
                       "410M": [0, 30, 60, 90]}[size])
        ax.tick_params(length=3, width=.6)
        ax.axhline(0, color="#93979C", linewidth=.65, linestyle=":", zorder=0)
        ax.set_title(f'({"abc"[index]}) {size} · A0 loss {panel["baseline"]["loss"]:.3f}',
                     loc="left", pad=44)
        for j, family in enumerate(STYLE):
            label, color, marker = STYLE[family]
            ceiling = 100 * panel["ceilings"][family[:2]]["R_model_max_fraction"]
            ax.axvline(ceiling, color=color, alpha=.4, linewidth=.85, zorder=0)
            ax.text(0, 1.17 - j * .105, f'{label[:6]} ceiling: {ceiling:.2f}%',
                    transform=ax.transAxes, color=color, fontsize=7.5, va="center")
            rows = [r for r in panel["trained"] if r["family"] == family]
            ax.plot([r["sparsity_percent"] for r in rows], [r["delta_loss_vs_A0"] for r in rows],
                    color=to_rgba(color, .72), linewidth=.9, marker=marker, markersize=4.3,
                    markerfacecolor=color, markeredgecolor=color, markeredgewidth=.5,
                    zorder=3, gid=f"{size}:{family}")
            for row, offset in zip((rows[0], rows[-1]), offsets[size][family]):
                if offset is None:
                    continue  # Combine the orange endpoint threshold and utilization below.
                ax.annotate(rf'$\kappa = {row["dose"]:g}$',
                            (row["sparsity_percent"], row["delta_loss_vs_A0"]),
                            xytext=offset, textcoords="offset points", fontsize=7, color=color,
                            ha="right" if offset[0] < 0 else "left", va="center",
                            arrowprops={"arrowstyle": "-", "color": color, "lw": .5,
                                        "alpha": .6, "shrinkA": 2, "shrinkB": 4})
        rows = panel["clipping"]
        ax.plot([r["sparsity_percent"] for r in rows], [r["delta_loss_vs_A0"] for r in rows],
                color="#999DA2", linestyle=":", linewidth=.8, marker="o", markersize=2.9,
                markerfacecolor="white", markeredgewidth=.6, zorder=1, gid=f"{size}:clipping")
        endpoint = next(r for r in panel["trained"] if r["family"] == "A7-OL1" and r["dose"] == .5)
        # The common A7 normalization replaces the old second row.
        ax.annotate(r'$\kappa = 0.5$' + f'\n{endpoint["sparsity_percent"]:.2f}% · {endpoint["A7_ceiling_used_percent"]:.1f}% of ceiling',
                    (endpoint["sparsity_percent"], endpoint["delta_loss_vs_A0"]),
                    xytext=(.98 * XLIM[size], endpoint["delta_loss_vs_A0"] - (.49 if size == "70M" else .26)),
                    textcoords="data", ha="right", va="top", linespacing=1.6,
                    fontsize=7, color=STYLE["A7-OL1"][1],
                    arrowprops={"arrowstyle": "-", "color": STYLE["A7-OL1"][1], "alpha": .45,
                                "lw": .55, "relpos": (.78, 1), "shrinkA": 3, "shrinkB": 5})
    axes[0].set_ylabel(r"$\Delta$ validation loss vs. A0")
    handles = [Line2D([], [], color=color, marker=marker, linewidth=.9, markersize=4.3, label=label)
               for label, color, marker in STYLE.values()]
    handles.append(Line2D([], [], color="#999DA2", linestyle=":", marker="o", markersize=3,
                          markerfacecolor="white", linewidth=.8, label="A0 + post-hoc clipping"))
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(.54, .27), ncol=3,
               frameon=False, fontsize=7.5, handlelength=2, columnspacing=1.6)
    fig.text(.54, .215, r"7-site + OL1 at $\kappa = 0.5$", fontsize=8, ha="center", va="center")
    table_ax = fig.add_axes((.16, .012, .76, .183))
    table_ax.axis("off")
    table = table_ax.table(cellText=table_rows(data),
                          colLabels=["Size", r"$\mathcal{S}_{\mathrm{model}}$ (%)", "7-site ceiling used",
                                     r"$\Delta L$ vs. A0"],
                          colWidths=[.12, .28, .34, .26], cellLoc="center", bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    for (row, _), cell in table.get_celld().items():
        cell.visible_edges = "TB" if row == 0 else ("B" if row == 3 else "")
        cell.set_linewidth(.5)
        cell.set_edgecolor("#A1A4A8")
        cell.PAD = .05
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
                              "Subject": "Analysis 021 O006; same-size A0 loss reference and common A7 ceiling",
                              "CreationDate": None, "ModDate": None})
    plt.close(fig)
    print(path.relative_to(ROOT).as_posix())
    print(json.dumps(table_rows(data)))


if __name__ == "__main__":
    main()
