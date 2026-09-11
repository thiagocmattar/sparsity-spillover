"""Figure 07 v2: model-wide sparsity against both full and projection gains."""

import hashlib
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("kernel_figure_base", HERE / "07_kernel_realization.py")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)


def read_evidence():
    data = base.read_evidence()
    full = {p["condition"]: p for p in data["points"] if p["candidate"] == "k050"}
    for point in data["projection_points"]:
        checkpoint = full[point["condition"]]
        assert checkpoint["evidence_id"] == point["evidence_id"]
        point["sparsity_percent"] = checkpoint["sparsity_percent"]
    data["projection_regression"] = base.regression(data["projection_points"], "sparsity_percent", "projection_sparse_gain")
    data["projection_x_metric"] = "sparsity_percent"
    data["question"] = "How does model-wide sparsity relate to native-relative full K050 speedup and the matched gain from projection skipping?"
    data["variant"] = "v2; model-wide sparsity replaces projection MMA bypass on panel (b)"
    data["projection_sweeps"] = []
    for recipe in ("A4", "A4-OL1", "A7", "A7-OL1"):
        points = sorted([p for p in full.values() if p["family"] == recipe], key=lambda p: p["dose"])
        assert [p["dose"] for p in points] == [0, .01, .05, .1, .5]
        data["projection_sweeps"].append({"recipe": recipe, "visual_family": points[0]["visual_family"],
                                          "with_ol1": recipe.endswith("-OL1"),
                                          "kappas": [p["dose"] for p in points],
                                          "conditions": [p["condition"] for p in points]})
    relu, = [p for p in full.values() if p["family"] == "A1-H"]
    data["local_projection_sweeps"] = []
    for recipe in ("A1-H-L1", "A1-H-OL1"):
        points = sorted([p for p in full.values() if p["family"] == recipe], key=lambda p: p["dose"])
        assert [p["dose"] for p in points] == [.05, .1, .5, 1.]
        data["local_projection_sweeps"].append({"recipe": recipe,
                                                "with_ol1": recipe.endswith("-OL1"),
                                                "weights": [0.] + [p["dose"] for p in points],
                                                "conditions": [relu["condition"]] + [p["condition"] for p in points]})
    original = HERE / "figures/07-kernel-realization.pdf"
    data["original_figure_sha256"] = hashlib.sha256(original.read_bytes()).hexdigest()
    return data


def make_figure(data):
    fig = base.make_figure(data, projection_x=data["projection_x_metric"])
    ax = fig.axes[1]
    fit_line, = [line for line in ax.lines if line.get_gid() == "OLS"]
    fit_line.remove()
    fit_label, = [text for text in ax.texts if text.get_text().startswith('$R^2')]
    fit_label.remove()
    points = {p["condition"]: p for p in data["projection_points"]}
    for sweep in data["local_projection_sweeps"]:
        selected = [points[condition] for condition in sweep["conditions"]]
        ax.plot([p["sparsity_percent"] for p in selected], [p["projection_sparse_gain"] for p in selected],
                color=base.STYLE["1-site"][0], ls="--" if sweep["with_ol1"] else "-",
                lw=.8, alpha=.55, zorder=2, gid=sweep["recipe"])
    pressured = set()
    endpoint_labels = {"A4-OL1": (6, -1, "left", "center"),
                       "A7-OL1": (-5, 7, "right", "bottom")}
    for sweep in data["projection_sweeps"]:
        selected = [points[condition] for condition in sweep["conditions"]]
        color = base.STYLE[sweep["visual_family"]][0]
        ax.plot([p["sparsity_percent"] for p in selected], [p["projection_sparse_gain"] for p in selected],
                color=color, ls="--" if sweep["with_ol1"] else "-", lw=.95, alpha=.85,
                zorder=2, gid=sweep["recipe"])
        if sweep["with_ol1"]:
            pressured.update(sweep["conditions"])
            endpoint = selected[-1]
            dx, dy, ha, va = endpoint_labels[sweep["recipe"]]
            ax.annotate(r'$\kappa = 0.5$', xy=(endpoint["sparsity_percent"], endpoint["projection_sparse_gain"]),
                        xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, color=color, fontsize=6.8,
                        bbox={"facecolor": "white", "edgecolor": "none", "pad": .6},
                        gid=f'threshold:{sweep["recipe"]}:0.5')
    for collection in ax.collections:
        family = collection.get_gid()
        if family not in ("4-site", "7-site"):
            continue
        selected = [p for p in data["projection_points"] if p["visual_family"] == family]
        color = base.STYLE[family][0]
        collection.set_facecolors([color if p["condition"] in pressured else "white" for p in selected])
        collection.set_edgecolors(["white" if p["condition"] in pressured else color for p in selected])
        collection.set_linewidths([.35 if p["condition"] in pressured else .8 for p in selected])
    handles = [base.Line2D([], [], color=".4", ls=style, lw=.95, marker="o", ms=3.5,
                           markerfacecolor=face, markeredgewidth=.7, label=label)
               for style, face, label in (("-", "white", "No OL1"), ("--", ".4", "+ OL1"))]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(.99, .20), fontsize=6.8,
              frameon=True, facecolor="white", edgecolor="none", framealpha=.95,
              borderpad=.25, handlelength=2.2, labelspacing=.4)
    return fig


def main():
    data = read_evidence()
    (HERE / "data/kernel-realization-v2.json").write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    fig = make_figure(data)
    fig.savefig(HERE / "figures/07-kernel-realization-v2.pdf",
                metadata={"Title": "Model-wide sparsity and K050 full-model and projection-skipping gains",
                          "Creator": "Analysis 021 / 07_kernel_realization_v2.py", "CreationDate": None})
    base.plt.close(fig)
    assert hashlib.sha256((HERE / "figures/07-kernel-realization.pdf").read_bytes()).hexdigest() == data["original_figure_sha256"]
    print(f'30 checkpoints; panel (a) R²={data["regression"]["r2"]:.6f}; panel (b): six ordered sweeps, no displayed fit')


if __name__ == "__main__":
    main()
