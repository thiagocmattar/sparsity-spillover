"""Visual-QA revision of Figure 02 using the immutable reduced coordinates."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
DATA_PATH = RUN_DIR / "artifacts" / "figure_data_sitewise.json"
FIGURE_PATH = RUN_DIR / "figures" / "02-h-vs-site-near-zero-grid.pdf"
COMPLETION_PATH = RUN_DIR / "artifacts" / "sitewise_figure_completion.json"
SITES = ("q_post", "k_post", "v", "m")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def refine() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if data.get("y_sites") != list(SITES) or float(data.get("epsilon", -1.0)) != 0.001:
        raise ValueError("Unexpected sitewise reduction input.")
    grouped = data["series"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 11,
            "axes.labelsize": 9.2,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
        }
    )
    styles = {
        "gelu": {"display": "GeLU", "color": "#0072B2", "marker": "o", "linestyle": "-"},
        "relu": {"display": "ReLU", "color": "#D55E00", "marker": "s", "linestyle": "--"},
    }
    offsets = {
        "q_post": {
            "gelu": [(5, -11), (0, 20), (0, 7), (8, 20), (0, 7)],
            "relu": [(0, -11), (0, 7), (-2, -11), (-2, 7), (0, -11)],
        },
        "k_post": {
            "gelu": [(5, -11), (0, 20), (0, 7), (8, 20), (0, 7)],
            "relu": [(0, -11), (0, 7), (-2, -11), (-2, 7), (0, -11)],
        },
        "v": {
            "gelu": [(4, -10), (4, 7), (0, 7), (0, 7), (0, 7)],
            "relu": [(0, -11), (0, 7), (-2, -11), (-2, 7), (0, -11)],
        },
        "m": {
            "gelu": [(5, -11), (0, 14), (3, 5), (9, 14), (0, 7)],
            "relu": [(0, -11), (0, 7), (-3, -11), (-2, 7), (1, -11)],
        },
    }
    all_points = grouped["gelu"] + grouped["relu"]
    x_max = max(point["near_zero"]["h"]["percent"] for point in all_points)
    y_max = max(point["near_zero"][site]["percent"] for point in all_points for site in SITES)

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.975, top=0.86, bottom=0.14, hspace=0.28, wspace=0.20)
    legend_handles = []
    for ax, site in zip(axes.flat, SITES, strict=True):
        for activation in ("gelu", "relu"):
            points = grouped[activation]
            style = styles[activation]
            line = ax.plot(
                [point["near_zero"]["h"]["percent"] for point in points],
                [point["near_zero"][site]["percent"] for point in points],
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                linewidth=1.7,
                markersize=5.7,
                markerfacecolor="white",
                markeredgewidth=1.6,
                label=style["display"],
            )[0]
            if site == SITES[0]:
                legend_handles.append(line)
            for index, point in enumerate(points):
                dx, dy = offsets[site][activation][index]
                ax.annotate(
                    "ctrl" if point["is_control"] else f"{point['pressure_weight']:g}",
                    (point["near_zero"]["h"]["percent"], point["near_zero"][site]["percent"]),
                    xytext=(dx, dy),
                    textcoords="offset points",
                    ha="left" if dx >= 5 else "center",
                    va="bottom" if dy > 0 else "top",
                    fontsize=7.6,
                    color="#202020",
                )
        ax.set_title(site)
        ax.set_xlim(0.0, max(1.0, x_max * 1.10))
        ax.set_ylim(0.0, max(0.01, y_max * 1.25))
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        ax.grid(axis="both", color="#D9D9D9", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes[-1, :]:
        ax.set_xlabel("Near-zero mass at h (%)")
    for ax, site in zip(axes[:, 0], ("q_post", "v"), strict=True):
        ax.set_ylabel(f"Near-zero mass at {site} (%)")
    for ax, site in zip(axes[:, 1], ("k_post", "m"), strict=True):
        ax.set_ylabel(f"Near-zero mass at {site} (%)")
    fig.suptitle("Pythia-14M L1N sitewise near-zero mass", y=0.955, fontsize=15)
    fig.legend(
        legend_handles,
        [styles[activation]["display"] for activation in ("gelu", "relu")],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=2,
        frameon=False,
    )
    fig.text(
        0.5,
        0.035,
        "Seed 0; |x| <= 1e-3; 451 updates per condition; all 338 validation blocks; labels are control or lambda",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#404040",
    )
    temporary_figure = FIGURE_PATH.with_suffix(".visual-qa.pdf")
    fig.savefig(temporary_figure, format="pdf", bbox_inches="tight")
    plt.close(fig)
    temporary_figure.replace(FIGURE_PATH)

    data["figure"] = {
        "path": FIGURE_PATH.relative_to(RUN_DIR).as_posix(),
        "bytes": FIGURE_PATH.stat().st_size,
        "sha256": _sha256(FIGURE_PATH),
    }
    data["plotting_revision"] = {
        "path": Path(__file__).name,
        "sha256": _sha256(Path(__file__)),
        "reason": "Visual-QA label decluttering; plotted coordinates and shared scales unchanged",
    }
    _write_json(DATA_PATH, data)
    completion = json.loads(COMPLETION_PATH.read_text(encoding="utf-8"))
    completion["completed_at"] = datetime.now(timezone.utc).isoformat()
    completion["figure"] = data["figure"]
    completion["plotting_revision"] = data["plotting_revision"]
    _write_json(COMPLETION_PATH, completion)
    return FIGURE_PATH


if __name__ == "__main__":
    print(refine())
