"""Visual-QA revision of Figure 03 using immutable reduced coordinates."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


RUN_DIR = Path(__file__).resolve().parent
DATA_PATH = RUN_DIR / "artifacts" / "figure_data_attention_output.json"
FIGURE_PATH = RUN_DIR / "figures" / "03-h-vs-attention-output-near-zero.pdf"
COMPLETION_PATH = RUN_DIR / "artifacts" / "attention_output_figure_completion.json"


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
    if data.get("y_site") != "attention_output" or float(data.get("epsilon", -1.0)) != 0.001:
        raise ValueError("Unexpected attention-output reduction input.")
    grouped = data["series"]
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
        }
    )
    styles = {
        "gelu": {"display": "GeLU", "color": "#0072B2", "marker": "o", "linestyle": "-"},
        "relu": {"display": "ReLU", "color": "#D55E00", "marker": "s", "linestyle": "--"},
    }
    offsets = {
        "gelu": [(5, -11), (5, 7), (5, 7), (5, 7), (0, -11)],
        "relu": [(0, -11), (0, 7), (-2, -11), (-2, 7), (0, -11)],
    }
    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    fig.subplots_adjust(left=0.13, right=0.97, top=0.86, bottom=0.22)
    for activation in ("gelu", "relu"):
        points = grouped[activation]
        style = styles[activation]
        ax.plot(
            [point["h_near_zero"]["percent"] for point in points],
            [point["attention_output_near_zero"]["percent"] for point in points],
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=1.8,
            markersize=6.5,
            markerfacecolor="white",
            markeredgewidth=1.8,
            label=style["display"],
        )
        for index, point in enumerate(points):
            dx, dy = offsets[activation][index]
            ax.annotate(
                point["label"],
                (point["h_near_zero"]["percent"], point["attention_output_near_zero"]["percent"]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="left" if dx >= 5 else "center",
                va="bottom" if dy > 0 else "top",
                fontsize=8.2,
                color="#202020",
            )

    all_points = grouped["gelu"] + grouped["relu"]
    x_max = max(point["h_near_zero"]["percent"] for point in all_points)
    y_max = max(point["attention_output_near_zero"]["percent"] for point in all_points)
    ax.set_xlim(0.0, max(1.0, x_max * 1.10))
    ax.set_ylim(0.0, max(0.01, y_max * 1.25))
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6, min_n_ticks=4))
    ax.set_xlabel("Near-zero mass at h, |x| <= 1e-3 (%)", labelpad=8)
    ax.set_ylabel("Near-zero mass after W_o (%)", labelpad=8)
    ax.set_title("Pythia-14M attention output before residual addition", pad=12)
    ax.grid(axis="both", color="#D9D9D9", linewidth=0.75)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.text(
        0.5,
        0.035,
        "Seed 0; 451 updates per condition; all 338 validation blocks; lines follow control to increasing lambda",
        ha="center",
        va="bottom",
        fontsize=8.2,
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
        "reason": "Visual-QA label decluttering and concise axis text; plotted coordinates unchanged",
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
