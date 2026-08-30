"""Generate Run 005's count-first sitewise near-zero figure."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


RUN_DIR = Path(__file__).resolve().parent
VERIFICATION_PATH = RUN_DIR / "artifacts" / "verification.json"
FIGURE_DATA_PATH = RUN_DIR / "artifacts" / "figure_data_sitewise.json"
FIGURE_PATH = RUN_DIR / "figures" / "01-h-vs-site-near-zero-grid.pdf"
COMPLETION_PATH = RUN_DIR / "artifacts" / "sitewise_figure_completion.json"
SITES = ("q_post", "k_post", "v", "m")
EPSILON = 0.001
EXPECTED_CONDITIONS = (
    "relu-control",
    "relu-ol1-0p01",
    "relu-ol1-0p1",
    "relu-ol1-0p5",
    "relu-ol1-1",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def _pooled_counts(
    statistics: Mapping[str, Any], site: str, epsilon: float
) -> dict[str, Any]:
    matches = [row for row in statistics["pooled_by_site"] if row["name"] == site]
    if len(matches) != 1:
        raise ValueError(f"Expected one pooled row for site {site!r}.")
    row = matches[0]
    key = f"{epsilon:g}"
    hits = int(row["threshold_hits"][key])
    total = int(row["total"])
    if total <= 0 or not 0 <= hits <= total:
        raise ValueError(f"Invalid near-zero counts for site {site!r}.")
    fraction = hits / total
    if not math.isclose(
        fraction,
        float(row["threshold_fractions"][key]),
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(f"Stored fraction disagrees with counts for site {site!r}.")

    layers = [
        layer for layer in statistics["rows"] if layer["name"].startswith(f"{site}.layer_")
    ]
    if len(layers) != 6:
        raise ValueError(f"Expected six layer rows for site {site!r}.")
    if sum(int(layer["total"]) for layer in layers) != total:
        raise ValueError(f"Layer totals do not pool to the {site!r} total.")
    if sum(int(layer["threshold_hits"][key]) for layer in layers) != hits:
        raise ValueError(f"Layer hits do not pool to the {site!r} hit count.")
    return {
        "hits": hits,
        "total": total,
        "fraction": fraction,
        "percent": 100.0 * fraction,
    }


def _load_points(verification: Mapping[str, Any]) -> list[dict[str, Any]]:
    if verification.get("status") != "verified":
        raise ValueError("Figure input requires terminally verified Run 005 evidence.")
    if verification.get("evidence_label") != "valid_with_provenance_limitation":
        raise ValueError("Unexpected Run 005 evidence label.")
    conditions = verification.get("conditions", [])
    if [row["condition"]["id"] for row in conditions] != list(EXPECTED_CONDITIONS):
        raise ValueError("Run 005 figure conditions are incomplete or misordered.")

    points = []
    for source in conditions:
        attempt_id = str(source["attempt_id"])
        attempt_dir = RUN_DIR / "artifacts" / "attempts" / attempt_id
        statistics_path = attempt_dir / "diagnostics" / "activation_statistics.json"
        inventory_path = attempt_dir / "transfer_inventory.json"
        statistics_sha256 = _sha256(statistics_path)
        inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
        entries = {
            row["path"]: row for row in inventory["files"]
        }
        relative_statistics = "diagnostics/activation_statistics.json"
        if entries[relative_statistics]["sha256"] != statistics_sha256:
            raise ValueError(f"Activation-statistics hash mismatch for {attempt_id}.")
        statistics = json.loads(statistics_path.read_text(encoding="utf-8"))
        near_zero = {
            site: _pooled_counts(statistics, site, EPSILON)
            for site in ("h", *SITES)
        }
        condition = source["condition"]
        points.append(
            {
                "attempt_id": attempt_id,
                "condition_id": condition["id"],
                "order": int(condition["order"]),
                "label": "control"
                if condition["is_control"]
                else f"lambda={condition['pressure_weight']:g}",
                "point_label": "ctrl"
                if condition["is_control"]
                else f"{condition['pressure_weight']:g}",
                "pressure_weight": float(condition["pressure_weight"]),
                "is_control": bool(condition["is_control"]),
                "near_zero": near_zero,
                "final_validation_loss": float(source["final_validation_loss"]),
                "activation_statistics": {
                    "path": statistics_path.relative_to(RUN_DIR).as_posix(),
                    "sha256": statistics_sha256,
                },
                "transfer_inventory": {
                    "path": inventory_path.relative_to(RUN_DIR).as_posix(),
                    "sha256": _sha256(inventory_path),
                },
            }
        )
    return points


def generate() -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    verification = json.loads(VERIFICATION_PATH.read_text(encoding="utf-8"))
    points = _load_points(verification)
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
    offsets = {
        "q_post": [(0, 8), (0, -10), (0, 8), (-2, -10), (2, 8)],
        "k_post": [(0, 8), (0, -10), (0, 8), (-2, -10), (2, -10)],
        "v": [(0, 8), (0, -10), (0, 8), (-2, 8), (2, 8)],
        "m": [(0, -10), (0, 8), (0, 8), (-2, 8), (2, -10)],
    }
    x_max = max(point["near_zero"]["h"]["percent"] for point in points)
    y_max = max(
        point["near_zero"][site]["percent"] for point in points for site in SITES
    )

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharex=True, sharey=True)
    fig.subplots_adjust(
        left=0.10, right=0.975, top=0.86, bottom=0.14, hspace=0.28, wspace=0.20
    )
    legend_handle = None
    for ax, site in zip(axes.flat, SITES, strict=True):
        legend_handle = ax.plot(
            [point["near_zero"]["h"]["percent"] for point in points],
            [point["near_zero"][site]["percent"] for point in points],
            color="#D55E00",
            marker="s",
            linestyle="--",
            linewidth=1.7,
            markersize=5.7,
            markerfacecolor="white",
            markeredgewidth=1.6,
            label="ReLU",
        )[0]
        for index, point in enumerate(points):
            dx, dy = offsets[site][index]
            ax.annotate(
                point["point_label"],
                (
                    point["near_zero"]["h"]["percent"],
                    point["near_zero"][site]["percent"],
                ),
                xytext=(dx, dy),
                textcoords="offset points",
                ha="center",
                va="bottom" if dy > 0 else "top",
                fontsize=7.6,
                color="#202020",
            )
        ax.set_title(site)
        ax.set_xlim(0.0, max(1.0, x_max * 1.10))
        ax.set_ylim(0.0, max(0.01, y_max * 1.22))
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
    fig.suptitle("Pythia-14M OL1 sitewise near-zero mass", y=0.955, fontsize=15)
    fig.legend(
        [legend_handle],
        ["ReLU"],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncol=1,
        frameon=False,
    )
    fig.text(
        0.5,
        0.035,
        "Seed 0; |x| <= 1e-3; 581 updates per condition; all 338 validation blocks; labels are control or lambda",
        ha="center",
        va="bottom",
        fontsize=8.1,
        color="#404040",
    )
    FIGURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_figure = FIGURE_PATH.with_suffix(".tmp.pdf")
    fig.savefig(temporary_figure, format="pdf", bbox_inches="tight")
    plt.close(fig)
    temporary_figure.replace(FIGURE_PATH)

    figure_data = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "epsilon": EPSILON,
        "x_site": "h",
        "y_sites": list(SITES),
        "estimand": "each coordinate is integer near-zero hits pooled across all validation batches and six layers, divided by its pooled element count",
        "coverage": {
            "documents": 500,
            "complete_blocks": 338,
            "input_tokens": 692224,
            "excluded_tail_tokens": 1444,
            "seeds": 1,
        },
        "source_verification": {
            "path": VERIFICATION_PATH.relative_to(RUN_DIR).as_posix(),
            "sha256": _sha256(VERIFICATION_PATH),
            "evidence_label": verification["evidence_label"],
        },
        "series": {"relu": points},
        "shared_axes": {
            "x_starts_at_zero": True,
            "y_starts_at_zero": True,
            "shared_y_scale": True,
        },
        "figure": {
            "path": FIGURE_PATH.relative_to(RUN_DIR).as_posix(),
            "bytes": FIGURE_PATH.stat().st_size,
            "sha256": _sha256(FIGURE_PATH),
        },
        "source_code": {Path(__file__).name: _sha256(Path(__file__))},
    }
    _write_json(FIGURE_DATA_PATH, figure_data)
    _write_json(
        COMPLETION_PATH,
        {
            "schema_version": 1,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "figure": figure_data["figure"],
            "figure_data": FIGURE_DATA_PATH.relative_to(RUN_DIR).as_posix(),
            "observation": "observations/O001-h-vs-site-near-zero-grid.md",
        },
    )
    return FIGURE_PATH


if __name__ == "__main__":
    print(generate())
