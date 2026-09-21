"""Plot the manuscript's 41-checkpoint 14M latency-quality grid; no new timings."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgba

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "analyses/028-2026-09-20-70m-optimized-grid/data/full-trained-results.json":
        "405f2cc56a6ee9fefcfb0d3135ecbd588feaaf822d62ff7773c33216f2fc18f1",
    "analyses/027-2026-09-20-run044-manuscript/data/figure-data.json":
        "7259f87a6be42c51d226a18ad4b9e6a895902e4b231bfd6dc1c706916f8c77da",
    "analyses/027-2026-09-20-run044-manuscript/figures/01-14m-complete-t2-grid.pdf":
        "bae2b5810341977b419f12a742ad237b7c82436cdb664c213901bf274f4f5ce6",
}
VIEWS = {
    "01-14m-latency-quality-frontier": {
        "title": "Latency-quality frontier on Pythia-14M",
        "xlim": (5.06, 6.08), "ylim": (.445, .671),
        "subtitle": "41 checkpoints, 10 recipes | K050 on RTX5090 | Lower is better on both axes",
        "annotations": [
            ("hz", "h", .1, (35, 5)),
            ("4", "h", .1, (16, -16)),
            ("hz", "h", .5, (11, 10)),
            ("4", "h", .5, (10, 14)),
            ("4", "all", .5, (-12, 20)),
        ],
    },
    "02-14m-latency-quality-frontier-near-base": {
        "title": "Latency-quality frontier near Base validation loss",
        "xlim": (5.085, 5.285), "ylim": (.517, .662),
        "subtitle": "Pythia-14M | 18 of 41 checkpoints | Same K050 measurements",
        "annotations": [
            ("hz", "h", .05, (13, 8)),
            ("hz", "h", .1, (13, -17)),
            ("4", "h", .05, (-12, -20)),
            ("4", "h", .1, (32, 56)),
        ],
    },
    "01-14m-latency-quality-frontier-v2": {
        "title": "Latency-quality frontier on Pythia-14M",
        "swap_axes": True,
        "xlim": (.445, .671), "ylim": (5.06, 6.08),
        "subtitle": "41 checkpoints, 10 recipes | K050 on RTX5090 | Lower is better on both axes",
        "annotations": [
            ("hz", "h", .1, (15, 5)),
            ("4", "h", .1, (9, 18)),
            ("hz", "h", .5, (16, 5)),
            ("4", "h", .5, (10, 4)),
            ("4", "all", .5, (22, -20)),
        ],
    },
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect():
    for name, expected in SOURCES.items():
        assert sha(ROOT / name) == expected, name
    full = json.loads((ROOT / next(iter(SOURCES))).read_text(encoding="utf-8"))
    style_data = json.loads((ROOT / list(SOURCES)[1]).read_text(encoding="utf-8"))
    rows = [r.copy() for r in full["trained_points"] if r["model"] == "14M"]
    assert len(rows) == len({r["checkpoint_key"] for r in rows}) == 45
    index = {r["checkpoint_key"]: r for r in rows}
    for old in style_data["trained_points"]:
        new = index[old["checkpoint_key"]]
        assert all(new[k] == old[k] for k in ("loss", "latency_ms", "kernel", "timing_session"))
    assert len(style_data["trained_points"]) == 41
    assert Counter(r["timing_session"] for r in rows) == {
        "Run029": 35, "Run033": 5, "Run041": 4, "Run044": 1}
    excluded = [r["checkpoint_key"] for r in rows if r["pressure"] == "L1"]
    rows = [r for r in rows if r["pressure"] != "L1"]
    assert len(excluded) == 4 and len(rows) == 41
    assert {r["checkpoint_key"] for r in rows} == {r["checkpoint_key"] for r in style_data["trained_points"]}
    for row in rows:
        assert row["kernel"] == "K050"
        assert math.isfinite(row["loss"]) and row["loss"] > 0
        assert math.isfinite(row["latency_ms"]) and row["latency_ms"] > 0
        assert row["coverage"]["sequences"] == 338
        assert row["coverage"]["excluded_tail_tokens"] == 1444
        row["dose"] = row["local_pressure_weight"] if row["scope"] == "1" else row["kappa"]
        row["dose_name"] = "lambda" if row["scope"] == "1" else "kappa"
    series = []
    for old in style_data["series"]:
        style = {k: v for k, v in old.items() if k != "keys"}
        series.append({**style, "marker": "o"})
    for style in series:
        group = [r for r in rows if (r["scope"], r["pressure"]) ==
                 (style["scope"], style["pressure"])]
        style["keys"] = [r["checkpoint_key"] for r in sorted(group, key=lambda r: r["dose"] or 0)]
        for row in group:
            row["recipe"] = style["label"]
        assert len(group) == (1 if style["pressure"] == "none" and style["scope"] in ("0", "1")
                              else 4 if style["scope"] == "1" else 5)
    assert len(series) == 10 and sum(len(s["keys"]) for s in series) == 41
    return rows, series, excluded


def draw(rows, series, name, view):
    swapped = view.get("swap_axes", False)
    xfield, yfield = ("latency_ms", "loss") if swapped else ("loss", "latency_ms")
    fig, ax = plt.subplots(figsize=(8.6, 5.7))
    fig.subplots_adjust(left=.10, right=.98, bottom=.20, top=.85)
    fig.suptitle(view["title"], fontsize=12.6, y=.98)
    fig.text(.54, .922, view["subtitle"], ha="center", fontsize=9, color="#646970")
    index = {r["checkpoint_key"]: r for r in rows}
    handles = []
    for series_style in series:
        group = [index[key] for key in series_style["keys"]]
        control = len(group) == 1
        color = series_style["color"]
        ls = series_style["linestyle"]
        if isinstance(ls, list):
            ls = (ls[0], tuple(ls[1]))
        style = dict(marker=series_style["marker"],
                     ms=8 if control else 5.5, mfc="white" if series_style["scope"] == "0" else color,
                     mec=color if series_style["scope"] == "0" else "white",
                     mew=1.2 if control else .65)
        x, y = [r[xfield] for r in group], [r[yfield] for r in group]
        ax.plot(x, y, color=color, ls=ls, lw=.85, alpha=.4, zorder=2)
        ax.plot(x, y, color=color, ls="none", **style, zorder=5 if control else 4)
        handles.append(Line2D([], [], color=to_rgba(color, .4), ls=ls, lw=.85,
                              **style, label=series_style["label"]))
    base = next(r for r in rows if r["scope"] == "0")
    if swapped:
        ax.axhline(base["loss"], color="#92969B", lw=.8, ls=(0, (2, 3)), zorder=1)
        ax.text(.98, base["loss"] + .01, "Base loss", transform=ax.get_yaxis_transform(),
                color="#747A81", fontsize=8.5, ha="right", va="bottom")
    else:
        ax.axvline(base["loss"], color="#92969B", lw=.8, ls=(0, (2, 3)), zorder=1)
        ax.text(base["loss"] + .005, .97, "Base loss", transform=ax.get_xaxis_transform(),
                color="#747A81", fontsize=8.5, ha="left", va="top")
    for scope, pressure, dose, offset in view["annotations"]:
        row = next(r for r in rows if (r["scope"], r["pressure"], r["dose"]) ==
                   (scope, pressure, dose))
        symbol = r"\lambda" if row["dose_name"] == "lambda" else r"\kappa"
        label = row["recipe"] + "\n" + rf"${symbol}={dose:g}$"
        ax.annotate(label, (row[xfield], row[yfield]), xytext=offset,
                    textcoords="offset points", fontsize=8.5,
                    ha="right" if offset[0] < 0 else "left",
                    va="bottom" if offset[1] > 0 else "top", color="#30343B",
                    arrowprops=dict(arrowstyle="-", color="#8A8E95", lw=.6), zorder=8)
    labels = {"loss": "Validation loss", "latency_ms": "Full-model latency (ms)"}
    ax.set(xlim=view["xlim"], ylim=view["ylim"], xlabel=labels[xfield], ylabel=labels[yfield])
    ax.tick_params(length=3, width=.65)
    ax.grid(axis="y", color="#E8EAED", lw=.6)
    ax.set_axisbelow(True)
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=9.2, handlelength=2.1, columnspacing=1.5,
               labelspacing=.85, bbox_to_anchor=(.52, .015))
    path = HERE / "figures" / f"{name}.pdf"
    fig.savefig(path, metadata={"Title": view["title"], "CreationDate": None, "ModDate": None})
    plt.close(fig)
    visible = [r["checkpoint_key"] for r in rows if view["xlim"][0] <= r[xfield] <= view["xlim"][1]
               and view["ylim"][0] <= r[yfield] <= view["ylim"][1]]
    return {"file": path.relative_to(HERE).as_posix(), "sha256": sha(path),
            "x_metric": xfield, "y_metric": yfield,
            "xlim": view["xlim"], "ylim": view["ylim"], "visible_keys": visible}


def main():
    for folder in ("figures", "data"):
        (HERE / folder).mkdir(parents=True, exist_ok=True)
    rows, series, excluded = collect()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9.9,
                         "axes.titlesize": 9.9, "axes.labelsize": 9.9,
                         "xtick.labelsize": 9, "ytick.labelsize": 9,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .65, "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans"})
    outputs = [draw(rows, series, name, view) for name, view in VIEWS.items()]
    assert [len(o["visible_keys"]) for o in outputs] == [41, 18, 41]
    result = {
        "question": "How do full-model latency and validation loss vary across the manuscript's 14M recipes?",
        "source_hashes": SOURCES, "script_sha256": sha(Path(__file__)),
        "display": "User-requested simplification: omit naive L1 and all Pareto markings; recipe lines use 0.85 pt width and 0.4 opacity, with opaque checkpoint markers.",
        "timing": "Retained K050 full-model geometric-mean host latency; RTX5090, BF16, batch 1, 2048 tokens, full logits, 64 inputs x 7 passes x 3 processes. Four sessions; no pooling or rescaling across checkpoints.",
        "loss": "Ordinary final-checkpoint FP16 validation on all 338 complete blocks from 500 MiniPile documents; 1444-token tail excluded.",
        "scope": "41 trained checkpoints, matching the main manuscript figure. Four naive-L1 appendix checkpoints are excluded at the user's request. Post-hoc clipping and execution ablations are not additional trained recipes. Run048 controls do not replace the historical T2 grid timings.",
        "rows": rows, "series": series,
        "excluded_naive_l1_keys": excluded, "outputs": outputs,
    }
    (HERE / "data/frontier.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"checkpoints": len(rows), "recipes": len(series),
                      "outputs": [{"file": o["file"], "visible_points": len(o["visible_keys"])} for o in outputs]}, indent=2))


if __name__ == "__main__":
    main()
