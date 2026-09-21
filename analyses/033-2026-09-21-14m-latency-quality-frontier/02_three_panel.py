"""Combine retained 14M trade-offs and the manuscript's conditional controls."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUTPUT = HERE / "figures/04-14m-quality-savings-sparsity.pdf"
SOURCES = {
    "analyses/033-2026-09-21-14m-latency-quality-frontier/data/frontier.json":
        "9402ea4238c95683230c11c3a5ca93531efb1bfe6bc163663c3931eea6cb1f32",
    "analyses/031-2026-09-20-quality-sites-quality/data/figure-data.json":
        "7cc82fbc17ffa4c482984b6de2f1eb14e95880b6d3bdde11aeb05e63960071af",
    "manuscript/draft/figures/24-quality-sites-quality.pdf":
        "3cfebd610f35828d8a36240911eb6d6c45860d56076405c5975f70aa2ff78e65",
}
HIGHLIGHTS = {("0", "none"), ("hz", "h"), ("7", "all")}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def draw_tradeoff(ax, rows, series, xfield):
    index = {r["checkpoint_key"]: r for r in rows}
    handles = []
    for style in sorted(series, key=lambda s: (s["scope"], s["pressure"]) in HIGHLIGHTS):
        group = [index[key] for key in style["keys"] if key in index]
        if not group:
            continue
        highlighted = (style["scope"], style["pressure"]) in HIGHLIGHTS
        base = style["scope"] == "0"
        color = style["color"] if highlighted else "#9DA3AB"
        ls = style["linestyle"]
        if isinstance(ls, list):
            ls = (ls[0], tuple(ls[1]))
        width, alpha = (1.6, 1.) if highlighted else (.85, .65)
        marker = dict(marker="o", ms=9 if base else 5.8 if highlighted else 4.2,
                      mfc="white" if base else color,
                      mec=color if base else "white",
                      mew=1.5 if base else .7 if highlighted else .5)
        x = [r[xfield] for r in group]
        y = [r["latency_ms"] for r in group]
        ax.plot(x, y, color=color, ls=ls, lw=width, alpha=alpha,
                zorder=4 if highlighted else 2)
        ax.plot(x, y, color=color, ls="none", **marker,
                zorder=5 if highlighted else 3)
        if highlighted:
            handles.append(Line2D([], [], color=color, ls=ls, lw=width,
                                  **marker, label=style["label"]))
    base = next(r for r in rows if r["scope"] == "0")
    is_loss = xfield == "loss"
    ax.axvline(base[xfield], color="#92969B", lw=.8, ls=(0, (2, 3)), zorder=1)
    ax.text(base[xfield] + (.045 if is_loss else 1.3), .985,
            "Base loss" if is_loss else "Base sparsity",
            transform=ax.get_xaxis_transform(), color="#747A81",
            fontsize=8.5, ha="left", va="top")
    xlim = (5.06, 6.08) if is_loss else (-1., 30.)
    ylim = (.445, .671)
    assert all(xlim[0] <= r[xfield] <= xlim[1] and
               ylim[0] <= r["latency_ms"] <= ylim[1] for r in rows)
    ax.set(xlim=xlim, ylim=ylim, ylabel="Full-model latency (ms)",
           xlabel="Validation loss" if is_loss else
           r"Model-wide sparsity $S_{\mathrm{model}}$ (%)")
    ax.set_xticks([5.2, 5.4, 5.6, 5.8, 6.0] if is_loss else [0, 10, 20, 30])
    ax.set_yticks([.45, .50, .55, .60, .65])
    return handles


def main():
    for name, digest in SOURCES.items():
        assert sha(ROOT / name) == digest, name
    frontier = json.loads((ROOT / list(SOURCES)[0]).read_text(encoding="utf-8"))
    reference = json.loads((ROOT / list(SOURCES)[1]).read_text(encoding="utf-8"))
    table = reference["bar_chart"]
    for name, digest in table["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    raw_name = "runs/037-2026-09-19-pythia14m-operation-latency/results/operation-latency.json"
    raw = json.loads((ROOT / raw_name).read_text(encoding="utf-8"))
    assert raw["status"] == "qualified"
    assert table["verification"]["qualified_processes"] == 30
    assert [r["operation"] for r in table["rows"]] == ["a", "m", "h", "z", "qk", "pv"]
    for row in table["rows"]:
        mode = "without-" + row["operation"]
        expected = 1000 * (raw["latencies_ms"][mode] - raw["latencies_ms"]["full"])
        low, high = raw["process_ranges_ms"][mode]
        full_low, full_high = raw["process_ranges_ms"]["full"]
        span = [1000 * (low - full_high), 1000 * (high - full_low)]
        assert math.isclose(expected, row["saved_microseconds"], abs_tol=1e-10)
        assert all(math.isclose(a, b, abs_tol=1e-10) for a, b in
                   zip(span, row["cross_process_difference_span_us"]))
    keys = frontier["outputs"][0]["visible_keys"]
    assert set(keys) == set(frontier["outputs"][3]["visible_keys"])
    rows = [r for r in frontier["rows"] if r["checkpoint_key"] in keys]
    assert len(rows) == len(set(keys)) == 40
    assert len({(r["scope"], r["pressure"]) for r in rows}) == 9
    assert not any(r["pressure"] == "L1" or
                   (r["scope"], r["pressure"]) == ("1", "none") for r in rows)
    for row in rows:
        assert row["kernel"] == "K050"
        assert row["coverage"]["sequences"] == 338
        assert row["coverage"]["excluded_tail_tokens"] == 1444
        assert math.isclose(row["sparsity"], 100 * row["zero_product_count"] /
                            row["model_product_count"], rel_tol=1e-12, abs_tol=1e-12)
    preserved = {p.name: sha(p) for p in (HERE / "figures").glob("*.pdf") if p != OUTPUT}

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 11, "axes.titlesize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .65, "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.6),
                             gridspec_kw={"width_ratios": [1, .92, 1]})
    fig.subplots_adjust(left=.06, right=.989, top=.87, bottom=.24, wspace=.36)
    handles = draw_tradeoff(axes[0], rows, frontier["series"], "sparsity")
    draw_tradeoff(axes[2], rows, frontier["series"], "loss")
    # A single measured pair keeps the same-kappa comparison legible.
    arrow_from = next(r for r in rows if (r["scope"], r["pressure"], r["kappa"]) == ("7", "all", .5))
    arrow_to = next(r for r in rows if (r["scope"], r["pressure"], r["kappa"]) == ("7", "h", .5))
    axes[0].annotate("", xy=(arrow_to["sparsity"], arrow_to["latency_ms"]),
                     xytext=(arrow_from["sparsity"], arrow_from["latency_ms"]),
                     arrowprops={"arrowstyle": "-|>", "color": "#747A81", "lw": 1.,
                                 "linestyle": (0, (3, 2)), "mutation_scale": 10,
                                 "shrinkA": 5, "shrinkB": 4}, zorder=3)
    axes[0].annotate(r"Same $\kappa=0.5$",
                     xy=((arrow_to["sparsity"] + arrow_from["sparsity"]) / 2,
                         (arrow_to["latency_ms"] + arrow_from["latency_ms"]) / 2),
                     xytext=(0, -8), textcoords="offset points", ha="center",
                     va="top", fontsize=8.5, color="#646B73")

    ax = axes[1]
    bars = table["rows"]
    values = [r["saved_microseconds"] for r in bars]
    errors = [[v - r["cross_process_difference_span_us"][0] for v, r in zip(values, bars)],
              [r["cross_process_difference_span_us"][1] - v for v, r in zip(values, bars)]]
    ax.bar(range(6), values, width=.64,
           color=["#9DA5AE", "#9DA5AE", "#485667", "#485667", "#9DA5AE", "#9DA5AE"],
           yerr=errors, capsize=3,
           error_kw={"elinewidth": 1, "capthick": 1, "ecolor": "#343B44"}, zorder=3)
    for i, row in enumerate(bars):
        value = row["saved_microseconds"]
        high = row["cross_process_difference_span_us"][1]
        ax.text(i, high + 5 if value > 0 else -15, f"{value:+.1f}",
                ha="center", va="bottom" if value > 0 else "top", fontsize=10)
    ax.axhline(0, color="#515861", lw=.8, zorder=2)
    ax.set(ylim=(-30, 177), ylabel=r"Saved time ($\mu$s)", xlabel="Sparsification site")
    ax.set_xticks(range(6), [r"$a$", r"$m$", r"$h$", r"$z$", r"$q,k$", r"$v$"])
    ax.set_yticks([0, 50, 100, 150])
    for ax, title in zip(axes, ["(a) Sparsity and latency",
                                "(b) Conditional time savings",
                                "(c) Latency-quality trade-off"]):
        ax.set_title(title, loc="left", pad=12)
        ax.tick_params(length=3, width=.65)
        ax.grid(axis="y", color="#E8EAED", lw=.6)
        ax.set_axisbelow(True)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=11, handlelength=2.2, columnspacing=1.7,
               bbox_to_anchor=(.52, -.005))
    fig.savefig(OUTPUT, bbox_inches="tight", pad_inches=.04,
                metadata={"Title": "Pythia-14M: sparsity, conditional savings and quality",
                          "CreationDate": None, "ModDate": None})
    plt.close(fig)
    assert all(sha(HERE / "figures" / name) == digest for name, digest in preserved.items())
    record = dict(
        output=OUTPUT.relative_to(HERE).as_posix(), output_sha256=sha(OUTPUT),
        script=Path(__file__).name, script_sha256=sha(Path(__file__)),
        source_sha256={**SOURCES, **table["source_sha256"]},
        panels={"a": {"x": "sparsity", "y": "latency_ms", "keys": keys,
                      "comparison_arrow": {"from_key": arrow_from["checkpoint_key"],
                                           "to_key": arrow_to["checkpoint_key"],
                                           "kappa": .5,
                                           "label": "Same kappa=0.5"}},
                "b": {"checkpoint": "Pythia-14M T7/Pall, kappa=0.5", "rows": bars,
                      "source_run": "Run037", "unit": "microseconds",
                      "estimand": "Same-checkpoint skip-disabled latency minus full execution, one path at a time; thresholding stays active.",
                      "whiskers": "Extrema of differences between three process means; not confidence intervals."},
                "c": {"x": "loss", "y": "latency_ms", "keys": keys}},
        points=rows,
        labeled_recipes=[h.get_label() for h in handles],
        scope="40 checkpoints in panels a and c; panel b is a separate controlled timing session, not a decomposition of the cross-recipe grid timings. Conditional savings are not additive.",
        timing=frontier["timing"], loss=frontier["loss"], sparsity=frontier["sparsity"],
        verification={"same_40_keys_in_a_and_c": True, "pooled_sparsity_verified": True,
                      "all_six_bar_values_and_spans_match_raw_timings": True,
                      "unchanged_previous_pdfs": preserved})
    (HERE / "data/three-panel.json").write_text(json.dumps(record, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")
    print(json.dumps({"pdf": str(OUTPUT), "checkpoints_per_tradeoff_panel": len(rows),
                      "conditional_controls": len(bars), "legend": record["labeled_recipes"]}))


if __name__ == "__main__":
    main()
