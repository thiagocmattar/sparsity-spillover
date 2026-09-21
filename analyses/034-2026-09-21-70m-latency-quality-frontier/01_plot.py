"""Two-panel 70M counterpart to Analysis033, using retained opt073 measurements."""
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "analyses/030-2026-09-20-70m-t2-optimized-endpoint/data/full-trained-results.json":
        "eaf1d4d8bdf8c86fa4231c81cec7cc498f20d21f1b2c948ba95352c587925752",
    "analyses/027-2026-09-20-run044-manuscript/data/figure-data.json":
        "7259f87a6be42c51d226a18ad4b9e6a895902e4b231bfd6dc1c706916f8c77da",
    "analyses/033-2026-09-21-14m-latency-quality-frontier/02_three_panel.py":
        "46c1231932ee6f9fb67f96e28daea6fad18a8bd11c7b2b6132d76e102597a500",
    "analyses/033-2026-09-21-14m-latency-quality-frontier/figures/04-14m-quality-savings-sparsity.pdf":
        "66832fe4b0bd3d037513e92856e55cf1b5b13017c337a5b88f860074770dc7b6",
}
HIGHLIGHTS = {("0", "none"), ("hz", "h"), ("7", "all")}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for name, digest in SOURCES.items():
        assert sha(ROOT / name) == digest, name
    source = json.loads((ROOT / list(SOURCES)[0]).read_text(encoding="utf-8"))
    palette = json.loads((ROOT / list(SOURCES)[1]).read_text(encoding="utf-8"))
    for name, digest in source["sources_sha256"].items():
        assert sha(ROOT / name) == digest, name
    cohort = [r for r in source["trained_points"] if r["model"] == "70M"]
    assert len(cohort) == 27
    omitted = [r["checkpoint_key"] for r in cohort if (r["scope"], r["pressure"]) == ("1", "none")]
    rows = [r for r in cohort if r["checkpoint_key"] not in omitted]
    assert len(omitted) == 1 and len(rows) == len({r["checkpoint_key"] for r in rows}) == 26
    assert Counter(r["timing_session"] for r in rows) == {"Run045": 25, "Run047": 1}
    groups = {(r["scope"], r["pressure"]) for r in rows}
    assert groups == {("0", "none"), ("hz", "h"), ("4", "h"), ("7", "h"), ("4", "all"), ("7", "all")}
    for key in groups - {("0", "none")}:
        assert sorted(r["kappa"] for r in rows if (r["scope"], r["pressure"]) == key) == [0, .01, .05, .1, .5]
    for row in rows:
        backend = "native_graph" if row["scope"] == "0" else "candidate_graph"
        assert row["qualified"][backend]
        assert row["kernel"] == "opt073"
        assert row["displayed_latency_ms"] == row["implementation_latency_ms"][backend]
        assert math.isclose(row["sparsity"], 100 * row["zero_product_count"] /
                            row["model_product_count"], rel_tol=1e-12, abs_tol=1e-12)
        assert row["coverage"]["sequences"] == 338 and row["coverage"]["excluded_tail_tokens"] == 1444
    base = next(r for r in rows if r["scope"] == "0")
    assert base["displayed_latency_ms"] == source["matched_summary"]["native_base_ms"]
    series = [s for s in palette["series"] if (s["scope"], s["pressure"]) in groups]
    series.sort(key=lambda s: (s["scope"], s["pressure"]) in HIGHLIGHTS)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 11, "axes.titlesize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .65, "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(1, 2, figsize=(8.45, 3.6))
    fig.subplots_adjust(left=.09, right=.985, top=.87, bottom=.24, wspace=.34)
    handles = []
    for ax, field, title in zip(axes, ("sparsity", "loss"),
                               ("(a) Sparsity and latency", "(b) Latency-quality trade-off")):
        for style in series:
            key = (style["scope"], style["pressure"])
            points = sorted((r for r in rows if (r["scope"], r["pressure"]) == key),
                            key=lambda r: r["kappa"] or 0)
            highlighted, is_base = key in HIGHLIGHTS, key == ("0", "none")
            color = style["color"] if highlighted else "#9DA3AB"
            ls = style["linestyle"]
            if isinstance(ls, list):
                ls = (ls[0], tuple(ls[1]))
            width, alpha = (1.6, 1.) if highlighted else (.85, .65)
            marker = dict(marker="o", ms=9 if is_base else 5.8 if highlighted else 4.2,
                          mfc="white" if is_base else color, mec=color if is_base else "white",
                          mew=1.5 if is_base else .7 if highlighted else .5)
            x = [r[field] for r in points]
            y = [r["displayed_latency_ms"] for r in points]
            ax.plot(x, y, color=color, ls=ls, lw=width, alpha=alpha,
                    zorder=4 if highlighted else 2)
            ax.plot(x, y, color=color, ls="none", **marker, zorder=5 if highlighted else 3)
            if ax is axes[0] and highlighted:
                handles.append(Line2D([], [], color=color, ls=ls, lw=width, **marker,
                                      label="Base (PyTorch)" if is_base else style["label"]))
        is_sparsity = field == "sparsity"
        ax.axvline(base[field], color="#92969B", lw=.8, ls=(0, (2, 3)), zorder=1)
        ax.axhline(base["displayed_latency_ms"], color="#92969B", lw=.8,
                   ls=(0, (2, 3)), zorder=1)
        ax.text(base[field] + (1.6 if is_sparsity else .06), .985,
                "Base sparsity" if is_sparsity else "Base loss",
                transform=ax.get_xaxis_transform(), color="#747A81",
                fontsize=8.5, ha="left", va="top")
        xlim, ylim = ((-1.5, 43.) if is_sparsity else (4.02, 5.47)), (1., 2.85)
        assert all(xlim[0] <= r[field] <= xlim[1] and
                   ylim[0] <= r["displayed_latency_ms"] <= ylim[1] for r in rows)
        ax.set(xlim=xlim, ylim=ylim, ylabel="Full-model latency (ms)",
               xlabel=r"Model-wide sparsity $S_{\mathrm{model}}$ (%)" if is_sparsity else "Validation loss")
        ax.set_xticks([0, 10, 20, 30, 40] if is_sparsity else [4.2, 4.5, 4.8, 5.1, 5.4])
        ax.set_yticks([1., 1.5, 2., 2.5])
        ax.set_title(title, loc="left", pad=12)
        ax.tick_params(length=3, width=.65)
        ax.grid(axis="y", color="#E8EAED", lw=.6)
        ax.set_axisbelow(True)
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=11, handlelength=2.2, columnspacing=1.7, bbox_to_anchor=(.52, -.005))
    (HERE / "figures").mkdir(exist_ok=True)
    (HERE / "data").mkdir(exist_ok=True)
    output = HERE / "figures/01-70m-sparsity-latency-quality.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=.04,
                metadata={"Title": "Pythia-70M: sparsity, latency and quality",
                          "CreationDate": None, "ModDate": None})
    plt.close(fig)
    result = dict(
        output=output.relative_to(HERE).as_posix(), output_sha256=sha(output),
        script=Path(__file__).name, script_sha256=sha(Path(__file__)),
        sources_sha256=SOURCES, upstream_sources_sha256=source["sources_sha256"],
        points=rows, omitted_relu_keys=omitted,
        panels={"a": {"x": "sparsity", "y": "displayed_latency_ms"},
                "b": {"x": "loss", "y": "displayed_latency_ms"}},
        labeled_recipes=[h.get_label() for h in handles],
        base_backend="native_graph", other_markers_backend="candidate_graph (opt073)",
        coverage=source["coverage"], loss_convention=source["loss_convention"],
        timing_note="25 unchanged Run045 points plus the Run047 T2/Ph kappa=0.5 endpoint; no pooling or rescaling. Base uses native PyTorch, as in the manuscript's 70M figure.",
        verification={"unique_checkpoints": len(rows), "recipes": len(groups),
                      "all_points_visible_in_both_panels": True, "pooled_integer_sparsity_verified": True,
                      "all_displayed_timings_match_qualified_backends": True,
                      "verified_upstream_source_hashes": len(source["sources_sha256"])})
    (HERE / "data/figure-data.json").write_text(json.dumps(result, indent=2) + "\n",
                                               encoding="utf-8", newline="\n")
    print(json.dumps({"pdf": str(output), **result["verification"]}))


if __name__ == "__main__":
    main()
