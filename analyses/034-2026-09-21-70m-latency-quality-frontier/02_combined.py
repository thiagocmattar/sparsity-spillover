"""Overlay retained 14M and 70M loss-latency measurements without rescaling."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "analyses/033-2026-09-21-14m-latency-quality-frontier/data/three-panel.json":
        "1ca8f595566044777198d9281b6b3f8986c9f2943b2d91054404c89030125d56",
    "analyses/034-2026-09-21-70m-latency-quality-frontier/data/figure-data.json":
        "77ea1cb237f078838d6a8eb28123c61226caf5a38a2c409c99849d1d914dbbc7",
    "analyses/027-2026-09-20-run044-manuscript/data/figure-data.json":
        "7259f87a6be42c51d226a18ad4b9e6a895902e4b231bfd6dc1c706916f8c77da",
}
HIGHLIGHTS = [("0", "none"), ("hz", "h"), ("7", "all")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for name, digest in SOURCES.items():
        assert sha(ROOT / name) == digest, name
    small, large, palette = [json.loads((ROOT / name).read_text(encoding="utf-8")) for name in SOURCES]
    cohorts = {"14M": small["points"], "70M": large["points"]}
    metrics = {"14M": "latency_ms", "70M": "displayed_latency_ms"}
    markers = {"14M": "o", "70M": "s"}
    assert [len(cohorts[size]) for size in cohorts] == [40, 26]
    assert len({r["checkpoint_key"] for rows in cohorts.values() for r in rows}) == 66
    for size, rows in cohorts.items():
        for row in rows:
            assert row["model"] == size
            assert row["coverage"]["sequences"] == 338 and row["coverage"]["excluded_tail_tokens"] == 1444
            assert row["pressure"] != "L1" and (row["scope"], row["pressure"]) != ("1", "none")
            if size == "14M":
                assert row["kernel"] == "K050"
            else:
                backend = "native_graph" if row["scope"] == "0" else "candidate_graph"
                assert row["qualified"][backend]
                assert row[metrics[size]] == row["implementation_latency_ms"][backend]
    preserved = {p.name: sha(p) for p in (HERE / "figures").glob("*.pdf")
                 if p.name != "02-14m-70m-latency-quality.pdf"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 11, "axes.titlesize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .65, "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    fig.subplots_adjust(left=.10, right=.985, top=.90, bottom=.23)
    styles = sorted(palette["series"], key=lambda s: (s["scope"], s["pressure"]) in HIGHLIGHTS)
    handles = {}
    drawn = []
    for style in styles:
        key = (style["scope"], style["pressure"])
        highlighted, is_base = key in HIGHLIGHTS, key == ("0", "none")
        for size, rows in cohorts.items():
            points = sorted((r for r in rows if (r["scope"], r["pressure"]) == key),
                            key=lambda r: (r.get("dose") if size == "14M" else r["kappa"]) or 0)
            if not points:
                continue
            drawn.extend(r["checkpoint_key"] for r in points)
            color = style["color"] if highlighted else "#9DA3AB"
            ls = style["linestyle"]
            if isinstance(ls, list):
                ls = (ls[0], tuple(ls[1]))
            width, alpha = (1.6, 1.) if highlighted else (.85, .65)
            marker = dict(marker=markers[size], ms=9 if is_base else 5.8 if highlighted else 4.2,
                          mfc="white" if is_base else color, mec=color if is_base else "white",
                          mew=1.5 if is_base else .7 if highlighted else .5)
            x, y = [r["loss"] for r in points], [r[metrics[size]] for r in points]
            ax.plot(x, y, color=color, ls=ls, lw=width, alpha=alpha,
                    zorder=4 if highlighted else 2)
            ax.plot(x, y, color=color, ls="none", **marker, zorder=5 if highlighted else 3)
            if highlighted:
                label = f"{size} Base ({'K050' if size == '14M' else 'PyTorch'})" if is_base else f"{size} {style['label']}"
                handles[(key, size)] = Line2D([], [], color=color, ls=ls, lw=width,
                                              **marker, label=label)
    assert len(drawn) == len(set(drawn)) == 66
    assert set(drawn) == {r["checkpoint_key"] for rows in cohorts.values() for r in rows}
    for size, rows in cohorts.items():
        base = next(r for r in rows if r["scope"] == "0")
        ax.axvline(base["loss"], color="#92969B", lw=.8, ls=(0, (2, 3)), zorder=1)
        ax.text(base["loss"] + .025, .98, f"{size} Base loss",
                transform=ax.get_xaxis_transform(), color="#747A81", fontsize=8.5,
                ha="left", va="top")
    xlim, ylim = (4.02, 6.10), (.40, 2.85)
    assert all(xlim[0] <= r["loss"] <= xlim[1] and ylim[0] <= r[metrics[size]] <= ylim[1]
               for size, rows in cohorts.items() for r in rows)
    ax.set(xlim=xlim, ylim=ylim, xlabel="Validation loss", ylabel="Full-model latency (ms)")
    ax.set_xticks([4.2, 4.5, 4.8, 5.1, 5.4, 5.7, 6.0])
    ax.set_yticks([.5, 1., 1.5, 2., 2.5])
    ax.set_title("Pythia-14M and Pythia-70M", loc="left", pad=12)
    ax.tick_params(length=3, width=.65)
    ax.grid(axis="y", color="#E8EAED", lw=.6)
    ax.set_axisbelow(True)
    legend = [handles[(key, size)] for key in HIGHLIGHTS for size in cohorts]
    fig.legend(handles=legend, loc="lower center", ncol=3, frameon=False,
               fontsize=10, handlelength=2.2, columnspacing=1.7,
               labelspacing=.9, bbox_to_anchor=(.54, .018))
    output = HERE / "figures/02-14m-70m-latency-quality.pdf"
    fig.savefig(output, bbox_inches="tight", pad_inches=.04,
                metadata={"Title": "Pythia-14M and Pythia-70M: validation loss and full-model latency",
                          "CreationDate": None, "ModDate": None})
    plt.close(fig)
    assert all(sha(HERE / "figures" / name) == digest for name, digest in preserved.items())
    result = dict(
        output=output.relative_to(HERE).as_posix(), output_sha256=sha(output),
        script=Path(__file__).name, script_sha256=sha(Path(__file__)), sources_sha256=SOURCES,
        cohorts=cohorts, latency_fields=metrics, markers=markers,
        x_metric="loss", y_unit="milliseconds", axis_scales={"x": "linear", "y": "linear"},
        xlim=xlim, ylim=ylim, labeled_series=[h.get_label() for h in legend],
        baseline_backends={"14M": "K050 specialized", "70M": "native PyTorch"},
        recipe_backends={"14M": "K050", "70M": "opt073"},
        interpretation="Retained absolute losses and latencies from the two source figures; no normalization or rescaling. This is not a same-kernel or same-session scale comparison.",
        verification={"counts": {s: len(r) for s, r in cohorts.items()},
                      "all_66_points_visible": True, "unchanged_previous_pdfs": preserved})
    (HERE / "data/combined-figure.json").write_text(json.dumps(result, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
    print(json.dumps({"pdf": str(output), "points": 66, "markers": markers}))


if __name__ == "__main__":
    main()
