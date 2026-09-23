"""Overlay retained 14M/70M measurements and both Base backends on log axes."""
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter
import recent_runs

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCES = {
    "analyses/033-2026-09-21-14m-latency-quality-frontier/data/three-panel.json":
        "1ca8f595566044777198d9281b6b3f8986c9f2943b2d91054404c89030125d56",
    "analyses/034-2026-09-21-70m-latency-quality-frontier/data/figure-data.json":
        "77ea1cb237f078838d6a8eb28123c61226caf5a38a2c409c99849d1d914dbbc7",
    "analyses/027-2026-09-20-run044-manuscript/data/figure-data.json":
        "7259f87a6be42c51d226a18ad4b9e6a895902e4b231bfd6dc1c706916f8c77da",
    "analyses/024-2026-09-17-h-only-kernel-latency/data/results.json":
        "f4bddaebcf23ba1118b487e951cd89c140149c2eab0dc46ccbc261793fcebd9a",
}
HIGHLIGHTS = [("hz", "h"), ("7", "all")]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    for name, digest in SOURCES.items():
        assert sha(ROOT / name) == digest, name
    small, large, palette, timing14 = [json.loads((ROOT / name).read_text(encoding="utf-8")) for name in SOURCES]
    cohorts = {"14M": small["points"], "70M": large["points"]}
    additions = recent_runs.load(cohorts['70M'])
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
    bases = {size: next(r for r in rows if r["scope"] == "0") for size, rows in cohorts.items()}
    base14, = [r for r in timing14["points"] if r["family"] == "A0"]
    assert base14["qualified"] and base14["session"] == bases["14M"]["timing_session"]
    assert base14["k050_gm_ms"] == bases["14M"]["latency_ms"]
    assert all(f["path"].startswith(bases["14M"]["source_attempt"] + "/") for f in base14["checkpoint_files"])
    assert len(base14["replicates"]) == 3 and all(r["qualified"] for r in base14["replicates"])
    for field in ("native_gm_ms", "k050_gm_ms"):
        mean = math.exp(math.fsum(math.log(r[field]) for r in base14["replicates"]) / 3)
        assert math.isclose(mean, base14[field], rel_tol=1e-12)
    assert bases["70M"]["qualified"]["candidate_graph"] and bases["70M"]["qualified"]["native_graph"]
    references = []
    for size, base in bases.items():
        for backend in ("kernel", "PyTorch"):
            field = ("k050_gm_ms" if backend == "kernel" else "native_gm_ms") if size == "14M" else (
                "candidate_graph" if backend == "kernel" else "native_graph")
            latency = base14[field] if size == "14M" else base["implementation_latency_ms"][field]
            references.append(dict(model=size, backend=backend, loss=base["loss"], latency_ms=latency,
                                   checkpoint_key=base["checkpoint_key"], timing_session=base["timing_session"],
                                   implementation=("K050" if size == "14M" else "opt073") if backend == "kernel" else "native_graph",
                                   source_field=field))
    preserved = {p.name: sha(p) for p in (HERE / "figures").glob("*.pdf")
                 if p.name != "02-14m-70m-latency-quality.pdf"}
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.labelsize": 11, "axes.titlesize": 12,
                         "xtick.labelsize": 10, "ytick.labelsize": 10,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.linewidth": .65, "pdf.fonttype": 42,
                         "mathtext.fontset": "dejavusans"})
    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    fig.subplots_adjust(left=.095, right=.985, top=.92, bottom=.385)
    styles = sorted(palette["series"], key=lambda s: (s["scope"], s["pressure"]) in HIGHLIGHTS)
    handles = {}
    drawn = []
    for style in styles:
        key = (style["scope"], style["pressure"])
        if key == ("0", "none"):
            continue
        highlighted = key in HIGHLIGHTS
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
            marker = dict(marker=markers[size], ms=5.8 if highlighted else 4.2,
                          mfc=color, mec="white", mew=.7 if highlighted else .5)
            x, y = [r["loss"] for r in points], [r[metrics[size]] for r in points]
            ax.plot(x, y, color=color, ls=ls, lw=width, alpha=alpha,
                    zorder=4 if highlighted else 2)
            ax.plot(x, y, color=color, ls="none", **marker, zorder=5 if highlighted else 3)
            if highlighted:
                label = f"{size} {style['label']}"
                handles[(key, size)] = Line2D([], [], color=color, ls=ls, lw=width,
                                              **marker, label=label)
    assert len(drawn) == len(set(drawn)) == 64
    assert set(drawn) == {r["checkpoint_key"] for rows in cohorts.values() for r in rows if r["scope"] != "0"}
    base_handles = {}
    for reference in references:
        size, backend = reference["model"], reference["backend"]
        kernel = backend == "kernel"
        # Smaller filled PyTorch markers keep both nearly coincident 14M references visible.
        marker = dict(marker=markers[size], ms=10 if kernel else 5.5,
                      mfc="white" if kernel else "#52565C", mec="#52565C",
                      mew=1.5 if kernel else .7)
        ax.plot(reference["loss"], reference["latency_ms"], color="#52565C", ls="none",
                **marker, zorder=6 if kernel else 7)
        base_handles[(backend, size)] = Line2D([], [], color="#52565C", ls="none", **marker,
                                               label=f"{size} Base ({backend})")
    for size, rows in cohorts.items():
        base = next(r for r in rows if r["scope"] == "0")
        ax.axvline(base["loss"], color="#92969B", lw=.8, ls=(0, (2, 3)), zorder=1)
        ax.text(base["loss"] + .025, .98, f"{size} Base loss",
                transform=ax.get_xaxis_transform(), color="#747A81", fontsize=8.5,
                ha="left", va="top")
    colors = {'T2/Ph': next(s['color'] for s in styles if (s['scope'], s['pressure']) == ('hz', 'h')),
              'T7/Pall': next(s['color'] for s in styles if (s['scope'], s['pressure']) == ('7', 'all'))}
    added_handles = recent_runs.draw(ax, additions, colors)
    xlim, ylim = (4.02, 6.10), (.40, 2.85)
    assert all(xlim[0] <= r["loss"] <= xlim[1] and ylim[0] <= r[metrics[size]] <= ylim[1]
               for size, rows in cohorts.items() for r in rows)
    assert all(xlim[0] <= r["loss"] <= xlim[1] and ylim[0] <= r["latency_ms"] <= ylim[1]
               for r in references)
    assert all(xlim[0] <= r['loss'] <= xlim[1] and ylim[0] <= r['latency_ms'] <= ylim[1]
               for r in additions['points'])
    ax.set(xlim=xlim, ylim=ylim, xlabel="Validation loss", ylabel="Full-model latency (ms)")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.set_xticks([4.2, 4.5, 4.8, 5.1, 5.4, 5.7, 6.0])
    ax.set_yticks([.5, 1., 1.5, 2., 2.5])
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(ScalarFormatter())
        axis.set_minor_locator(NullLocator())
        axis.set_minor_formatter(NullFormatter())
    ax.set_title("Pythia-14M and Pythia-70M", loc="left", pad=12)
    ax.tick_params(length=3, width=.65)
    ax.grid(axis="y", color="#E8EAED", lw=.6)
    ax.set_axisbelow(True)
    legend = [base_handles[(backend, size)] for backend in ("kernel", "PyTorch") for size in cohorts]
    legend += [handles[(key, size)] for key in HIGHLIGHTS for size in cohorts]
    old_legend = fig.legend(handles=legend, loc="lower center", ncol=4, frameon=False,
               fontsize=8.5, handlelength=2., columnspacing=1.4,
               labelspacing=.8, bbox_to_anchor=(.54, .19),
               title='Existing measurements: K050 (14M), opt073 (70M)', title_fontsize=9)
    new_legend = fig.legend(handles=added_handles, loc='lower center', ncol=3, frameon=False,
               fontsize=8.5, handlelength=2., columnspacing=2., labelspacing=.7,
               bbox_to_anchor=(.54, .055), title='Added 70M measurements: C = sparse h in five layers, dense z',
               title_fontsize=9)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    assert not ax.xaxis.label.get_window_extent(renderer).overlaps(old_legend.get_window_extent(renderer))
    assert not old_legend.get_window_extent(renderer).overlaps(new_legend.get_window_extent(renderer))
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
        x_metric="loss", y_unit="milliseconds", axis_scales={"x": "log", "y": "log"},
        xlim=xlim, ylim=ylim, labeled_series=[h.get_label() for h in legend],
        base_references=references,
        added_measurements=additions,
        added_legend=[h.get_label() for h in added_handles],
        recipe_backends={"14M": "K050", "70M": "opt073"},
        interpretation="All 68 original execution points retained, plus 18 Run051/052 measurements (two unqualified). Absolute measurements on logarithmic axes; no session normalization, pooling or inferred full-model timings. This is not a same-kernel or same-session scale comparison.",
        verification={"counts": {s: len(r) for s, r in cohorts.items()},
                      "unique_checkpoints": 66, "intervention_points": 64, "base_execution_points": 4,
                      "all_68_execution_points_visible": True, 'added_execution_points': 18,
                      'added_qualified_points': 16, 'added_unqualified_points': 2,
                      'all_86_execution_points_visible': True, "unchanged_previous_pdfs": preserved})
    (HERE / "data/combined-figure.json").write_text(json.dumps(result, indent=2) + "\n",
                                                  encoding="utf-8", newline="\n")
    print(json.dumps({"pdf": str(output), "execution_points": 86, 'added_qualified': 16,
                      'added_unqualified': 2, 'verified_raw_sources': len(additions['raw_evidence_sha256'])}))


if __name__ == "__main__":
    main()
