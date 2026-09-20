"""70M Figure 3: canonical quality, original-port latency, native base reference."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN040 = ROOT / "runs/040-2026-09-20-pythia70m-kernel-overhead"
OUTPUT = HERE / "figures/23-70m-quality-sparsity-native-latency.pdf"
DATA = HERE / "data/70m-quality-sparsity-native-latency.json"
TITLE = "Quality, sparsity and latency on Pythia-70M"
FONT_SCALE = .9


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, sources):
    sources[path.relative_to(ROOT).as_posix()] = sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    sources = {}
    original = read(HERE / "data/14m-70m-quality-sparsity-latency.json", sources)
    audit = read(HERE / "data/kernel-appendix.json", sources)
    style = read(HERE / "data/14m-main-quality-sparsity-latency.json", sources)
    for parent in (original, audit, style):
        for path, digest in parent["sources_sha256"].items():
            assert sha(ROOT / path) == digest, path
    trained = [r for r in original["trained_points"] if r["model"] == "70M"]
    clips = [r for r in original["clipping_points"] if r["model"] == "70M"]
    native = audit["base_references"]["70M"]
    assert len(trained) == 22 and len(clips) == 20
    assert len({r["checkpoint_key"] for r in trained}) == 22
    base, = [r for r in trained if r["scope"] == "0"]
    assert base["checkpoint_key"] == native["checkpoint_key"]
    assert base["latency_ms"] == native["specialized_ms"]
    assert {r["timing_device_uuid"] for r in trained} == {native["device_uuid"]}
    reference = {r["checkpoint_key"]: r for r in audit["points"]
                 if r["model"] == "70M" and r["kind"] == "trained"}
    for row in trained:
        checked = reference[row["checkpoint_key"]]
        assert row["latency_ms"] == checked["latency_ms"]
        assert row["loss"] == checked["loss"]
        assert row["sparsity"] == checked["model_sparsity_percent"]
    # Retain the source latency unchanged, and explicitly export the plotted backend.
    plotted = [dict(r, displayed_latency_ms=native["native_ms"] if r["scope"] == "0"
                    else r["latency_ms"], execution="native PyTorch/SDPA" if r["scope"] == "0"
                    else "original 70M port") for r in trained]
    series = [dict(s) for s in original["series"] if s["model"] == "70M"]
    order = [("0", "none"), ("1", "none"), ("4", "h"), ("7", "h"),
             ("4", "all"), ("7", "all")]
    series.sort(key=lambda s: order.index((s["scope"], s["pressure"])))
    style_index = {(s["scope"], s["pressure"]): s for s in style["series"]}
    for s in series:
        prior = style_index[(s["scope"], s["pressure"])]
        assert all(s[k] == prior[k] for k in ("color", "linestyle", "label"))
    index = {r["checkpoint_key"]: r for r in plotted}
    limits = {"sparsity": [-1.25, 51], "loss": [4., 5.58], "latency_ms": [1.48, 3.43]}

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11 * FONT_SCALE,
        "axes.titlesize": 11 * FONT_SCALE, "axes.labelsize": 11 * FONT_SCALE,
        "xtick.labelsize": 10 * FONT_SCALE, "ytick.labelsize": 10 * FONT_SCALE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans",
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.85), sharex=True)
    fig.subplots_adjust(left=.08, right=.985, top=.84, bottom=.26, wspace=.26)
    fig.suptitle(TITLE, fontsize=14 * FONT_SCALE, y=.975)
    handles = []
    for s in series:
        group = [index[key] for key in s["keys"]]
        control = s["scope"] in ("0", "1")
        if not control:
            assert [r["kappa"] for r in group] == [0, .01, .05, .1, .5]
        color = s["color"]
        ls = tuple(s["linestyle"]) if isinstance(s["linestyle"], list) else s["linestyle"]
        line_style = dict(color=color, ls=ls, lw=1.6, marker="o", ms=9 if control else 5.8,
                          mfc="white" if s["scope"] == "0" else color,
                          mec=color if s["scope"] == "0" else "white",
                          mew=1.5 if s["scope"] == "0" else .7)
        for ax, metric in zip(axes, ("loss", "displayed_latency_ms")):
            ax.plot([r["sparsity"] for r in group], [r[metric] for r in group],
                    **line_style, zorder=5 if control else 4)
        handles.append(Line2D([], [], **line_style, label=s["label"]))
    for scope, color in (("0", series[0]["color"]), ("1", series[1]["color"])):
        group = sorted((r for r in clips if r["scope"] == scope), key=lambda r: r["target"])
        assert [r["target"] for r in group] == [p / 10 for p in range(10)]
        axes[0].plot([r["sparsity"] for r in group], [r["loss"] for r in group],
                     color=color, ls=":", lw=1.3, zorder=2)
    axes[0].text(15.5, 4.72, "Post-hoc", rotation=68, rotation_mode="anchor",
                 color="#646970", fontsize=9.5 * FONT_SCALE, ha="left", va="bottom")
    for scope in ("4", "7"):
        ceiling = original["ceilings"]["70M"][scope]["R_model_max_percent"]
        axes[0].axvline(ceiling, color="#92969B", lw=.8, ls=(0, (2, 3)), alpha=.8, zorder=1)
        axes[0].text(ceiling - .25, .98, rf"$T_{scope}$ ceiling",
                     transform=axes[0].get_xaxis_transform(), ha="right", va="top",
                     fontsize=9.5 * FONT_SCALE, color="#6C7177")
    axes[1].axhline(native["native_ms"], color="#52565C", lw=.9, ls=":", zorder=1)
    axes[1].text(2, 1.72, f"PyTorch base: {native['native_ms']:.3f} ms",
                 color="#52565C", fontsize=9.5 * FONT_SCALE, ha="left", va="bottom")
    for ax, metric, title, ylabel in zip(axes, ("loss", "latency_ms"),
            ("(a) Quality-sparsity trade-off", "(b) Full-model latency"),
            ("Validation loss", "Full-model latency (ms)")):
        ax.set(xlim=limits["sparsity"], ylim=limits[metric], ylabel=ylabel,
               xlabel=r"Model-wide sparsity $S_{\mathrm{model}}$ (%)")
        ax.set_title(title, loc="left", pad=11)
        ax.set_xticks([0, 10, 20, 30, 40, 50])
        ax.tick_params(length=3, width=.65)
        ax.grid(axis="y", color="#E8EAED", lw=.6)
        ax.set_axisbelow(True)
        field = "loss" if metric == "loss" else "displayed_latency_ms"
        assert all(limits[metric][0] <= r[field] <= limits[metric][1] for r in plotted)
    axes[1].set_yticks([1.5, 2., 2.5, 3.])
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=10.2 * FONT_SCALE, handlelength=2.1, columnspacing=2.1,
               labelspacing=.7, bbox_to_anchor=(.53, .005))
    fig.savefig(OUTPUT, metadata={"Title": TITLE, "CreationDate": None, "ModDate": None})
    plt.close(fig)

    # The follow-up is reported separately, never spliced into the original-session curves.
    followup = read(RUN040 / "results/component-overhead.json", sources)
    qualification = read(RUN040 / "results/closeout-verification.json", sources)
    assert followup["status"] == "complete" and qualification["status"] == "passed"
    for item in followup["sources"]:
        assert sha(RUN040 / item["path"]) == item["sha256"]
    for q in qualification["final_quality"]:
        raw = json.loads((RUN040 / q["result"]["path"]).read_text())
        assert raw["qualified"] and raw["validation_blocks"] == 338
    nonbase = [r for r in trained if r["scope"] != "0"]
    result = {
        "title": TITLE, "output": OUTPUT.relative_to(HERE).as_posix(),
        "output_sha256": sha(OUTPUT), "script": Path(__file__).name,
        "script_sha256": sha(Path(__file__)), "sources_sha256": sources,
        "trained_points": trained, "plotted_points": plotted, "clipping_points": clips,
        "series": series, "native_base_reference": native, "limits": limits,
        "layout": {"rows": 1, "columns": 2, "width_inches": 8.6, "height_inches": 3.85,
                   "legend_rows": 2, "legend_columns": 3, "font_scale": FONT_SCALE},
        "panels": [{"panel": "a", "trained_keys": list(index),
                    "clipping_ids": [r["id"] for r in clips],
                    "clipping_outside_y": [r["id"] for r in clips if r["loss"] > limits["loss"][1]]},
                   {"panel": "b", "trained_keys": list(index), "clipping_ids": []}],
        "summary": {"ported_base_over_native_ratio": base["latency_ms"] / native["native_ms"],
                    "best_original_port_native_base_speedup": max(native["native_ms"] / r["latency_ms"] for r in nonbase),
                    "original_port_points_below_native_base": sum(r["latency_ms"] < native["native_ms"] for r in nonbase)},
        "followup_not_plotted": {"native_base_ms": followup["native_base_reference_ms"],
            "device_uuid": followup["device_uuid"],
            "rows": [r for r in followup["rows"] if r["mode"] in ("full", "opt002", "native-hz")]},
        "coverage": {"validation_blocks": 338, "validation_documents": 500,
                     "excluded_tail_tokens": 1444, "timing_inputs": 64, "passes": 7, "processes": 3},
        "note": "Only the Base model marker in (b) uses native PyTorch/SDPA; all other latency markers use the original 70M port. All 70M trained timings share Run035's GPU. Post-hoc paths appear only in (a). Run040 optimization results remain separate; no Run042 development results are used.",
    }
    DATA.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"pdf": str(OUTPUT), "summary": result["summary"]}))


if __name__ == "__main__":
    main()
