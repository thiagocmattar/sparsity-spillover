"""Preserve Figure 1 styling and add the fifth verified T2/Ph endpoint.

Rendering body adapted unchanged from Analysis024/34_plot_14m_main_figure.py.
"""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
OUTPUT = HERE / "figures/01-14m-complete-t2-grid.pdf"
TITLE = "Quality, sparsity and latency on Pythia-14M"
FONT_SCALE = .9


def main():
    path = HERE / "data/figure-data.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    trained, clips, series = (data[k] for k in ["trained_points", "clipping_points", "series"])
    panels, limits, ceilings = (data[k] for k in ["panels", "limits", "ceilings"])
    index = {r["checkpoint_key"]: r for r in trained}
    assert len(index) == 41 and len(series) == 10
    assert all(p["trained_keys"] == list(index) for p in panels)
    assert [r["kappa"] for r in trained if r["scope"] == "hz"] == [0, .01, .05, .1, .5]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11 * FONT_SCALE,
        "axes.titlesize": 11 * FONT_SCALE, "axes.labelsize": 11 * FONT_SCALE,
        "xtick.labelsize": 10 * FONT_SCALE, "ytick.labelsize": 10 * FONT_SCALE,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.85), sharex=True)
    fig.subplots_adjust(left=.08, right=.985, top=.84, bottom=.26, wspace=.26)
    fig.suptitle(TITLE, fontsize=14 * FONT_SCALE, y=.975)
    handles = []
    for s in series:
        group = [index[key] for key in s["keys"]]
        control = s["pressure"] == "none" and s["scope"] in ("0", "1")
        if s["scope"] in ("4", "7"):
            assert [r["kappa"] for r in group] == [0, .01, .05, .1, .5]
        color = s["color"]
        ls = tuple(s["linestyle"]) if isinstance(s["linestyle"], list) else s["linestyle"]
        style = dict(color=color, ls=ls, lw=1.6, marker="o",
                     ms=9 if control else 5.8,
                     mfc="white" if s["scope"] == "0" else color,
                     mec=color if s["scope"] == "0" else "white",
                     mew=1.5 if s["scope"] == "0" else .7)
        for ax, metric in zip(axes, ("loss", "latency_ms")):
            ax.plot([r["sparsity"] for r in group], [r[metric] for r in group],
                    **style, zorder=5 if control else 4)
        handles.append(Line2D([], [], **style, label=s["label"]))

    # Restore the earlier quality frontiers; latency remains trained checkpoints only.
    for scope, color in (("0", series[0]["color"]), ("1", series[1]["color"])):
        group = sorted((p for p in clips if p["scope"] == scope), key=lambda p: p["target"])
        assert [p["target"] for p in group] == [p / 10 for p in range(10)]
        axes[0].plot([p["sparsity"] for p in group], [p["loss"] for p in group],
                     color=color, lw=1.3, ls=":", zorder=2)
    axes[0].text(4., 5.62, "Post-hoc", rotation=78, rotation_mode="anchor",
                 color="#646970", fontsize=9.5 * FONT_SCALE, ha="left", va="bottom")
    for scope in ("2", "4", "7"):
        ceiling = ceilings[scope]["R_model_max_percent"]
        axes[0].axvline(ceiling, color="#92969B", lw=.8, ls=(0, (2, 3)), alpha=.8, zorder=1)
        axes[0].text(ceiling + .4 if scope != "7" else ceiling - .25, .025 if scope == "2" else .98,
                     rf"$T_{scope}$ ceiling", transform=axes[0].get_xaxis_transform(),
                     ha="left" if scope != "7" else "right", va="bottom" if scope == "2" else "top",
                     fontsize=9.5 * FONT_SCALE, color="#6C7177")
    for ax, panel, title, ylabel in zip(
            axes, panels, ("(a) Quality-sparsity trade-off", "(b) Full-model latency"),
            ("Validation loss", "Full-model latency (ms)")):
        metric = panel["metric"]
        ax.set(xlim=limits["sparsity"], ylim=limits[metric], ylabel=ylabel,
               xlabel=r"Model-wide sparsity $S_{\mathrm{model}}$ (%)")
        ax.set_title(title, loc="left", pad=11)
        ax.set_xticks([0, 10, 20, 30])
        ax.tick_params(length=3, width=.65)
        ax.grid(axis="y", color="#E8EAED", lw=.6)
        ax.set_axisbelow(True)
        lo, hi = limits[metric]
        assert all(lo <= r[metric] <= hi for r in trained)
        shown_clips = [p for p in clips if p["id"] in panel["clipping_ids"]]
        assert [p["id"] for p in shown_clips if not lo <= p[metric] <= hi] == panel["clipping_outside_y"]
    axes[0].set_yticks([5.2, 5.4, 5.6, 5.8, 6.0])
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=10.2 * FONT_SCALE, handlelength=2.1, columnspacing=1.4,
               labelspacing=.7, bbox_to_anchor=(.53, .005))
    fig.savefig(OUTPUT, metadata={"Title": TITLE, "CreationDate": None, "ModDate": None})
    plt.close(fig)
    data.update(output=OUTPUT.relative_to(HERE).as_posix(),
                output_sha256=hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
                script=Path(__file__).name, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(OUTPUT), "trained": len(trained)}))


if __name__ == "__main__":
    main()
