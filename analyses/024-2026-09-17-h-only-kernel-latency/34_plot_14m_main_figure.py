"""14M-only Figure 1, preserving the measured coordinates of Figure 08."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = HERE / "data/14m-70m-quality-sparsity-latency.json"
OUTPUT = HERE / "figures/22-14m-main-quality-sparsity-latency.pdf"
DATA_OUTPUT = HERE / "data/14m-main-quality-sparsity-latency.json"
TITLE = "Quality, sparsity and latency on Pythia-14M"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = json.loads(SOURCE.read_text(encoding="utf-8"))
    assert sha(HERE / source["output"]) == source["output_sha256"]
    trained = [r for r in source["trained_points"] if r["model"] == "14M"]
    clips = [r for r in source["clipping_points"] if r["model"] == "14M"]
    series = [s for s in source["series"] if s["model"] == "14M"]
    panels = [p for p in source["panels"] if p["model"] == "14M"]
    limits = source["limits"]["14M"]
    ceilings = source["ceilings"]["14M"]
    assert len(trained) == 22 and len(clips) == 20 and len(series) == 6
    assert len(panels) == 2
    assert panels[0]["trained_keys"] == panels[1]["trained_keys"]
    assert panels[0]["clipping_ids"] == panels[1]["clipping_ids"]
    index = {r["checkpoint_key"]: r for r in trained}

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 11, "axes.labelsize": 11,
        "xtick.labelsize": 10, "ytick.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "pdf.fonttype": 42,
        "mathtext.fontset": "dejavusans",
    })
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.85), sharex=True)
    fig.subplots_adjust(left=.09, right=.985, top=.80, bottom=.29, wspace=.30)
    fig.suptitle(TITLE, fontsize=14, y=.975)
    handles = []
    for s in series:
        group = [index[key] for key in s["keys"]]
        control = s["scope"] in ("0", "1")
        if not control:
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

    for scope, color in (("0", series[0]["color"]), ("1", series[1]["color"])):
        group = sorted((p for p in clips if p["scope"] == scope), key=lambda p: p["target"])
        assert [p["target"] for p in group] == [p / 10 for p in range(10)]
        for ax, metric in zip(axes, ("loss", "latency_ms")):
            ax.plot([p["sparsity"] for p in group], [p[metric] for p in group],
                    color=color, lw=1.3, ls=":", zorder=2)
    axes[0].text(4., 5.62, "Post-hoc", rotation=78, rotation_mode="anchor",
                 color="#646970", fontsize=9.5, ha="left", va="bottom")
    axes[1].text(4.5, .803, "Post-hoc", color="#646970", fontsize=9.5, va="center")
    for scope in ("4", "7"):
        ceiling = ceilings[scope]["R_model_max_percent"]
        axes[0].axvline(ceiling, color="#92969B", lw=.8, ls=(0, (2, 3)), alpha=.8, zorder=1)
        axes[0].text(ceiling + .4 if scope == "4" else ceiling - .25, .98,
                     rf"$T_{scope}$ ceiling", transform=axes[0].get_xaxis_transform(),
                     ha="left" if scope == "4" else "right", va="top",
                     fontsize=9.5, color="#6C7177")
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
        outside = [p["id"] for p in clips if not lo <= p[metric] <= hi]
        assert outside == panel["clipping_outside_y"]
    axes[0].set_yticks([5.2, 5.4, 5.6, 5.8, 6.0])
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=11, handlelength=2.3, columnspacing=2.3,
               labelspacing=.7, bbox_to_anchor=(.53, .005))
    fig.savefig(OUTPUT, metadata={"Title": TITLE, "CreationDate": None, "ModDate": None})
    plt.close(fig)
    result = {
        "title": TITLE, "output": OUTPUT.relative_to(HERE).as_posix(),
        "output_sha256": sha(OUTPUT), "script": Path(__file__).name,
        "script_sha256": sha(Path(__file__)),
        "sources_sha256": {SOURCE.relative_to(ROOT).as_posix(): sha(SOURCE)},
        "layout": {"rows": 1, "columns": 2, "width_inches": 8.6, "height_inches": 3.85},
        "limits": limits, "series": series, "panels": panels,
        "trained_points": trained, "clipping_points": clips, "ceilings": ceilings,
        "scope": "14M only; exact subset of Figure 08, no new evaluation or timing.",
        "quality_view_note": "Eight high-loss clipping points extend above panel (a); all records retained.",
        "latency_note": "K050; Run029/Run033 trained timings and Run036 clipping timings, with recurring clipping included.",
    }
    DATA_OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(OUTPUT), "trained": len(trained), "clipping": len(clips)}))


if __name__ == "__main__":
    main()
