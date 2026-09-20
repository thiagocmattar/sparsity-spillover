"""Preserve the two-panel design and add the complete five-point T2/Ph series."""
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
DATA = HERE / "data/70m-quality-sparsity-native-latency.json"
OUTPUT = HERE / "figures/23-70m-quality-sparsity-native-latency.pdf"
FONT_SCALE = .9


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    data = json.loads(DATA.read_text(encoding="utf-8"))
    plotted, series = data["plotted_points"], data["series"]
    native, clips = data["native_base_reference"], data["clipping_points"]
    limits, TITLE = data["limits"], data["title"]
    index = {r["checkpoint_key"]: r for r in plotted}
    assert len(index) == 27 and len(series) == 7
    OUTPUT.parent.mkdir(exist_ok=True)
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
    for scope in ("hz", "4", "7"):
        ceiling = data["ceilings"][scope]["R_model_max_percent"]
        axes[0].axvline(ceiling, color="#92969B", lw=.8, ls=(0, (2, 3)), alpha=.8, zorder=1)
        axes[0].text(ceiling - .25, .98, rf"$T_{{{2 if scope == 'hz' else scope}}}$ ceiling",
                     transform=axes[0].get_xaxis_transform(), ha="right", va="top",
                     fontsize=9.5 * FONT_SCALE, color="#6C7177")
    axes[1].axhline(native["native_ms"], color="#52565C", lw=.9, ls=":", zorder=1)
    axes[1].text(2, 1.60, f"PyTorch base: {native['native_ms']:.3f} ms",
                 color="#52565C", fontsize=9.5 * FONT_SCALE, ha="left", va="top")
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
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False,
               fontsize=10.2 * FONT_SCALE, handlelength=2.1, columnspacing=1.5,
               labelspacing=.7, bbox_to_anchor=(.53, .005))
    fig.savefig(OUTPUT, metadata={"Title": TITLE, "CreationDate": None, "ModDate": None})
    plt.close(fig)

    data.update(output=OUTPUT.relative_to(HERE).as_posix(), output_sha256=sha(OUTPUT),
                script=Path(__file__).name, script_sha256=sha(Path(__file__)))
    DATA.write_text(json.dumps(data, indent=2)+"\n", encoding="utf-8", newline="\n")
    print(json.dumps({"pdf":str(OUTPUT), "sha256":data["output_sha256"]}))


if __name__ == "__main__":
    main()
