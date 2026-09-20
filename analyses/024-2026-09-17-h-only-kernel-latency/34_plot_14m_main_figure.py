"""14M Figure 1: all ten OL1/control recipes from Analysis026, with T2=h,z."""
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
ADDED_SOURCE = ROOT / "analyses/026-2026-09-20-14m-hz-quality-sparsity-latency/data/figure-data.json"
OUTPUT = HERE / "figures/22-14m-main-quality-sparsity-latency.pdf"
DATA_OUTPUT = HERE / "data/14m-main-quality-sparsity-latency.json"
TITLE = "Quality, sparsity and latency on Pythia-14M"
FONT_SCALE = .9


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
    added = json.loads(ADDED_SOURCE.read_text(encoding="utf-8"))
    assert sha(ADDED_SOURCE.parent.parent / added["figure"]) == added["figure_sha256"]
    for path, expected in added["sources_sha256"].items():
        assert sha(ROOT / path) == expected, path
    added_index = {r["checkpoint_key"]: r for r in added["trained_points"]}
    for row in trained:
        assert all(row[field] == added_index[row["checkpoint_key"]][field]
                   for field in ("loss", "sparsity", "latency_ms", "scope", "pressure", "kappa"))
    trained = [dict(added_index[row["checkpoint_key"]]) for row in trained]
    extra_series = []
    for scope, label, color, doses in (
            ("1", r"$T_1/P_h$", "#B9568B", [.05, .1, .5, 1]),
            ("hz", r"$T_2/P_h$", "#173FAD", [0, .01, .05, .1])):
        dose = "local_pressure_weight" if scope == "1" else "kappa"
        group = sorted((r for r in added["trained_points"]
                        if (r["scope"], r["pressure"]) == (scope, "h")), key=lambda r: r[dose])
        assert [r[dose] for r in group] == doses
        trained.extend(group)
        extra_series.append(dict(model="14M", scope=scope, pressure="h", label=label,
                                 color=color, linestyle=[0, [3, 2]], dose_field=dose,
                                 keys=[r["checkpoint_key"] for r in group]))
    series = series[:2] + extra_series + series[2:]
    for scope, color in (("4", "#BA5256"), ("7", "#B19A39")):
        group = sorted((r for r in added["trained_points"]
                        if (r["scope"], r["pressure"]) == (scope, "none")), key=lambda r: r["kappa"])
        assert [r["kappa"] for r in group] == [0, .01, .05, .1, .5]
        trained.extend(group)
        series.append(dict(model="14M", scope=scope, pressure="none", label=rf"$T_{scope}/P_0$",
                           color=color, linestyle="-.", keys=[r["checkpoint_key"] for r in group]))
    # Matplotlib fills legends down columns: the final three columns pair T4/T7
    # under P0, Ph and Pall; the first two hold Base/ReLU and T1/T2.
    order = [("0", "none"), ("1", "none"), ("1", "h"), ("hz", "h"),
             ("4", "none"), ("7", "none"), ("4", "h"), ("7", "h"),
             ("4", "all"), ("7", "all")]
    series.sort(key=lambda s: order.index((s["scope"], s["pressure"])))
    assert len(trained) == 40 and len(series) == 10
    assert len({r["checkpoint_key"] for r in trained}) == 40
    assert {r["checkpoint_key"] for r in trained} == {r["checkpoint_key"] for r in added["trained_points"]}
    for panel in panels:
        panel["trained_keys"] = [r["checkpoint_key"] for r in trained]
    ceilings["2"] = added["ceilings"]["hz"]
    limits["loss"] = [5.06, 6.19]
    index = {r["checkpoint_key"]: r for r in trained}

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

    for scope, color in (("0", series[0]["color"]), ("1", series[1]["color"])):
        group = sorted((p for p in clips if p["scope"] == scope), key=lambda p: p["target"])
        assert [p["target"] for p in group] == [p / 10 for p in range(10)]
        for ax, metric in zip(axes, ("loss", "latency_ms")):
            ax.plot([p["sparsity"] for p in group], [p[metric] for p in group],
                    color=color, lw=1.3, ls=":", zorder=2)
    axes[0].text(4., 5.62, "Post-hoc", rotation=78, rotation_mode="anchor",
                 color="#646970", fontsize=9.5 * FONT_SCALE, ha="left", va="bottom")
    axes[1].text(4.5, .803, "Post-hoc", color="#646970", fontsize=9.5 * FONT_SCALE, va="center")
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
        outside = [p["id"] for p in clips if not lo <= p[metric] <= hi]
        assert outside == panel["clipping_outside_y"]
    axes[0].set_yticks([5.2, 5.4, 5.6, 5.8, 6.0])
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               fontsize=10.2 * FONT_SCALE, handlelength=2.1, columnspacing=1.4,
               labelspacing=.7, bbox_to_anchor=(.53, .005))
    fig.savefig(OUTPUT, metadata={"Title": TITLE, "CreationDate": None, "ModDate": None})
    plt.close(fig)
    result = {
        "title": TITLE, "output": OUTPUT.relative_to(HERE).as_posix(),
        "output_sha256": sha(OUTPUT), "script": Path(__file__).name,
        "script_sha256": sha(Path(__file__)),
        "sources_sha256": {SOURCE.relative_to(ROOT).as_posix(): sha(SOURCE),
                           ADDED_SOURCE.relative_to(ROOT).as_posix(): sha(ADDED_SOURCE)},
        "layout": {"rows": 1, "columns": 2, "width_inches": 8.6, "height_inches": 3.85,
                   "legend_rows": 2, "legend_columns": 5, "font_scale": FONT_SCALE,
                   "axes_bounds": [ax.get_position().bounds for ax in axes]},
        "limits": limits, "series": series, "panels": panels,
        "trained_points": trained, "clipping_points": clips, "ceilings": ceilings,
        "scope": "14M only; all40 endpoints and ten recipes from Analysis026. Figure22 retains both measured post-hoc clipping latency paths. No new evaluation or timing.",
        "display_nomenclature": {"T2": "One-sided gates at h,z and OL1 pressure at h; raw scope hz. This is not the operational A2=m,h topology."},
        "dose_note": "T1/Ph varies lambda=.05,.1,.5,1; T2/Ph varies kappa=0,.01,.05,.1; T4/T7 vary kappa=0,.01,.05,.1,.5.",
        "quality_view_note": "Eight high-loss clipping points extend above panel (a); all records retained.",
        "latency_note": "K050; Run029/Run033/Run041 trained timings and Run036 clipping timings, with recurring clipping included. Different physical GPU/host sessions; small cross-session latency differences are descriptive.",
    }
    DATA_OUTPUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"output": str(OUTPUT), "trained": len(trained), "clipping": len(clips)}))


if __name__ == "__main__":
    main()
