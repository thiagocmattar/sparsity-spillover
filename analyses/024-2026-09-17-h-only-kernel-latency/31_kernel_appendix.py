"""Connect retained instruction counts to speedup over one native base per size."""
import hashlib
import json
import math
from importlib import import_module
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COUNTS = HERE / "data/operation-bypass.json"
SPEED = HERE / "data/14m-70m-sparsity-base-speedup.json"
COHORT = HERE / "data/paper-checkpoints.json"
OUTPUT = HERE / "figures/20-kernel-structure-native-base-speedup.pdf"
DATA = HERE / "data/kernel-appendix.json"
PROJECTION = ("qkv_projection", "mlp_w1", "mlp_w2", "attention_output_projection")
ATTENTION = ("qk_scores", "probability_value")
STYLES = import_module("16_plot_matched_quality_latency").STYLES


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path, sources):
    sources[path.relative_to(ROOT).as_posix()] = sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def pool(operations, names):
    rows = [operations[n] for n in names]
    issued = sum(r["issued_mmas"] for r in rows)
    bypassed = sum(r["bypassed_mmas"] for r in rows)
    potential = sum(r["potential_mmas"] for r in rows)
    assert potential == issued + bypassed > 0
    products = sum(r["logical_products"] for r in rows)
    zeros = sum(r["bf16_activation_zero_products_lower_bound"] for r in rows)
    return {"issued_mmas": issued, "bypassed_mmas": bypassed,
            "potential_mmas": potential, "bypass_percent": 100 * bypassed / potential,
            "simt_products": sum(r["simt_products"] for r in rows),
            "scalar_products": products, "scalar_zero_products": zeros,
            "scalar_zero_percent": 100 * zeros / products}


def base_times(row, counter, sources):
    folder = ROOT / Path(counter["diagnostic_source"]).parent
    values = {"native_graph": [], "candidate_graph": []}
    for rep in (1, 2, 3):
        attempt = folder.with_name(folder.name.replace("r1-", f"r{rep}-"))
        result = read(attempt / "result.json", sources)
        assert result["qualified"] and result["validation_blocks"] == 338
        weight = next(f for f in result["checkpoint"]["files"]
                      if Path(f["path"]).name == "model.safetensors")
        assert weight["sha256"] == counter["weight_sha256"]
        timing = read(attempt / "timing.json", sources)
        assert timing["indices"] == row["timing_indices"]
        for mode in values:
            records = [r for r in timing["samples"] if r["mode"] == mode]
            assert len(records) == 448
            assert {(r["input_index"], r["repeat"]) for r in records} == {
                (i, p) for i in range(64) for p in range(7)}
            assert all(r["output_shape"] == [1, 2048, 50304] for r in records)
            values[mode].extend(r["host_ms"] for r in records)
    means = {mode: math.exp(math.fsum(map(math.log, times)) / len(times))
             for mode, times in values.items()}
    assert math.isclose(means["native_graph"], row["native_latency_ms"], rel_tol=1e-12)
    assert math.isclose(means["candidate_graph"], row["latency_ms"], rel_tol=1e-12)
    return {"checkpoint_key": row["checkpoint_key"], "native_ms": means["native_graph"],
            "specialized_ms": means["candidate_graph"], "session": row["timing_session"],
            "device_uuid": row["timing_device_uuid"], "samples_per_implementation": 1344}


def reduce_saved():
    sources = {}
    counts, speed, cohort = [read(p, sources) for p in (COUNTS, SPEED, COHORT)]
    # Check the primary diagnostics/timing records used by the retained reduction.
    for relative, digest in counts["sources_sha256"].items():
        assert sha(ROOT / relative) == digest, relative
    trained = {r["checkpoint_key"]: r for r in cohort["checkpoints"]}
    counters = {(r["kind"], r["checkpoint_key"], r["p"]): r for r in counts["settings"]}
    assert len(counters) == len(counts["settings"])
    bases = {}
    for size in ("14M", "70M"):
        base, = [r for r in trained.values() if r["model"] == size and r["scope"] == "0"]
        bases[size] = base_times(base, counters[("trained", base["checkpoint_key"], None)], sources)
    points = []
    for point in speed["points"]:
        kind = "clipping" if point["kind"] == "posthoc" else "trained"
        key = (kind, point["checkpoint_key"], point.get("target"))
        counter = counters[key]
        assert counter["model"] == point["model"]
        assert math.isclose(counter["latency_ms"], point["latency_ms"], rel_tol=1e-12)
        assert math.isclose(100 * counter["fp16_model_sparsity_fraction"], point["sparsity"], abs_tol=1e-10)
        assert counter["weight_sha256"] == trained[point["checkpoint_key"]]["checkpoint_files"]["model.safetensors"]["sha256"]
        base = bases[point["model"]]
        row = {"id": counter["setting_id"], "kind": kind, "model": point["model"],
               "checkpoint_key": point["checkpoint_key"], "weight_sha256": counter["weight_sha256"],
               "scope": counter["scope"], "pressure": counter["pressure"],
               "kappa": counter["kappa"], "p": counter["p"], "latency_ms": point["latency_ms"],
               "loss": point["loss"], "model_sparsity_percent": point["sparsity"],
               "native_base_ms": base["native_ms"], "specialized_base_ms": base["specialized_ms"],
               "native_base_speedup": base["native_ms"] / point["latency_ms"],
               "specialized_base_speedup": base["specialized_ms"] / point["latency_ms"],
               "projection": pool(counter["operations"], PROJECTION),
               "attention": pool(counter["operations"], ATTENTION),
               "operations": counter["operations"], "session": counter["session"],
               "diagnostic_source": counter["diagnostic_source"]}
        assert math.isclose(row["specialized_base_speedup"], point["speedup"], rel_tol=1e-12)
        points.append(row)
    assert len(points) == len({r["id"] for r in points}) == 84
    summary = {}
    for size in bases:
        rows = [r for r in points if r["model"] == size]
        trained_rows = [r for r in rows if r["kind"] == "trained"]
        assert len(trained_rows) == 22 and len(rows) == 42
        for group in ("projection", "attention"):
            assert len({r[group]["potential_mmas"] for r in rows}) == 1
        predictors = {"model_sparsity": [r["model_sparsity_percent"] for r in trained_rows],
                      "projection_bypass": [r["projection"]["bypass_percent"] for r in trained_rows],
                      "attention_bypass": [r["attention"]["bypass_percent"] for r in trained_rows]}
        yy = [r["native_base_speedup"] for r in trained_rows]
        summary[size] = {"max_native_base_speedup": max(yy),
                         "max_specialized_base_speedup": max(r["specialized_base_speedup"] for r in trained_rows),
                         "trained_faster_than_native_base": sum(y > 1 for y in yy),
                         "descriptive_pearson_r": {k: float(np.corrcoef(x, yy)[0, 1]) for k, x in predictors.items()}}
    return {"base_references": bases, "points": points, "summary": summary,
            "sources_sha256": sources, "verified_primary_sources": len(counts["sources_sha256"]),
            "coverage": {"trained_per_size": 22, "clipping_per_size": 20,
                         "counter_blocks": 338, "timing_inputs": 64, "timing_passes": 7, "processes": 3},
            "definition": "Fixed same-size native T0/P0 geometric-mean latency / specialized setting latency",
            "limits": ["No fit or causal claim; one training seed per trained setting.",
                       "Projection bypass includes scalar substitution and padded MMA rows.",
                       "Attention bypass includes causal/padded work; no model-wide instruction average.",
                       "Projection scalar counts are BF16; canonical model-wide sparsity remains FP16.",
                       "Full-validation counters and 64-input timing have different coverage.",
                       "Timing sessions and the optimization budgets differ across sizes."]}


def make_figure(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
        "axes.titlesize": 11.5, "axes.labelsize": 11, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .65, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
    fig, axes = plt.subplots(2, 3, figsize=(12.8, 7.65))
    fig.subplots_adjust(left=.065, right=.987, top=.905, bottom=.155, hspace=.49, wspace=.34)
    fig.suptitle("Scalar sparsity, instruction bypass and base-model speedup", fontsize=15, y=.98)
    for i, size in enumerate(("14M", "70M")):
        rows = [r for r in data["points"] if r["model"] == size]
        for j, ax in enumerate(axes[i]):
            def xy(group):
                if j == 0:
                    return ([r["projection"]["scalar_zero_percent"] for r in group],
                            [r["projection"]["bypass_percent"] for r in group])
                key = "projection" if j == 1 else "attention"
                return ([r[key]["bypass_percent"] for r in group],
                        [r["native_base_speedup"] for r in group])
            for scope, pressure, label, color, ls in STYLES:
                group = sorted((r for r in rows if r["kind"] == "trained"
                                and (r["scope"], r["pressure"]) == (scope, pressure)),
                               key=lambda r: -1 if r["kappa"] is None else r["kappa"])
                assert len(group) == (1 if scope in ("0", "1") else 5)
                ax.plot(*xy(group), color=color, ls=ls, lw=1.35, marker="o",
                        ms=7 if scope in ("0", "1") else 4.8,
                        mfc="white" if scope == "0" else color,
                        mec=color if scope == "0" else "white", mew=1 if scope == "0" else .5, zorder=4)
            for scope, color in (("0", STYLES[0][3]), ("1", STYLES[1][3])):
                group = sorted((r for r in rows if r["kind"] == "clipping" and r["scope"] == scope), key=lambda r: r["p"])
                assert len(group) == 10
                ax.plot(*xy(group), color=color, ls=":", lw=1.15, marker=".", ms=3, alpha=.8, zorder=2)
            if j == 0:
                ax.set(xlabel="Projection scalar sparsity (%)", ylabel="Projection MMA bypass (%)",
                       xlim=(-3, 103), ylim=(-3, 103), xticks=(0, 25, 50, 75, 100), yticks=(0, 25, 50, 75, 100))
            else:
                ax.axhline(1, color="#8D9298", lw=.8, zorder=1)
                ax.set(xlabel=("Projection" if j == 1 else "Attention") + " MMA bypass (%)",
                       ylabel=r"Speedup over native base ($\times$)",
                       ylim=(.78, 1.49) if size == "14M" else (.42, 1.09))
                if j == 1: ax.set(xlim=(-3, 103), xticks=(0, 25, 50, 75, 100))
                else: ax.set(xlim=(-1, 67) if size == "14M" else (0, 9),
                             xticks=(0, 20, 40, 60) if size == "14M" else (0, 2, 4, 6, 8))
            title = ("Sparsity and bypass", "Projection bypass and speedup", "Attention bypass and speedup")[j]
            ax.set_title(f"({chr(97+3*i+j)}) {size}: {title}", loc="left", pad=9)
            ax.grid(axis="y", color="#E8EAED", lw=.6)
            ax.set_axisbelow(True)
    handles = [Line2D([], [], color=c, ls=ls, marker="o", ms=5, lw=1.35,
               mfc="white" if s == "0" else c, label=label)
               for s, _, label, c, ls in STYLES]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=11, handlelength=2.3, columnspacing=2.3, bbox_to_anchor=(.5, .016))
    fig.text(.5, .003, "Dotted paths: post-hoc clipping of the base and ReLU models", ha="center", fontsize=9.5, color="#60656A")
    return fig


def main():
    data = reduce_saved()
    fig = make_figure(data)
    fig.savefig(OUTPUT, metadata={"Title": "Kernel structure and fixed-native-base speedup", "CreationDate": None, "ModDate": None})
    plt.close(fig)
    data.update(script=Path(__file__).name, script_sha256=sha(Path(__file__)),
                output=OUTPUT.relative_to(ROOT).as_posix(), output_sha256=sha(OUTPUT))
    DATA.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({"summary": data["summary"], "base_references": data["base_references"],
                      "verified_primary_sources": data["verified_primary_sources"]}, indent=2))


if __name__ == "__main__":
    main()
