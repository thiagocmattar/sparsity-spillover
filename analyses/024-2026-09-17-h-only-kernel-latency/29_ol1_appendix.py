"""Retained OL1 geometry for Figure 1, and the matched T1/P1 L1 comparison."""

import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
COHORT = HERE / "data/paper-checkpoints.json"
FIGURE1 = HERE / "data/14m-70m-quality-sparsity-latency.json"
OUTPUT = HERE / "figures/18-14m-70m-ol1-geometry.pdf"
DATA = HERE / "data/ol1-appendix.json"
TABLE = HERE / "data/t1-l1-ol1.tex"
STEPS = 712
KAPPAS = (0., .01, .05, .1, .5)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


geometry = load_module("retained_geometry", ROOT / (
    "analyses/021-2026-09-10-training-results-figures/02_ol1_geometry.py"))
STYLES = load_module("figure1_style", HERE / "14_plot_14m_quality_latency.py").STYLES[2:]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path, sources):
    sources[path.relative_to(ROOT).as_posix()] = sha(path)
    return json.loads(path.read_text(encoding="utf-8"))


def verify_code(attempt, manifest, verified):
    for entry in manifest["run_code"]["files"]:
        if Path(entry["path"]).name not in geometry.CODE_NAMES:
            continue
        base = ROOT if entry["path"].startswith(("runs/", "src/")) else attempt.parents[2]
        path = (base / entry["path"]).resolve()
        contents = path.read_bytes()
        canonical = "canonical_lf_bytes" in entry
        if canonical:
            contents = contents.replace(b"\r\n", b"\n")
        assert hashlib.sha256(contents).hexdigest() == entry["sha256"], path
        verified[path.relative_to(ROOT).as_posix()] = {
            "sha256": entry["sha256"], "canonical_lf": canonical}


def read_geometry(checkpoints, figure1, sources):
    selected = [r for r in checkpoints if r["model"] in ("14M", "70M")
                and r["scope"] in ("4", "7") and r["pressure"] in ("h", "all")]
    plotted = {r["checkpoint_key"] for r in figure1["trained_points"] if r["scope"] in ("4", "7")}
    assert len(selected) == 40 and {r["checkpoint_key"] for r in selected} == plotted
    assert {(r["model"], r["scope"], r["pressure"], r["kappa"]) for r in selected} == {
        (size, scope, pressure, k) for size in ("14M", "70M")
        for scope in ("4", "7") for pressure in ("h", "all") for k in KAPPAS}
    conditions, verified, pooled = [], {}, {}
    for item in sorted(selected, key=lambda r: (r["model"], r["scope"], r["pressure"], r["kappa"])):
        attempt = ROOT / item["source_attempt"]
        manifest = read_json(attempt / "manifest.json", sources)
        pressure = manifest["activation_pressure"]
        assert manifest["status"] == "completed" and manifest["completed_steps"] == STEPS
        assert pressure["method"] == "orthogonal_l1"
        assert pressure["weight"] == pressure["step_budget"] == 1
        assert pressure["eps"] == 1e-12
        verify_code(attempt, manifest, verified)
        sites = (["h"] if item["pressure"] == "h" else ["a", "m", "h", "z"]
                 if item["scope"] == "4" else ["a", "m", "h", "q_post", "k_post", "v", "z"])
        historical_h_only = item["model"] == "14M" and item["scope"] == "4" and item["pressure"] == "h"
        if historical_h_only:
            # Previously audited Run012 wrapper captures only h despite its manifest.
            assert attempt.parts[-4].startswith("012-")
            assert pressure["sites"] == ["a", "m", "h", "z"]
            wrapper = next(e for e in manifest["run_code"]["files"] if e["path"] == "optimizer_boundary.py")
            assert wrapper["sha256"] == "d9158e72cdb95c7df9237b90e895fb54d96da08b997740c475a982210f3da0d5"
        else:
            assert pressure["sites"] == sites
        path = attempt / "events.jsonl"
        sources[path.relative_to(ROOT).as_posix()] = sha(path)
        events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        rows = [r for r in events if r["event"] == "train"]
        assert [r["step"] for r in rows] == list(range(1, STEPS + 1))
        assert len({r["condition_id"] for r in rows}) == 1
        for row in rows:
            geometry.audit_step(row, pressure)
            assert row["task_direction_norm"]**2 > pressure["eps"]
            if not historical_h_only:
                assert row["pressure_capture_tensor_count"] == 6 * len(sites)
        key = f"{item['model']}/T{item['scope']}/P{item['pressure']}"
        pooled.setdefault(key, []).extend(rows)
        conditions.append({
            **{k: item[k] for k in ("checkpoint_key", "model", "scope", "pressure", "kappa", "source_attempt")},
            "declared_pressure_sites": pressure["sites"], "realized_pressure_sites": sites,
            "historical_h_only_audit": historical_h_only,
            **geometry.summarize(rows),
            "trace": {"rho_opp": [geometry.opposing_component_ratio(r) for r in rows],
                      "r_over_b": [r["pressure_to_task_ratio_raw"] / pressure["step_budget"] for r in rows],
                      "trust_scale": [r["trust_scale"] for r in rows]},
        })
    return conditions, {k: geometry.summarize(v) for k, v in pooled.items()}, verified


def read_comparison(checkpoints, sources):
    selected = [r for r in checkpoints if r["model"] == "14M" and r["scope"] == "1"
                and r["pressure"] in ("L1", "h")]
    assert len(selected) == 8
    assert len({r["initial_parameter_sha256"] for r in selected}) == 1
    assert len({r["training_schedule_hash"] for r in selected}) == 1
    rows, identities = [], []
    for item in sorted(selected, key=lambda r: (r["local_pressure_weight"], r["pressure"])):
        attempt = ROOT / item["source_attempt"]
        m = read_json(attempt / "manifest.json", sources)
        metrics = read_json(attempt / "metrics.json", sources)
        logical = read_json(attempt / "diagnostics/logical_products.json", sources)
        assert m["completed_steps"] == STEPS and m["status"] == "completed"
        assert m["seeds"] == {"data_order": 1234, "model": 1234}
        assert m["condition"]["activation"] == "relu"
        assert m["activation_pressure"]["sites"] == ["h"]
        assert m["activation_pressure"]["weight"] == item["local_pressure_weight"]
        method = m["activation_pressure"]["method"]
        assert method == ("l1_naive" if item["pressure"] == "L1" else "orthogonal_l1")
        assert m["activation_pressure"]["step_budget"] == (None if method == "l1_naive" else 1.)
        final = metrics["validation"]["final"]
        for coverage in (final, logical["coverage"]):
            assert coverage["complete_block_coverage"] and coverage["sequences"] == 338
            assert coverage["excluded_tail_tokens"] == 1444
        assert m["data"]["validation"]["documents"] == 500
        assert final["loss"] == item["loss"]
        assert logical["measured"] == item["counts"]
        counts = logical["measured"]
        sparsity = 100 * counts["block_zero_product_count"] / counts["model_product_count"]
        assert np.isclose(sparsity, item["sparsity"], rtol=0, atol=1e-14)
        identities.append({"seeds": m["seeds"], "data": m["data"],
                           "optimizer": metrics["recipe_mapping"]["optimizer"],
                           "input_tokens": m["input_tokens"]})
        rows.append({**{k: item[k] for k in ("checkpoint_key", "source_attempt", "initial_parameter_sha256", "training_schedule_hash")},
                     "method": "L1" if method == "l1_naive" else "OL1",
                     "lambda": item["local_pressure_weight"], "sparsity_percent": sparsity,
                     "validation_loss": final["loss"], "loss_evaluation": item["loss_evaluation"],
                     "zero_products": counts["block_zero_product_count"],
                     "total_products": counts["model_product_count"]})
    assert all(identity == identities[0] for identity in identities)
    assert {(r["method"], r["lambda"]) for r in rows} == {
        (method, weight) for method in ("L1", "OL1") for weight in (.05, .1, .5, 1.)}
    return {"matched_protocol": identities[0], "rows": rows}


def write_table(comparison):
    text = [r"% Generated by Analysis024/29_ol1_appendix.py; Observation039.",
            r"\begin{table}[!htbp]", r"\centering", r"\small",
            r"\caption{\textbf{L1 and OL1 at $T_1/P_1$ on Pythia-14M.} Final validation loss and model-wide sparsity after 712 matched training steps; OL1 uses $b=1$. Both metrics cover all 338 complete validation blocks (500 documents; 1,444 tail tokens excluded). One initialization per condition.}",
            r"\label{tab:l1-ol1}", r"\begin{tabular}{r rr rr}", r"\toprule",
            r" & \multicolumn{2}{c}{L1} & \multicolumn{2}{c}{OL1} \\",
            r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
            r"$\lambda$ & $S_{\mathrm{model}}$ (\%) & Val. loss & $S_{\mathrm{model}}$ (\%) & Val. loss \\",
            r"\midrule"]
    for weight in (.05, .1, .5, 1.):
        a, b = [next(r for r in comparison["rows"] if r["lambda"] == weight and r["method"] == method)
                for method in ("L1", "OL1")]
        text.append(f"{weight:g} & {a['sparsity_percent']:.4f} & {a['validation_loss']:.4f} & "
                    f"{b['sparsity_percent']:.4f} & {b['validation_loss']:.4f} " + r"\\")
    text += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    TABLE.write_text("\n".join(text) + "\n", encoding="utf-8", newline="\n")


def make_figure(conditions):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                        "axes.titlesize": 11.5, "axes.labelsize": 11,
                        "xtick.labelsize": 10, "ytick.labelsize": 10,
                        "axes.spines.top": False, "axes.spines.right": False,
                        "axes.linewidth": .65, "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(8.6, 3.85))
    fig.subplots_adjust(left=.125, right=.985, top=.79, bottom=.27, wspace=.38)
    fig.suptitle("OL1 Geometry on Pythia-14M", fontsize=14, y=.98)
    for offset, (scope, pressure, label, color, ls) in zip((-.12, -.04, .04, .12), STYLES):
        group = sorted((r for r in conditions if (r["model"], r["scope"], r["pressure"]) ==
                        ("14M", scope, pressure)), key=lambda r: r["kappa"])
        med = 100 * np.array([r["rho_opp"]["median"] for r in group])
        low = 100 * np.array([r["rho_opp"]["p25"] for r in group])
        high = 100 * np.array([r["rho_opp"]["p75"] for r in group])
        left.errorbar(np.arange(5) + offset, med, yerr=(med-low, high-med),
                      color=color, ls=ls, lw=1.5, marker="o", ms=5.2,
                      mec="white", mew=.6, elinewidth=.8, capsize=2, capthick=.8)
        traces = np.array([r["trace"]["r_over_b"] for r in group])
        for trace in traces:
            right.plot(np.arange(1, STEPS+1), trace, color=color, lw=.45, alpha=.10)
        right.plot(np.arange(1, STEPS+1), np.median(traces, axis=0), color=color, ls=ls, lw=1.6)
    # All plotted 14M medians and IQR endpoints are positive; aligned zeros
    # remain in the underlying training-step samples used for those quantiles.
    left.set(yscale="log", ylim=(.0005, 300), xlim=(-.35, 4.35), xlabel=r"Threshold $\kappa$",
             ylabel="Removed opposing component\n" + r"(% of $\Vert u\Vert_2$)")
    left.set_xticks(range(5), ["0", "0.01", "0.05", "0.1", "0.5"])
    left.set_yticks([.001, .01, .1, 1, 10, 100], ["0.001", "0.01", "0.1", "1", "10", "100"])
    left.minorticks_off()
    left.set_title("(a) Opposing component", loc="left", pad=10)
    right.set(yscale="log", ylim=(.004, 3500), xlim=(1, STEPS),
              xlabel="Optimizer step", ylabel=r"Pre-cap norm ratio $r/b$")
    right.set_xticks([1, 200, 400, 600])
    right.set_yticks([.01, .1, 1, 10, 100, 1000], ["0.01", "0.1", "1", "10", "100", "1000"])
    right.minorticks_off()
    right.axhline(1, color="#666B71", lw=.8, ls=(0, (2, 3)))
    right.text(690, 1.35, "Norm cap", fontsize=9, color="#666B71", ha="right", va="bottom")
    right.set_title("(b) Pressure budget", loc="left", pad=10)
    for ax in (left, right):
        ax.tick_params(length=3, width=.65)
        ax.grid(axis="y", color="#E8EAED", lw=.6)
        ax.set_axisbelow(True)
    handles = [Line2D([], [], color=color, ls=ls, lw=1.6, marker="o", ms=5.2,
                      mec="white", mew=.6, label=label) for _, _, label, color, ls in STYLES]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.53, .02), ncol=4,
               frameon=False, fontsize=11, handlelength=2.4, columnspacing=2.5)
    fig.savefig(OUTPUT, metadata={"Title": "OL1 Geometry on Pythia-14M",
                                  "CreationDate": None, "ModDate": None})
    return fig


def main():
    sources = {path.relative_to(ROOT).as_posix(): sha(path) for path in (
        Path(geometry.__file__), HERE / "14_plot_14m_quality_latency.py",
        ROOT / "analyses/023-2026-09-17-14m-pressure-targets-paper-table/results.json")}
    checkpoints = read_json(COHORT, sources)["checkpoints"]
    figure1 = read_json(FIGURE1, sources)
    conditions, families, verified = read_geometry(checkpoints, figure1, sources)
    comparison = read_comparison(checkpoints, sources)
    figure = make_figure(conditions)
    plt.close(figure)
    write_table(comparison)
    data = {
        "script": Path(__file__).relative_to(ROOT).as_posix(), "script_sha256": sha(Path(__file__)),
        "source_sha256": sources, "verified_historical_code": verified,
        "geometry": {"lambda": 1, "budget": 1, "eps": 1e-12,
                     "conditions": conditions, "families": families, "optimizer_records": 40*STEPS,
                     "interval": "within-condition 25th/75th percentiles over 712 optimizer steps, not confidence intervals",
                     "rho_opp": "max(0,-dot_before)/(task_direction_norm**2+eps); before cap and group learning rates",
                     "plotted_models": ["14M"],
                     "plotted_checkpoint_keys": [r["checkpoint_key"] for r in conditions if r["model"] == "14M"],
                     "left_axis": "100*rho_opp; log; all plotted 14M medians and IQR endpoints are positive; underlying aligned zeros retained in quantiles",
                     "right_axis": "r/b; log scale; thin traces per kappa, bold unsmoothed median over five kappas",
                     "cap_rule": "trust_scale < 1; includes first step with zero learning rate",
                     "historical_h_only_source": "analyses/023-2026-09-17-14m-pressure-targets-paper-table/results.json"},
        "t1_comparison": comparison,
        "output_sha256": {OUTPUT.relative_to(ROOT).as_posix(): sha(OUTPUT),
                          TABLE.relative_to(ROOT).as_posix(): sha(TABLE)}}
    DATA.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    for key, summary in families.items():
        print(f"{key}: cap {summary['cap_active_steps']}/{summary['observations']} "
              f"({summary['cap_active_percent']:.2f}%), median r/b {summary['median_pre_cap_ratio']:.4f}")


if __name__ == "__main__":
    main()
