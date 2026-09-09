"""Reduce retained boundary logs; no model execution or checkpoint loading.

Declared before reduction: dynamics cohort = A0 and A4/A7-OL1 kappa=.5 at
all three sizes. Early = steps 1..142; late = steps 570..712 (final 20%).
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import statistics

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "data"


def sha(path, canonical=False):
    data = path.read_bytes()
    return hashlib.sha256(data.replace(b"\r\n", b"\n") if canonical else data).hexdigest()


def csv_write(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(values):
    values = sorted(values)
    return dict(min=min(values), median=statistics.median(values), max=max(values),
                mean=statistics.mean(values))


def main():
    OUT.mkdir(exist_ok=True)
    source = ROOT / "analyses/018-2026-09-08-results-materials/figure_data.json"
    data = json.loads(source.read_text(encoding="utf-8"))
    sources, saturation, ol1_steps, curves, dynamics = [], [], [], [], []
    for r in data["trained"]:
        attempt = ROOT / r["source"]
        manifest = json.loads((attempt / "manifest.json").read_text(encoding="utf-8"))
        pressure = r["identity"]["pressure"]
        assert manifest["activation_pressure"] == pressure
        code = []
        for item in manifest["run_code"]["files"]:
            if Path(item["path"]).name not in {"pressure.py", "optimization.py", "optimizer_boundary.py", "training.py"}:
                continue
            p = (attempt.parents[2] / item["path"]).resolve()
            canonical = "canonical_lf_bytes" in item
            assert sha(p, canonical) == item["sha256"], (r["id"], p)
            code.append(dict(path=p.relative_to(ROOT).as_posix(), sha256=item["sha256"],
                             normalization="LF" if canonical else "raw bytes"))
        all_events, invalid_lines = [], []
        for line_number, line in enumerate((attempt / "events.jsonl").read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                all_events.append(json.loads(line))
            except json.JSONDecodeError:
                invalid_lines.append(dict(line=line_number, characters=len(line), null_characters=line.count("\x00")))
        events = [e for e in all_events if e.get("event") == "train"]
        missing = sorted(set(range(1, 713)) - {e["step"] for e in events})
        assert [e["step"] for e in events] == sorted({e["step"] for e in events})
        assert all(e["input_tokens_seen"] == e["step"] * 2097152 for e in events)
        skipped = sum(bool(e["optimizer_step_skipped"]) for e in events)
        sources.append(dict(id=r["id"], source=r["source"],
                            files={name: sha(attempt/name) for name in ["events.jsonl", "manifest.json", "config.yaml"]},
                            verified_code=code, boundaries=len(events), skipped=skipped, missing_steps=missing, invalid_lines=invalid_lines,
                            validation_events=[{k: v for k, v in e.items() if k in {"event", "step", "loss", "validation_loss"}} for e in all_events if "valid" in e.get("event", "")],
                            pressure=pressure, precision=r["identity"]["training"]["precision"]))
        if pressure["method"] == "orthogonal_l1":
            assert not missing and not invalid_lines, r["id"]
            assert pressure["step_budget"] == 1
            assert pressure["weight"] in ([.05, .1, .5, 1] if r["family"] == "A1-H-OL1" else [1])
            valid = [e for e in events if not e["optimizer_step_skipped"]]
            for e in valid:
                ratio = e["pressure_to_task_ratio_raw"]
                expected = min(1., pressure["step_budget"] / (ratio + pressure["eps"])) if ratio > 0 else 1.
                assert abs(expected - e["trust_scale"]) < 1e-12
                assert abs(ratio * expected - e["pressure_to_task_ratio_final"]) < 1e-12
                e["cap_active"] = ratio > 0 and pressure["step_budget"] / (ratio + pressure["eps"]) < 1
                ol1_steps.append(dict(id=r["id"], step=e["step"], learning_rate=e["learning_rate"],
                                     cap_active=e["cap_active"], **{k:e[k] for k in ["task_direction_norm", "pressure_to_task_ratio_raw", "trust_scale", "pressure_to_task_ratio_final", "projection_applied", "eligible_parameter_tensors", "skipped_parameter_tensors"]}))
            for phase, selected in [("all", valid), ("early_1_142", [e for e in valid if e["step"] <= 142]), ("late_570_712", [e for e in valid if e["step"] >= 570])]:
                positive_lr = [e for e in selected if e["learning_rate"] > 0]
                saturation.append(dict(id=r["id"], scale=r["scale"], family=r["family"], dose=r["dose"],
                                       weight=pressure["weight"], phase=phase, boundaries=len(selected),
                                       cap_active=sum(e["cap_active"] for e in selected),
                                       positive_lr_boundaries=len(positive_lr), positive_lr_cap_active=sum(e["cap_active"] for e in positive_lr),
                                       projected=sum(e["projection_applied"] for e in selected),
                                       correction_ratio=summarize([e["pressure_to_task_ratio_final"] for e in selected]),
                                       raw_ratio=summarize([e["pressure_to_task_ratio_raw"] for e in selected]),
                                       eligible=sorted({e["eligible_parameter_tensors"] for e in selected})))
        selected_cohort = r["family"] == "A0" or (r["family"] in {"A4-OL1", "A7-OL1"} and r["dose"] == .5)
        if selected_cohort:
            assert not missing and not invalid_lines, r["id"]
            successful = 0
            for e in events:
                successful += not e["optimizer_step_skipped"]
                curves.append(dict(id=r["id"], scale=r["scale"], family=r["family"], step=e["step"],
                                   successful_updates=successful, input_tokens=e["input_tokens_seen"],
                                   **{k:e[k] for k in ["task_loss", "adamw_gradient_norm_pre_clip", "adamw_gradient_norm_post_clip", "learning_rate", "optimizer_step_skipped"]}))
            late = [e for e in events if e["step"] >= 570]
            # OLS over raw equal-batch task losses; no adaptive window or smoothing.
            x = [e["input_tokens_seen"]/1e9 for e in late]
            y = [e["task_loss"] for e in late]
            xm, ym = statistics.mean(x), statistics.mean(y)
            slope = sum((a-xm)*(b-ym) for a,b in zip(x,y))/sum((a-xm)**2 for a in x)
            dynamics.append(dict(id=r["id"], scale=r["scale"], family=r["family"], late_steps=[570,712],
                                 late_loss_slope_per_billion_tokens=slope,
                                 late_first_20_loss_mean=statistics.mean(y[:20]), late_last_20_loss_mean=statistics.mean(y[-20:]),
                                 late_preclip_norm=summarize([e["adamw_gradient_norm_pre_clip"] for e in late if e["adamw_gradient_norm_pre_clip"] is not None]),
                                 late_postclip_norm=summarize([e["adamw_gradient_norm_post_clip"] for e in late if e["adamw_gradient_norm_post_clip"] is not None]),
                                 late_clipped=sum(bool(e["adamw_gradient_was_clipped"]) for e in late),
                                 learning_rate_first_late=late[0]["learning_rate"], learning_rate_final=late[-1]["learning_rate"],
                                 endpoint_loss=r["loss"], skipped=skipped))
    csv_write(OUT / "ol1-steps.csv", ol1_steps)
    csv_write(OUT / "training-curves.csv", curves)
    audit = dict(source=dict(path=source.relative_to(ROOT).as_posix(), sha256=sha(source)),
                 log_coverage=sources, saturation=saturation, dynamics=dynamics,
                 late_window="Steps 570..712, final 20% of 712 boundaries; fixed before reading outcomes.",
                 norm_scope="Accumulated, AMP-unscaled pre-clip global gradient norm over all trainable parameters; task-only for the nine selected curves. OL1 direction/cap norms are pre-learning-rate over eligible parameters.")
    (OUT / "log-audit.json").write_text(json.dumps(audit, indent=2)+"\n", encoding="utf-8")
    table = [r"% Analysis 020 / 01_audit_logs.py; fixed steps 570--712.",
             r"\begin{tabular}{@{}llrr@{}}", r"\toprule",
             r"Size & Recipe & Median pre-clip norm & Loss slope / billion tokens \\", r"\midrule"]
    for r in sorted(dynamics, key=lambda r: (["A0","A4-OL1","A7-OL1"].index(r["family"]),["14M","70M","410M"].index(r["scale"]))):
        table.append(f"{r['scale']} & {r['family']} & {r['late_preclip_norm']['median']:.4f} & ${r['late_loss_slope_per_billion_tokens']:+.4f}$ " + r"\\")
    table += [r"\bottomrule", r"\end{tabular}"]
    (HERE / "tables").mkdir(exist_ok=True)
    table_path = HERE / "tables/training-dynamics.tex"
    table_path.write_text("\n".join(table)+"\n", encoding="utf-8", newline="\n")
    shutil.copyfile(table_path, ROOT / "manuscript/draft/tables/training-dynamics.tex")
    print(json.dumps(dict(conditions=len(sources), boundaries=sum(s["boundaries"] for s in sources),
                          skipped=sum(s["skipped"] for s in sources), ol1_boundaries=len(ol1_steps),
                          cap_active=sum(r["cap_active"] for r in ol1_steps), dynamics=dynamics), indent=2))


if __name__ == "__main__":
    main()
