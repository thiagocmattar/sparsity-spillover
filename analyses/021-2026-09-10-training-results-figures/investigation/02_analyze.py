"""Descriptive associations and all ten matched four-/seven-site pairs."""

import csv
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
FAMILIES = ("baseline/local", "4-site", "7-site")
PREDICTORS = (
    ("s_model_percent", "full_speedup"),
    ("projection_scalar_zero_fraction", "projection_sparse_gain"),
    ("projection_mma_bypass_fraction", "projection_sparse_gain"),
    ("attention_scalar_zero_fraction", "attention_sparse_gain"),
    ("attention_mma_skip_fraction", "attention_sparse_gain"),
    ("projection_arithmetic_avoidance_proxy_fraction", "projection_sparse_gain"),
    ("s_model_percent", "full_candidate_gm_ms"),
    ("s_model_percent", "native_baseline_gm_ms"),
    ("projection_mma_bypass_fraction", "projection_sparse_gain_native_normalized"),
    ("attention_mma_skip_fraction", "attention_sparse_gain_native_normalized"),
)


def fit(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(x) < 3 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return {"n": len(x), "predictor_unique_values": len(np.unique(x)), "pearson_r": None, "ols_r2": None, "slope": None,
                "intercept": None, "status": "undefined: constant predictor/outcome or fewer than 3 observations"}
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    r2 = 1 - np.sum((y - design @ [intercept, slope]) ** 2) / np.sum((y - y.mean()) ** 2)
    r = np.corrcoef(x, y)[0, 1]
    assert np.isclose(r2, r * r)
    return {"n": len(x), "predictor_unique_values": len(np.unique(x)), "pearson_r": float(r), "ols_r2": float(r2),
            "slope": float(slope), "intercept": float(intercept), "status": "descriptive; unweighted OLS with intercept"}


def analyze(rows):
    correlations = []
    for predictor, outcome in PREDICTORS:
        for family in ("all", *FAMILIES):
            group = rows if family == "all" else [r for r in rows if r["family"] == family]
            correlations.append({"predictor": predictor, "outcome": outcome, "family": family,
                                 **fit([r[predictor] for r in group], [r[outcome] for r in group])})
    # Within-family centering asks whether the global relation is only group offsets.
    centered_x, centered_y = [], []
    y = np.array([r["full_speedup"] for r in rows])
    group_means = {f: np.mean([r["full_speedup"] for r in rows if r["family"] == f]) for f in FAMILIES}
    for family in FAMILIES:
        group = [r for r in rows if r["family"] == family]
        gx = np.array([r["s_model_percent"] for r in group])
        gy = np.array([r["full_speedup"] for r in group])
        centered_x.extend(gx - gx.mean())
        centered_y.extend(gy - gy.mean())
    family_only_r2 = 1 - sum((r["full_speedup"] - group_means[r["family"]]) ** 2 for r in rows) / np.sum((y - y.mean()) ** 2)
    centered = fit(centered_x, centered_y)
    family_summary = {"family_only_r2": float(family_only_r2),
                      "family_centered_smodel_vs_speedup": centered}
    paired = []
    metrics = ("s_model_percent", "projection_model_contribution_pp", "projection_mma_bypass_fraction",
               "projection_arithmetic_avoidance_proxy_fraction", "projection_simt_products",
               "attention_model_contribution_pp", "attention_mma_skip_fraction",
               "qkv_mma_bypass_fraction", "ffn_up_mma_bypass_fraction", "ffn_down_mma_bypass_fraction", "attn_out_mma_bypass_fraction",
               "projection_sparse_gain", "attention_sparse_gain", "full_speedup",
               "fusion_only_speedup_common_native", "native_baseline_gm_ms",
               "all_skips_off_candidate_gm_ms", "projection_on_candidate_gm_ms", "full_candidate_gm_ms")
    for pressure in ("none", "orthogonal_l1"):
        for kappa in (0, .01, .05, .1, .5):
            four, = [r for r in rows if r["family"] == "4-site" and r["pressure"] == pressure and r["kappa"] == kappa]
            seven, = [r for r in rows if r["family"] == "7-site" and r["pressure"] == pressure and r["kappa"] == kappa]
            pair = {"pressure": pressure, "kappa": kappa, "four_condition": four["condition"], "seven_condition": seven["condition"]}
            for metric in metrics:
                pair[f"four_{metric}"] = four[metric]
                pair[f"seven_{metric}"] = seven[metric]
                pair[f"delta_{metric}"] = seven[metric] - four[metric]
            for metric in ("native_baseline_gm_ms", "full_candidate_gm_ms", "full_speedup",
                           "fusion_only_speedup_common_native", "projection_sparse_gain", "attention_sparse_gain"):
                pair[f"seven_over_four_{metric}"] = seven[metric] / four[metric]
            assert np.isclose(pair["seven_over_four_full_speedup"],
                              pair["seven_over_four_native_baseline_gm_ms"] / pair["seven_over_four_full_candidate_gm_ms"])
            paired.append(pair)
    return correlations, paired, family_summary


def main():
    rows = json.loads((HERE / "data/checkpoints.json").read_text())
    correlations, paired, family_summary = analyze(rows)
    for name, records in (("associations", correlations), ("matched-pairs", paired)):
        with (HERE / f"data/{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
    result = {"associations": correlations, "matched_pairs": paired, "family_summary": family_summary}
    (HERE / "data/analysis.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for r in correlations[:20]:
        print(r["predictor"], r["family"], "r=",r["pearson_r"],"R2=",r["ols_r2"])
    print('Family summary:', family_summary)
    for p in paired:
        print(p['pressure'],p['kappa'], 'd projection pp',round(p['delta_projection_model_contribution_pp'],4),
              'd bypass pp',round(100*p['delta_projection_mma_bypass_fraction'],3),
              'full speedup ratio',round(p['seven_over_four_full_speedup'],4),
              'K050 latency ratio',round(p['seven_over_four_full_candidate_gm_ms'],4))


if __name__ == "__main__":
    main()
