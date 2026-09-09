# Active rewrite: claim and evidence ledger

All observations below are in
[Analysis 013](../../analyses/013-2026-09-04-matched-intervention-manuscript/observations/INDEX.md).
The current user request authorizes descriptive manuscript integration; the
finding registry remains unchanged.

| Claim in the draft | Source observation / table | Exact scope |
|---|---|---|
| Local magnitude pressure can improve loss and opportunity | O001; trained-14M | ReLU, fixed seed and budget; OL1 and L1 have mixed ordering |
| Expanding pressure from h to a,m,h,z raises loss at every tested threshold | O001; Run 012 realization audited in Analysis 009; Run 015 corrected scope | Gates remain A4, lambda and step budget remain 1; tensor means are averaged over the selected set |
| Adding Q/K/V gates without pressure creates additional opportunity at 14M | O001; paired-effects | Joint addition of three post-RoPE/value gates; one-seed contrasts |
| Pressure responses differ between A4 and A7 | O001; paired-effects and pressure-differences | Exploratory comparison of configuration-specific objectives; pressure sites expand with gates |
| Trained conditions extend the evaluated clipping frontier | O002; loss-budgets; full clipping grids | Uniform quantile allocation; 70M clipping wins at tight quality allowances |
| High-dose A7 adds attention-product opportunity | O003; operation PDF and raw counters | Counted QK/PV operand zeros under full-sequence causal accounting |
| Higher 70M R_model partly reflects the denominator | O003; architecture | A7 R_block falls from 14M to 70M while the block share grows |
| High-dose branch tensors are almost entirely zero | O003; pooled exact-zero counts | h and z exceed 99.8% zeros in the four main high-dose endpoints; function/capacity attribution needs further tests |
| 410M reverses the high-dose within-family loss direction | O004/O005; all 410M tables | Equal tokens, lower tokens/parameter, unresolved cause; disclosed in main text |
| Higher screened learning rates fail to improve late A0 train loss | O004; lr-screen, generated from Run 021 selection.json | Three tested rates; larger data/schedule changes remain open |
| Tested 70M sparse paths run slower than dense | O006; Run 023 observation and verified timing artifacts | One H100 NVL, BF16 prefill, two batch sizes, this derivative and coverage |

## Claim boundaries enforced in the rewrite

- The contribution is the intervention study; counting supports its evaluation.
- A7 gates plus pressure at four fixed sites remain an unexecuted control.
- Thresholds and timing blocks provide different kinds of repeated measurement;
  independent training seeds remain absent.
- The 0.0008 loss change is reported numerically, with the zero-threshold
  A4/A7 numerical comparison alongside it; it is not a resolved quality gain.
- No superiority to published Q-Sparse, optimized TEAL, or Spark is asserted.
- No confirmed scaling law, undertraining diagnosis, or inference speedup is asserted.
- Joint activation/weight and shared-channel/block pressure are future hypotheses.
- Full-model runtime coverage excludes sparse attention; separate attention
  microbenchmarks are labeled accordingly.

Earlier review-v3 table values used terminal losses. This draft consistently
pairs eager counting-pass losses with R_model, changing a few final decimal
places. Terminal values remain in Analysis 013's evidence JSON for audit.
