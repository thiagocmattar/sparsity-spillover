# K050 sparsity, realized speedup and matched kernel ablations

## Question and scope

How does model-wide sparsity relate to speedup under the same K050
implementation, and how do projection and attention skipping change the
cohort result relative to the fused implementation with skips disabled?

The author approved this redesign on 11 September 2026. It remains
**analysis-only**: no manuscript changes, new kernel development, training or
benchmarking. Search history is omitted from this figure; retained historical
artifacts are unchanged.

## Sources, method and coverage

- Immediate source: Analysis 018 [figure_data.json](../../018-2026-09-08-results-materials/figure_data.json),
  using its corrected 30-checkpoint runtime cohort.
- Underlying source: Run 029
  [matched-retrospective-001.json](../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/results/matched-retrospective-001.json).
  The reduction verifies its hash against Analysis 018, and reconciles each
  retained speedup, count record and replicate with the source.
- Exactly c01-c30, final phase: 30 K050 checkpoints plus the same 30 for each
  of `k050-no-skip` and `k050-attention-dense`. All 90 comparisons qualify,
  with three complete fresh-process replicates each. Historical h-only A4
  pressure checkpoints are outside this manuscript cohort.
- One physical RTX5090; BF16, batch one, uncached 2,048-token inference with
  all 50,304 output logits. Each candidate is paired with its same-checkpoint
  native PyTorch/SDPA CUDA-graph reference on 64 fixed validation inputs,
  seven paired passes and three fresh processes. Checkpoint speedup equally
  pools the 448 host-latency ratios per process geometrically.
- Numerical qualification covers all 500 validation documents packed into
  338 complete blocks: 692,224 input tokens, 691,886 prediction tokens and a
  1,444-token excluded tail. These are existing randomly pretrained models
  with seed 1234; no new validation pass is performed here.
- Sparsity uses retained canonical FP16 integer counts: block zero-operand
  products divided by all counted block products plus the dense output
  projection. Operation counts are pooled before division; future-masked
  attention pairs are excluded. Each runtime point is reconciled to its
  corresponding trained endpoint.
- The fit is unweighted OLS with an intercept over all 30 K050 points.
  Each ablation GM weights those same 30 checkpoints equally. Increment
  labels use `100 * (GM(treatment_speedup / reference_speedup) - 1)` with
  checkpoint identities matched before taking ratios. They are relative
  percentages, not additive native-relative speedup components.

## Figure and proposed caption

[Publication PDF](../figures/07-kernel-realization.pdf)

**Model-wide sparsity relates to realized acceleration, while matched
ablations separate the effects of sparse skipping.** (a) Full-model K050
speedup versus model-wide sparsity for all 30 Pythia-14M checkpoints. Gray-green
circles denote baseline/local recipes, blue diamonds four-site recipes and
orange triangles seven-site recipes. The dashed line is an unweighted OLS fit
with intercept (R² = 0.817); the dotted line marks native speed. The labeled
seven-site + OL1 checkpoint at kappa = 0.5 reaches 1.78×. (b) Cohort
geometric-mean speedups for the fused implementation with all sparse skips
off (1.183×), projection skipping with attention dense (1.251×), and full
K050 with attention skipping (1.234×). All use the same 30 checkpoints and
native reference protocol. Arrows report relative matched changes between
implementations: +5.7% and -1.3%, rather than an additive decomposition.
Timings use one RTX5090 and BF16 execution; sparsity uses canonical FP16
counts. The regression is descriptive across different trained checkpoints.

The 7.4-by-3.4-inch figure uses two horizontal panels in a 65:35 width ratio.
Panel (a) retains only three recipe families, with no separate pressure
encoding; panel (b) uses neutral implementation colors. Only the high-sparsity
search checkpoint is annotated individually. Both speedup axes have restricted
ranges. The requested word "predicts" in panel (a)'s title refers to the
in-sample linear fit, not a held-out prediction test.

## Results and limits

| Implementation | Qualified checkpoints | Native-relative speedup GM |
| --- | ---: | ---: |
| Fusion only / all skips off | 30 | 1.182858× |
| Projection skipping / attention dense | 30 | 1.250559× |
| Full K050 / attention skipping | 30 | 1.233992× |

OLS R² is 0.8167176698. The labeled c30 checkpoint has 27.482684% model-wide
sparsity and 1.783174564× speedup. Projection skipping increases the cohort
GM by 5.7235%; adding attention skipping decreases it by 1.3248% and is slower
at every matched checkpoint in this cohort.

"Fusion only" names the retained fused/no-skip control; it does not isolate
every non-sparse optimization individually. Ablations were measured in
separate randomized fresh processes against native, so their normalized-speedup
ratios are not direct within-process toggle measurements. Checkpoint quality,
topology and weights vary in panel (a); the fit does not identify a causal
sparsity effect, a universal speed law or equal-quality acceleration. No
cached-decoding, other-hardware or larger-model inference claim follows.

## Reproduction and verification

Source script: [07_kernel_realization.py](../07_kernel_realization.py).
The [reduction](../data/kernel-realization.json) retains source hashes, all
90 qualified comparisons and their replicate records, pooled integer counts,
the fitted coefficients, cohort summaries and all matched ablation ratios.

```powershell
.venv/Scripts/python.exe -X utf8 analyses/021-2026-09-10-training-results-figures/07_kernel_realization.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_kernel_realization.py -q
```

All three focused tests pass: cohort/count coverage, independent OLS with
intercept and plotted-point coverage, and multiplicative matched GM changes.
The PDF was rendered and visually checked; all fonts are embedded and all
text lies within the page. The manuscript remains unchanged.
