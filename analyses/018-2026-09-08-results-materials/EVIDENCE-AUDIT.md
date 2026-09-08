# Evidence scope and discrepancies

## Included evidence

| Evidence | Retained source | Scope |
| --- | --- | --- |
| A0, A1-H, four local L1 doses at 14M | Run 004 | Six training conditions |
| Four local OL1 doses at 14M | Run 009 | Four conditions |
| A4, A7, A7-OL1, corrected A4-OL1 at 14M | Runs 011, 013, 014, 015 | Five thresholds each |
| Selected ladder at 70M/410M | Runs 018/019 | Twelve conditions each |
| Uniform clipping at 14M | Analysis 006 raw combined artifact | 150 actual evaluations |
| Uniform clipping at 70M/410M | Runs 018/019 raw TEAL artifacts | Twenty evaluations each |
| Complete cohort clipping | Run 030 measurement release | 540 evaluations across 54 checkpoints; Figure 01 uses all 300 at 14M |
| Matched kernel retrospective | Run 029 matched-retrospective-001.json | Final cohort restricted to c01–c30; existing qualified progress |

The user explicitly excluded historical A4-OL1[h]. Run 012 is not admitted
to current endpoints, contrasts, frontiers, activation/operation tables,
runtime summaries or figures. The corrected four-site recipe comes from
Run 015. The prior Analysis 013 index remains a provenance source; its
embedded historical results are not copied into this package.

The reduction reopens all 54 retained trained endpoints' raw logical counts,
activation statistics, manifests and configs. It checks complete coverage,
integer pooling, site/layer reconstruction, RMS, same-size initialization and
schedule hashes, random initialization, seed 1234, 712 updates and
1,493,172,224 input tokens. It checks all 190 clipping points against actual
evaluation counts and coverage, and reconciles the larger points against
Analysis 011. Figure 01 additionally reads the hashed Run 030 release, checks
all 30 source identities and ten-target grids, and reconciles coverage and
integer counts for its 300 evaluations. Other figures preserve their original
clipping subsets. Run 030 records checkpoint hashes, calibration thresholds,
full raw measurements and transfer verification for the complete cohort.

## Presentation-sensitive definitions

1. **Loss pairing.** Loss and R_model come from the same eager full-validation
   logical pass. Some run summaries use terminal SDPA loss. The terminal value
   is retained in JSON, but not mixed into the figure's quality/sparsity pair.
2. **Clipping anchors.** p=0 is its actual raw sweep evaluation, not a trained
   display anchor substituted by an older plotting reduction. Tiny numerical
   differences are not claimed as scientifically meaningful improvements.
3. **Normalization.** Trained ceilings use the selected gate topology. Clipped
   controls use their verified evaluation sites {a,m,h,z}, including p=0.
   Unmodified A0 has zero reach and undefined utilization. U_arch includes
   natural zeros outside reach; U_reach excludes them. Neither is speedup.
4. **Ceiling workload.** Store integer logical scalar products per complete,
   uncached T=2048 sequence; multiply by 338 for measured denominators. The
   dense LM head receives no zero numerator credit. The new 12-point ceiling
   grid is recomputed with the shared architecture function, not estimated.
5. **Notation and ports.** Draft S_model/S_model^max correspond to operational
   R_model/R_model_max. Plot h,m,q,k,v follows the architecture ladder; q/k
   are post-RoPE. The available post-Wo attention_output is not pre-Wo z.
   A0 lacks a/z near-zero diagnostics, so the activation figure omits them.
6. **Retained distributions.** Artifacts hold exact-zero and magnitude-.001/.01
   counts plus moments. Figure 05 restores four measured magnitude bins in
   κ=0,.05,.5 rows; finer .05/.5 boundaries require new measurement. The
   diagnostic design is pending, not silently approximated from the >.01 tail.
   Tables preserve all retained bands/RMS. No density or raw sample is invented.
7. **Numerical controls.** No-pressure A4/A7 gates agree mathematically at κ=0
   but have small trained numerical differences. A1-H and G+,0 also differ in
   boundary derivative. These are disclosed comparison limits.
8. **Runtime cohort.** The final 35-checkpoint source is reduced to 30; all
   candidate summaries, ablations and OLS statistics are recomputed. Each
   runtime point links to an included trained evidence ID with the same
   canonical R_model. Qualified search progress used only c01/c11/c25/c30.
   The current analysis therefore intentionally differs from the unedited
   draft's 35-checkpoint runtime report. No timing is rerun.

## Coverage and interpretation limits

All scientific endpoints use the 500 validation documents: 338 complete
2048-token blocks, 692,224 inputs and 691,886 prediction targets. The 1,444-token
tail is excluded. The case-study statistics pool integer counts before division.
Activation and logical diagnostics are separate complete passes.

Actual training uses FP16 dynamic loss scaling and FP32 parameters, with one
initialization/order seed (1234). Larger cohorts share the token budget, not
tokens/parameter or all optimizer values; the 410M peak LR differs. They lack
unpressured A4/A7 and local pressure grids, so they test recipe transfer, not
the full isolated 14M intervention effects. No between-seed confidence intervals,
new scaling-law fit or untouched confirmatory holdout is available.

Short pilots, failed initialization attempts 016/017 and the alternative LR
screen do not replace the frozen cohorts. Runs 022–024 retain the original
exact-ELL correctness/transfer limitations. Earlier adaptive systems runs used
different hardware/baselines and are not pooled into the matched Run 029 curve.
The current RTX5090 runtime evidence does not establish cached-decoding,
larger-scale or equal-quality acceleration.

No model execution, new experiment, cloud action, finding promotion or manuscript
edit was performed. This is an authorized reanalysis of retained local evidence.
