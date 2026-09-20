# Consolidated kernel appendix and fixed-native-base diagnostic

## Question and authorized scope

The user requested a critical refactor of `app:kernel-results` and following
kernel material: one compact section, an exact implementation description,
test settings, unsuccessful approaches and a defensible relationship between
sparsity, MMA bypass and speedup across 14M/70M. No new GPU experiment is run.
The manuscript remains uncompiled while the user has its PDF open.

The revised section is `manuscript/draft/kernel-appendix.tex`, included at the
end of `results-appendix.tex`. The preceding non-kernel results are unchanged.
The old search-history figure, per-checkpoint native-relative latency tables,
fit figure and export inventory are removed from the rendered appendix, but
their files and original evidence remain available. No operational candidate
identifiers are used in the new rendered prose.

## Critical findings

The old appendix combined three different quantities: acceleration over each
checkpoint's native implementation, fixed-base-model speedup, and conditional
gains from disabling sparse paths. The first has a varying numerator across
recipes. The third is a valid within-checkpoint ablation but does not measure
speedup over the base model. The previous fit therefore cannot be read as a
prediction of base-model speedup. A varying reference can change associations
and even rankings; it does not by itself prove that every reported R-squared
was inflated. With a common reference, projection bypass still correlates
strongly with measured speedup.

The choice of fixed reference materially changes the 70M scale claim:

| Size | Native base latency (ms) | Specialized base latency (ms) | Best trained / native base | Best trained / specialized base |
|---|---:|---:|---:|---:|
| 14M | 0.6554423980056099 | 0.651573035007824 | 1.4265007912491527x | 1.4180795334929153x |
| 70M | 1.6638832965631953 | 3.313247016348893 | 1.0281229219334886x | 2.047274115060905x |

Here a speedup is reference latency divided by specialized recipe latency.
The 70M port is much slower than native execution on dense activations. Only
three of the 22 trained 70M settings beat the native base. The larger gain
relative to its specialized base cannot establish a twofold improvement over
native inference, or an increase in native-base speedup with model size.
Changing between the two *fixed* references rescales a size's y values by a
constant, so it cannot itself change within-size Pearson correlation or OLS
R-squared with an intercept.

The original main Figure 13 and its normalization are preserved in this
appendix-only change. The discrepancy is explicitly disclosed in the new
appendix and surfaced to the user; a separate question asks whether to extend
the revision to the main figure and scale claim. The main text also uses
stronger output-equivalence wording than the retained numerical tolerances.
The appendix now states those tolerances precisely, without asserting bitwise
equivalence. These main-text issues remain for an authorized main-text pass.

Other stale claims are removed: operation-level attribution and scalar-work
counters are available from later measurements; they are not universally
unavailable. The new text distinguishes the completed 14M attribution from
the absent 70M operation-level ablation.

## Method and coverage

[31_kernel_appendix.py](../31_kernel_appendix.py) joins the original
Figure 13's 44 trained settings and 40 post-hoc controls to full-validation
operation counters by checkpoint identity and clipping dose. All 801 primary
source hashes in the counter reduction are checked. Both base references are
independently reconstructed from their original three-process timing records:
64 inputs times seven paired passes, or 1,344 observations per implementation.
Every joined latency, canonical sparsity and checkpoint weight hash is checked.

Counts cover six layers and 338 complete validation sequences per setting;
timing covers the same fixed 64-input subset used in the paper. One training
seed is represented. No counter is inferred from latency or canonical sparsity.

Projection totals pool QKV, FFN-up, FFN-down and attention-output operations.
Attention totals pool QK and PV. Bypass is summed bypassed instructions divided
by summed issued plus bypassed instructions, never the unweighted mean of
percentages. A separate denominator is retained for each group and size.
Projection scalar sparsity is the BF16 activation-zero product count divided
by projection products. It differs from canonical FP16 model-wide sparsity.
No pooled model-wide MMA fraction is introduced.

MMA bypass includes padded rows and scalar substitution at h/z; it is not
eliminated arithmetic or measured memory traffic. Attention includes causal
padding, giving nonzero baseline PV bypass (10.4167% / 5.1471%). The aggregate
attention baseline is half those fractions because QK and PV have equal
eligible counts and baseline QK bypass is zero. The two attention schedules
differ between sizes, so raw bypass percentages are implementation-specific.

## Results and interpretation

For the 22 trained points per size, descriptive Pearson correlations with
fixed-native-base speedup are:

| Predictor | 14M | 70M |
|---|---:|---:|
| Canonical model-wide sparsity | 0.69258 | 0.66690 |
| Projection MMA bypass | 0.96973 | 0.99110 |
| Attention MMA bypass | 0.37480 | 0.30668 |

These are audit summaries, not fitted figure lines or causal estimates.
Projection bypass tracks runtime more closely in this sample, while substantial
scalar sparsity can leave most instruction tiles active. Clipping illustrates
that discrepancy and includes its separate masking overhead. Attention bypass
varies meaningfully at only the extreme T7/Pall trained endpoint; that isolated
contrast does not establish a general relation. At 70M, its native-base speedup
is 0.95165x, below T7/Ph's 1.02812x despite more attention bypass.

The 14M leave-one-path-out experiment is retained as the direct mechanistic
evidence through main Table `tab:kernel-mechanisms`. There is no corresponding
70M causal decomposition. A 70M shape adaptation is not an equal-budget kernel
optimization search. Session differences and changed task quality remain.

## Figure, caption and manuscript association

PDF: [20-kernel-structure-native-base-speedup.pdf](../figures/20-kernel-structure-native-base-speedup.pdf).
Exact points, count denominators, both references and provenance:
[kernel-appendix.json](../data/kernel-appendix.json).

**Scalar sparsity, instruction bypass and speedup over a fixed native base
model.** Rows show 14M and 70M, each with 22 trained settings and 20 post-hoc
control settings. Left: projection scalar sparsity versus projection MMA
bypass. Middle/right: projection/attention bypass versus full-model speedup.
Every speedup within a row uses the same native T0/P0 reference; the horizontal
line is 1x. The Base model marker uses specialized inference, so it need not
lie on that line. Lines connect measured thresholds, not fitted trends;
dotted paths are post-hoc clipping. Counts are pooled, include the stated
instruction-padding and scalar-substitution conventions, and do not measure
saved memory traffic.

The figure replaces `fig:kernel-diagnostics` in the unified appendix. Its
caption and accompanying paragraphs distinguish scalar sparsity, bypass,
common-reference performance and conditional ablations. The copied PDF's
hash is recorded in `manuscript/draft/figures/SOURCES.json`.

## Implementation evidence

- Run028 `candidates/k050/candidate.py` and `norm.cu`: shared-input LayerNorm
  statistics with separate affine parameters and gates.
- Run028 `candidates/k042/projection.cu`: loaded-operand 16x16 zero tests;
  CUTLASS 32x64x64 pipeline and 16x8x16 MMA instructions.
- Run028 `candidates/k049/joint.cu`: short rows, finite-range guards, padded
  8-row fallback, early weight-load avoidance and joint residual output.
- Run028 `candidates/k035/sparse_gemm.h`, `kernel.cu`: complete-fragment
  predicates after loading; causal attention and retained softmax.
- Run035 `kernel/candidate.py`, `22_native_attention_port.py` and README:
  dimensions, unsplit 128x128 attention schedule, failed first schedule and
  final full-validation qualification without relaxed tolerances.
- Run029 Observation06: specialization limits, unchanged-threshold-grid
  qualification, counters and candidate-code audit.
- Analysis024 Observations027, 029, 034 and 035: saved counter definitions,
  unsuccessful attention approaches, completed per-operation attribution and
  the correct-but-slower a/m hybrid port.

## Verification

Seven focused tests pass: fixed-reference identity and all 84 members, integer
pooling and scalar/substitution distinctions, full plot coverage and bounds,
existing counter invariants, plus source/output provenance. The standalone
PDF is rendered and visually checked. Manuscript references and TeX structure
are checked without a build; `main.pdf` retains its pre-task SHA256.

```powershell
.venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/31_kernel_appendix.py
.venv/Scripts/python.exe -X utf8 -m pytest -q analyses/024-2026-09-17-h-only-kernel-latency/test_kernel_appendix.py analyses/024-2026-09-17-h-only-kernel-latency/test_operation_bypass.py
```
