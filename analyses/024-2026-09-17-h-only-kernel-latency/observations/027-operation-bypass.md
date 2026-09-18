# Per-operation scalar sparsity and instruction bypass

Figure: [10-operation-bypass.pdf](../figures/10-operation-bypass.pdf).
Complete per-setting table: [TABLE_OPERATION_BYPASS.md](../TABLE_OPERATION_BYPASS.md).
Integer counts, scalar opportunities and exact provenance: [operation-bypass.json](../data/operation-bypass.json).

## Question and approved scope

The user selected a per-setting bypass percentage, separated into QKV, QK, PV
and the other matrix-product families, and asked for a figure or table testing
whether attention's scalar sparsity fails to form useful structure. This local
analysis implements the previously proposed instruction-bypass definition from
saved counters. It changes no scientific input, checkpoint, kernel or timing.
The hypothesis is that h/z sparsity forms skippable groups more readily than
QK/PV sparsity, potentially explaining their different runtime benefits.

All 62 retained trained settings are included: 40 at 14M (32 primary settings
and eight historical local L1/OL1 settings), and 22 at 70M. The 40 dense/GeLU
and ReLU post-hoc clipping settings at p=0,...,0.9 are included separately.
The four p=0 clipping records repeat existing weights in a separate evaluation
session; the 102 rows are evaluation settings, not 102 independent checkpoints.
Every counter record covers six layers and all 338 complete 2,048-token blocks
from 500 MiniPile validation documents: 692,224 input tokens and a 1,444-token
excluded tail. Seeds, initialization, optimizer, training data and budget remain
those of the retained checkpoints. There is no new training or cloud launch.

This supplies descriptive evidence for the manuscript's existing distinction
between scalar logical opportunity, MMA bypass and measured speedup in
`manuscript/draft/kernel-autoresearch.tex`. It does not redefine S_model or
change manuscript text. Low bypass despite high scalar opportunity supports
the proposed structural explanation. Substantial bypass without a timing gain
instead shows that instruction structure alone is insufficient to explain cost.

## Metric and implementation

For each setting and operation o, pool integer counts over blocks and layers:

```text
B_o = sum(bypassed MMA instructions)
      / sum(issued MMA instructions + bypassed MMA instructions)
```

JSON stores fractions; tables and the PDF display 100 * B_o. A value of 100%
means every potential matrix instruction in that operation is bypassed. It does
not mean the entire operator has zero cost. Bypass is normalized separately
within each operation; no unweighted or model-wide average is introduced.

| Operation | Actual operand sites | Counter interpretation |
|---|---|---|
| QKV projection | a | All-zero 16x16 A-fragment tests reconstructed from actual operands |
| FFN-up | m | Same 16x16 A-fragment predicate |
| FFN-down | h | Instrumented hybrid 8x16 activation groups, padded to M16; includes short-row substitution |
| Attention-output | z | Same hybrid mechanism; z and attention-output are one site/operation pair |
| QK scores | q_post, k_post | Instrumented attention matrix instructions after RoPE, including kernel padding |
| PV product | attention probabilities, v | Instrumented attention instructions, including causal masking/padding |

The final kernels are K050 at 14M and the qualified k050-70m-v2 shape port at
70M. Their BF16 matrix instructions have shape 16x8x16. The h/z path may move
rows with at most two nonzero entries to scalar arithmetic, subject to numerical
safety checks. Its bypass counts therefore include substitution as well as
elimination. The JSON preserves the actual SIMT product counts separately.

Raw attention bypass is not entirely intervention-induced. Base-model PV
bypass is 10.4167% at 14M and 5.1471% at 70M. The figure shows these references;
the JSON also retains B_o minus the same-size base B_o in percentage points.
That difference is descriptive, not a count of purely activation-induced zero
tiles. The primary B_o remains unadjusted and reaches 100% at full MMA bypass.

The figure compares BF16 bypass against the same diagnostic's BF16
activation-derived scalar zero-product lower bound. This scalar diagnostic
excludes weight zeros and softmax-probability underflow: PV's X coordinate
counts V-induced zeros only. The canonical full FP16 scalar counts are also
retained in the JSON, separately labeled. Neither scalar logical counts nor
their denominator include the kernel's extra padded arithmetic.

## Figure caption and legend

**Scalar Sparsity and Matrix-Instruction Bypass.** Each panel shows one of six
operation families: (a) QKV projection, (b) FFN-up, (c) FFN-down, (d) attention
output, (e) QK scores and (f) PV product. X is BF16 activation-derived scalar
zero-product opportunity (a lower bound); Y is the percentage of potential
matrix instructions bypassed by the frozen implementation. Blue denotes 14M
and orange denotes 70M. Filled circles are trained settings; open circles and
dotted connecting lines are the dense/ReLU post-hoc clipping sweeps. All 102
settings are present in every panel; coincident markers can overlap. Labels in
QK/PV identify T7/Pall at kappa=0.5. Dashed PV guides show the same-size base
model's bypass percentage. Counts pool all 338 validation blocks and six layers.
Bypass includes scalar substitution at h/z and attention masking/padding; it is
not a runtime speedup or a fraction of S_model recovered.

## Results

The following comparison uses **T7/Pall, kappa=0.5** at both sizes. Scalar
percentages below are the BF16 lower bound plotted in the figure.

| Operation | 14M scalar zeros (%) | 14M bypass (%) | 70M scalar zeros (%) | 70M bypass (%) |
|---|---:|---:|---:|---:|
| QKV | 75.63 | 8.44 | 67.20 | 0.00 |
| FFN-up | 68.65 | 0.00 | 68.97 | 0.00 |
| FFN-down (h) | 99.88 | 94.50 | 99.86 | 91.43 |
| Attention-output (z) | 99.92 | 99.64 | 99.84 | 96.78 |
| QK | 97.24 | 58.00 | 81.72 | 8.71 |
| PV | 98.69 | 67.21 | 86.86 | 6.90 |

The evidence supports strong h/z bypass at the high-threshold endpoints, but
the blanket claim that QKV is structured while QK/PV are not is not supported.
QKV depends strongly on recipe: T4/Pall at kappa=0.5 bypasses 67.12% at 14M
and 27.67% at 70M, whereas T7/Ph at the same threshold bypasses 0% at both sizes.
Attention bypass at 14M can be substantial. At the 14M T7/Pall endpoint,
QK bypass is 58.00% and PV bypass is 67.21% (56.79 percentage points above its
base model). At 70M the corresponding values are 8.71% and 6.90%; the latter
is only 1.75 points above its 5.15% baseline. This is descriptive of the retained
recipes and kernels, not an independent-seed scaling result.

The 35 historical 14M settings also have matched skip-toggle timings. The
reducer rebuilds geometric mean host times from all 1,344 raw timings in each
mode: all skipping disabled, projection skipping only, and full skipping.
Projection gain is t_all_off / t_projection_only. Attention gain is
t_projection_only / t_full. All 35 attention gains are below 1: enabling
attention skipping slows the full model in these measurements. At T7/Pall,
kappa=0.5, the gain is 0.989436x (0.468365 ms to 0.473366 ms), despite its
substantial QK/PV bypass. Thus the observed attention limitation includes
implementation overhead or other remaining costs, not merely insufficient
skippable structure. These ablations do not isolate the particular overhead,
nor separate QK timing from PV timing.

No matched skip ablation exists in this analysis for the five new 14M A7/Ph
settings, the 22 70M settings, or the 40 clipping settings. Their final-model
latencies are retained, but they cannot establish the separate causal latency
effect of attention skipping. The 70M shape port also has a different tuning
history. No new inference, timing, training or RunPod resources were needed.

## Sources and verification

- [18_reduce_operation_bypass.py](../18_reduce_operation_bypass.py): joins
  checkpoint weight hashes to full-validation diagnostics in Runs029/033/035/036,
  pools counters, retains scalar work and writes the complete table and JSON.
- [19_plot_operation_bypass.py](../19_plot_operation_bypass.py): renders all
  612 setting/operation records and saves [plot coordinates and hashes](../data/operation-bypass-figure.json).
- [test_operation_bypass.py](../test_operation_bypass.py): count-first pooling,
  integer validity, 100% bypass semantics, cohort identity and dosage coverage,
  baseline padding, scalar substitution and the attention counterexample.

All 30 previously reduced historical ablation results reproduce from raw
timings to relative tolerance 1e-12. All 35 full-mode latencies agree with the
existing Analysis024 table at the same tolerance. Four focused tests pass;
all 801 source hashes and all 612 plotted coordinates were verified. Precision,
coverage and exact diagnostic paths are retained. The six-panel PDF was
rendered and visually checked, with embedded fonts. No manuscript or promoted
finding was changed.
