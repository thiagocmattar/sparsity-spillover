# Operation latency evidence in the manuscript

## Question and manuscript scope

Why do h/z sparsity reduce latency while additional QK/PV bypass does not
under the implemented kernel? At the user's request, rewrite
`manuscript/draft/kernel-autoresearch.tex` with the mechanism first, then
the new data and conclusions. A six-row table replaces the previous grouped
two-checkpoint table; the earlier export and generator remain available.
No new training, GPU timing, kernel changes or cloud work are included.

## Evidence, method and coverage

The source is [Run037's completed operation ablation](../../../runs/037-2026-09-19-pythia14m-operation-latency/observations/001-operation-latency.md),
using the retained 14M T7/Pall checkpoint at kappa=0.5. Disable one path at
a time with weights, gates and all other paths fixed. QK jointly controls
q and k; PV controls v. The h/z fusion remains intact.

All 30 scientific processes pass all 338 complete 2048-token validation
blocks from 500 MiniPile documents (692224 input tokens, 1444-token excluded
tail). Recorded maximum absolute logit error is zero. Timing uses one RTX5090,
BF16, batch one, 64 fixed inputs, seven passes and three processes per mode,
with CUDA graphs and the full 50304-token vocabulary output. Run037's
`12_report.py` verifies all 26880 raw candidate/native timings, ten diagnostic
passes, counter controls and full validation results.

[The table builder](../25_build_operation_latency_table.py) verifies the source
hash and reconstructs the six displayed bypass ratios, conditional time
differences and process spans from the audited source. It writes
[unrounded data and source hashes](../data/operation-latency.json), the
[analysis table](../data/operation-latency.tex), and the identical manuscript
copy `manuscript/draft/tables/operation-latency.tex`.

## Caption, observations and interpretation

**Conditional latency savings by operation.** Pythia-14M T7/Pall, kappa=0.5;
RTX5090, BF16, batch one, 2048 tokens and full vocabulary output. Saved time
is full-model latency with one path off minus latency with all paths on:
positive values help. Geometric means use 64 inputs, seven passes and three
processes per mode; spans compare process extrema, not confidence intervals.
All 338 validation blocks match reference outputs. Bypass pools all layers
and blocks, including scalar replacement at h/z and masked/padded attention
work. Effects are conditional, not additive.

- h saves 150.1 us and z saves 29.0 us. Their process spans remain positive.
- a, m and QK have small negative point estimates with spans crossing zero;
  their individual signs remain unresolved against process variation.
- PV saves -6.2 us, with its entire observed difference span negative.
- Enabling QK/PV skipping jointly raises full-model latency from
  0.514037 to 0.521078 ms: 7.041 us overhead. The two process ranges do not
  overlap. The table's individual effects must not be summed.

The code-level explanation retains the audited distinction: h/z inspect
8x16 activation groups before weight loads, while a/m and attention inspect
loaded operands. Attention still performs softmax; zero scores do not imply
zero probabilities. Sparsifying the probabilities is an untested opportunity.
Bypass is skipped divided by issued plus skipped matrix instructions, not
all operator work or the fraction of runtime removed.

## Limits and verification

The intervention measures each path's net effect, not separate memory,
checking, softmax or arithmetic costs. The scope is one 14M checkpoint and
one implementation/GPU/workload; it does not establish per-operation
attribution for 70M or a general limitation of sparse attention. Run037's
absolute timings are from a separate session and do not replace historical
figure latencies. Process-extrema spans are descriptive, not confidence
intervals. Scalar replacements are included in h/z bypass counts, so the
percentage is not solely whole-zero-tile skipping.

The appendix's table descriptions are updated to the new scope, including
the untouched-K050 control, exact span construction and attribution limits.
Re-running the raw-data audit reproduces the retained Run037 exports unchanged;
the new builder reproduces every displayed number and identical table copies.
The 34-page manuscript compiles with resolved references and citations and no
overfull boxes. The subsection spans pages 8-9, with Table 2 and the conclusions
together on page 9; these pages and the revised appendix pages were visually
checked. Unrelated user edits in the current draft are preserved.
