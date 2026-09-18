# Appendix A4: Scalar zeros, instruction bypass, and projection-skipping benefit

PDF: [A4-instruction-level-explanation.pdf](../figures/A4-instruction-level-explanation.pdf).
Manuscript placement: [Results appendix](../../../manuscript/draft/results-appendix.tex),
supporting [the instruction explanation](../../../manuscript/draft/kernel-autoresearch.tex),
`sec:kernel-autoresearch` and `app:kernel-diagnostics`.

## Question, method, and coverage

How do scalar zeros and instruction bypass relate to a matched projection-skipping
ablation? Retain exactly the established Analysis021 investigation's 30-checkpoint
14M cohort: dense, ReLU, eight local L1/OL1 settings, and twenty 4/7-site
no-pressure/all-site settings. Each row has matching counters and all-skips-off
versus projection-enabled ablations in Run029. Join its historical checkpoint
evidence ID to the common table's exact source attempt/content identity. The
left fraction pools FP16 scalar zero-product counts over the four projection
families. The right fraction pools bypassed/potential BF16 projection MMAs, using
operand reconstruction for QKV/FFN-up and instrumentation for FFN-down/output.
Both panels use t0/tP from the matching retained candidate timing records.

## Publication caption

**Scalar zeros, instruction bypass, and projection-skipping benefit.** The same
30 established 14M checkpoints appear in both panels. X is the fraction of
projection scalar products with a zero operand (left) or projection MMAs bypassed
(right). Y is matched full-model latency with all skipping disabled, t0, divided
by latency with projection skipping enabled and attention skipping disabled, tP.
The horizontal line is 1×. Colors/shapes identify dense, ReLU/1-site, 4-site,
and 7-site checkpoints. Open markers denote no pressure; filled markers include
the historical local naive-L1/OL1 and multisite all-site OL1 settings. Scalar
counters use full-validation FP16 diagnostics; MMA counters reflect BF16 kernel
operands/predicates, and bypass includes replacement by scalar work rather than
only eliminated arithmetic. Timing uses RTX 5090, BF16, batch one, 2,048 tokens,
full 50,304 logits, and retained K050 ablations in the same Run029 session.
No 70M points or newly timed h-only points are added, and no historical regression
statistics are extended to a larger cohort. No independent-seed uncertainty is shown.

## Result and proposed manuscript writing

> Scalar projection zeros alone do not specify whether a kernel can omit
> matrix-multiply instructions. Fragment structure and scalar fallback affect
> the projection-skipping benefit. In the established 14M cohort at κ=.5,
> 4-site OL1(all) bypasses 82.30% of projection MMAs and achieves a 1.378×
> projection-skipping gain, compared with 57.68% and 1.325× for 7-site OL1(all).
> These matched ablations explain execution structure for this cohort; they do
> not supply an attention-skipping ablation for the new 70M measurements.

## Caveats and provenance

This diagnostic cohort intentionally differs from the 54-point primary comparison
and from the manuscript's later 35-point restored-ablation figure. No regression
or R² is copied from either cohort. Some bypasses replace MMAs with SIMT arithmetic;
the fraction cannot be equated with arithmetic eliminated. FP16 scalar and BF16
instruction counts have distinct operational meanings and are preserved separately.

Source: [paper_execution_figures.py](../paper_execution_figures.py),
[13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py),
[joined instruction records](../data/paper-checkpoints.json), and
[the original 30-checkpoint investigation](../../021-2026-09-10-training-results-figures/investigation/README.md).
Associated paragraph is proposed writing, without a manuscript edit.
