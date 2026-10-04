# Does T2/P2 reduce inference latency relative to T2/Ph?

P2 is slightly slower at kappa=0 and slightly faster at kappa=.1 under the
frozen14M K050 implementation. Every fresh-process comparison has the same
direction within its threshold. The gain at.1 is small and accompanies a
higher validation loss.

## Method and coverage

Use the four original Run058 seed1234, step712 checkpoints without retraining,
clipping or kernel search. Both interventions gate h/z; Ph pressures h and P2
pressures h/z. A/m have no gates. All comparisons use the same physical
RTX5090, BF16, batch1,2048-token uncached causal inference and all50304 logits.
Recurring detection, packing, gates and full logits are timed; input staging,
compilation, graph setup, validation and diagnostics are excluded.

Each threshold has64 fixed validation inputs, seven randomized paired passes
and three fresh processes:1344 direct Ph/P2 latency pairs. The estimator is the
geometric mean of same-input/pass synchronized-host Ph/P2 latency ratios.
Process repetitions quantify systems variation, not training-seed variation.

Every process checks both native and K050 graphs against its own checkpoint's
native eager anchor on all500 MiniPile validation documents:338 complete blocks,
692224 input tokens and691886 prediction tokens, with the1444-token tail excluded.
The declared elementwise, relative-L2 and pooled-loss gates all pass; observed
maximum absolute and relative-L2 logit error is zero. Four final diagnostics
cover the same full validation set, retaining activation counts/norms, weight
norms, row histograms, independent MMA/SIMT work checks and the explicitly named
BF16 scalar-opportunity lower bound. Canonical Run058 FP16 logical metrics remain
separate from these operands and from measured speed.

## Result

| kappa | Ph host geomean ms | P2 host geomean ms | P2/Ph paired speedup | P2 latency reduction | Process ratio range |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.659690 | 0.664153 | 0.993279x | -0.677% | 0.988689--0.996753 |
| 0.1 | 0.625165 | 0.614790 | 1.016875x | +1.660% | 1.015749--1.018378 |

The unmodified native graph control is effectively tied, with pooled P2/Ph
ratios0.998415 and0.998565. BF16 full-validation losses are5.128739 versus5.285315
at0 and5.153716 versus5.331877 at.1 (Ph/P2). Thus the small K050 latency gain at.1
costs+.178161 loss at the matched threshold. At0, P2 costs+.156577 loss and has
higher latency. The complete table retains medians, CUDA-event timings and
input-token throughput; these are full-sequence measurements, not decoding
throughput with a KV cache.

## Figure caption and legend

[PDF figure](../figures/01-p2-ph-latency.pdf): left, full-model host geometric-mean
latency for Ph (blue) and P2 (orange); lower is faster. Right, filled circles are
three fresh-process paired ratios and the hollow diamond is the pooled paired
ratio. The dashed line marks equal latency; above1 favors P2. Every process is
numerically qualified. The left panel begins at zero; the ratio panel shows the
small differences and systems variation.

## Caveats and provenance

This is one training seed, one GPU and one frozen kernel at two thresholds.
There is no equal-loss comparison, training-speed measurement or general
hardware/decoding claim. Equal training lambda also changes the tensor-average
pressure weighting when h/z replaces h, as documented in Run058.

Slow network-backed setup, a missing historical HZ allowlist entry and duplicate
compilation are retained as infrastructure history. No scientific source or
checkpoint was corrected. Four original processes plus two bounded cached
continuations complete the same six fixed jobs. Both smoke checks pass, all99
transferred files verify, and the owned Pod was terminated before its original
deadline. Lifecycle cost is approximatelyUSD1.45; the posted ledger is incomplete.

Sources: [reduction](../20_analyze.py), [figure script](../21_figure.py),
[results and source hashes](../results/results.json),
[tables](../results/tables.md), [transfer receipt](../transfer/receipt-001.json),
[original cohort](../artifacts/scientific/summary-004.json),
[continuation](../artifacts/scientific/summary-005.json),
[combined cohort](../artifacts/scientific/cohort-summary.json), and
[compute closeout](../results/compute-closeout.json).
