# 010: Speedup over size-matched A0 versus model-wide sparsity

## Question

Normalize Figure09's full-model latencies separately for 14M and 70M using
each size's A0 baseline, and show speedup on one panel.

## Method and coverage

Retain 42 qualified points and all eight family curves from
`data/matched-combined-latency.json`: A0 and A4/A7 with OL1(h)/OL1(all)
at kappa `{0, 0.01, 0.05, 0.1, 0.5}`. At the user's request, exclude
the two unpressured A1-H ReLU controls (one per size) from the 44-point source.
This leaves 21 matching recipe/kappa conditions per size.
For checkpoint c of size s, plot

```text
speedup_vs_A0(s, c) = latency_final_kernel(s, A0) / latency_final_kernel(s, c).
```

This ratio expresses speedup: values above one indicate lower latency than
A0. The inverse, checkpoint latency divided by A0 latency, would instead
express normalized latency. Both A0 controls equal exactly 1x. Use the final
candidate kernel's A0 latency, consistent with the raw-latency figure.

| Model size | Reference | Session | A0 final-kernel latency (ms) |
| --- | --- | --- | --- |
| 14M | c01 | Run029 | 0.651573035007824 |
| 70M | c00 | Run035 | 3.313247016348893 |

Each latency is the retained geometric mean of 1,344 raw synchronized host
timings: 64 fixed inputs, seven paired passes, and three fresh processes.
The ratio uses those full-precision means without rounding or mixing model
sizes. Timings use RTX5090, BF16, batch one, uncached causal sequence length
2,048, and all 50,304 vocabulary logits. Qualification covers 338 complete
validation blocks from all 500 documents, with the 1,444-token tail excluded.
The x coordinate remains the canonical FP16 integer-pooled model-wide
logical-product sparsity. No new measurement or cloud compute is performed.

## Legend and caption

**Speedup over size-matched A0 versus model-wide sparsity for Pythia-14M and
Pythia-70M.** Each checkpoint's speedup is its model size's final-kernel A0
latency divided by its own final-kernel latency. Open markers indicate 14M;
filled markers indicate 70M. Colors and shapes distinguish A0, 4-sites A4*
and 7-sites A7. Short dashes identify OL1(h), and long dashes
identify OL1(all). Each family curve connects increasing kappa within one
model size and recipe. Both axes are linear, with y ticks labeled as speedup
multiples and a horizontal 1x guide. Subtle 14M and 70M labels sit above
their respective multisite groups. Both A0 points are at 1x and near zero sparsity;
the larger open 14M star surrounds the smaller filled 70M star so both are
retained at their true coordinates. Baseline controls have no curves.

## Result

The normalized view removes the absolute latency offset between sizes and
shows relative changes from their own A0 endpoints. The maximum observed
A0-relative speedup is 1.4180795335x at 14M and 2.0472741151x at 70M.
Equal vertical distances now represent equal differences in speedup multiples.
All 42 plotted sparsity and latency values remain recoverable in the normalized
figure's data JSON. The two excluded controls remain available in Figure09's
source data. Neither the normalization nor any retained point value changes.

## Caveats

These ratios are comparisons across checkpoints to a shared A0 reference.
They are not the earlier paired, same-checkpoint native-reference kernel
speedups. In particular, the 70M candidate A0 takes about 3.31 ms whereas
its native reference takes about 1.66 ms; a roughly 2x improvement over
candidate A0 therefore must not be reported as roughly 2x over native.
Normalizing by candidate A0 does not establish equal-quality performance,
causal attribution to sparsity, or an implementation-independent size effect.

14M uses K050; 70M uses the qualified `k050-70m-v2` shape port, without an
equal optimization-search budget. Hardware/host sessions differ. The 14M
A7 h-only endpoints use Run033 while their A0 uses Run029, so that comparison
also crosses sessions. No uncertainty intervals are inferred: a common
estimated reference introduces shared uncertainty across each size's ratios.
The source observations' training-seed and workload limits remain applicable.

## Sources and reproduction

Run `10_plot_a0_normalized_speedup.py` to produce
[Figure10](../figures/.archive/10-14m-70m-matched-sparsity-a0-speedup.pdf) and
[`data/matched-a0-normalized-speedup.json`](../data/matched-a0-normalized-speedup.json).
The JSON records the formula, exact A0 references, raw latency and normalized
speedup for all plotted points, the two excluded identities, unchanged family
connections, group-label positions, source hashes, axis limits and PDF hash.
Source definitions remain in
[Observation009](009-matched-combined-latency.md),
[Observation003](003-final-latency-topology.md) and
[Observation004](004-70m-final-latency-topology.md).

Verification checks both A0 controls equal 1x, all ratios reconstruct from
the retained latencies, exact 21-key matching per size, 42 unique point
identities, eight separate curves, linear axes, both group labels and source hashes.
The single-page PDF was rendered and visually inspected, and all fonts are
embedded. Earlier PDFs and data are unchanged. No manuscript update or
finding promotion is made.

The current display revises Figure10 in place at the user's request: remove
the 1-site controls, label the model-size groups, and use a linear y axis.
The original 44-point logarithmic display is preserved in Git commit `3924817`.
