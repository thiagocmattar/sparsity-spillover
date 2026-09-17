# 002 - Absolute native and final K050 latency

**Question.** Does the native-relative speedup also correspond to lower
absolute full-model latency, and how does the reference vary by topology?

**Method and coverage.** The same 40 qualified checkpoints and raw timing pairs
as [001](001-sparsity-speedup.md). Each absolute latency is the geometric mean
of all 1,344 host timings for that implementation. This is the raw-latency
estimand used by Analysis021's latency investigation; it is not the original
Run029 summary's geometric mean of three process medians. No old model was
rerun. Numerical qualification covers all 338 blocks from 500 documents for
every process, with the 1,444-token tail excluded.

**Caption and legend.** Native PyTorch SDPA CUDA-graph latency (left) and final
K050 CUDA-graph latency (right) against canonical FP16 model-wide logical
sparsity. BF16, RTX5090, B=1, T=2048, all 50,304 logits; units are milliseconds
per full-sequence forward. Axes use the same latency scale with zero shown.
Colors, marker fills and line styles follow Figure01: blue A4, orange A7;
solid no OL1, dashed all-site OL1, dotted h-only OL1; lines connect the five
thresholds in increasing order. A0 and A1-H controls are retained references.
The five A7 h-only points come from a second physical RTX5090/host session.

**Result.** New A7 h-only K050 latency decreases across kappa from 0.617605 to
0.473888 ms. Native latency is 0.744850 ms at kappa=0 and approximately
0.841--0.848 ms at positive thresholds. At kappa=0.5, the retained all-site A7
endpoint is 0.473366 ms, close to the new h-only endpoint's 0.473888 ms.
The historical A4 h-only endpoint is 0.462252 ms: its smaller 1.599928x
native-relative speedup does not imply larger absolute latency than A7 h-only,
whose native reference has the additional A7 gate cost.

**Limits.** Absolute cross-session latency can shift with physical GPU/host
conditions. These close A7 endpoint values are not a formal equivalence claim.
The reference is checkpoint-specific native execution; no new common A0
numerator was measured or substituted. Logical opportunity and wall time are
different estimands, and no projection/attention ablation is inferred here.

**Source and output.** `../01_reduce.py`, `../data/results.json`, and
`../02_plot.py`; output `../figures/02-14m-k050-sparsity-latency.pdf`.
Rendered and visually checked; embedded TrueType fonts, no point annotations.
