# Kernel mechanism evidence for the manuscript

## Question and scope

Why does QK/PV bypass not reduce measured latency while h/z skipping does?
At the user's request, replace the main-text bypass bar chart with a compact
table connecting per-operation instruction counts to retained timing controls.
No new measurements, kernel changes, training or cloud work are included.

Sources: [Observation 027](027-operation-bypass.md),
[table builder](../24_build_kernel_mechanism_table.py),
[unrounded export and hashes](../data/kernel-mechanisms.json), and
[analysis-owned TeX table](../data/kernel-mechanisms.tex).
The manuscript copy is `manuscript/draft/tables/kernel-mechanisms.tex`;
the associated writing is `manuscript/draft/kernel-autoresearch.tex`.

## Method and coverage

Select Run029 c35 (14M T4/Ph) and c30 (14M T7/Pall), both kappa=0.5.
Counters pool all six layers and all 338 complete 2048-token blocks from 500
MiniPile validation documents: 692224 input tokens, 1444-token excluded tail.
Rebuild the twelve operation counters from their original diagnostics and
recompute six full-model geometric-mean latencies from 8064 raw candidate
host-time samples. Also verify their paired native timings, checkpoint hashes,
64 input identities, seven passes and three fresh processes per mode.

The denominator is issued plus bypassed matrix instructions, per operation.
It includes padded work; h/z bypass includes scalar substitution. Attention
bypass includes causal masking/padding; the base-model PV reference is 10.4%.
It is neither a percentage of all operator work nor a fraction of latency saved.
Timing uses BF16 CUDA graphs on RTX 5090, batch one, length 2048 and the full
50304-token vocabulary output. All-off retains the optimized K050 implementation.

## Table caption and results

**Where the kernel skips work, and where it saves time.** Pythia-14M,
kappa=0.5, frozen K050. Six rows report instruction bypass and three rows
report full-model latency with all skipping off, projection skipping only,
and projection plus attention skipping. Both columns retain exactly the same
checkpoint throughout their timing controls. This table replaces the figure;
the earlier figure PDF and evidence remain unchanged in the analysis.

- T7/Pall: a/m/h/z/QK/PV bypass is 8.4/0.0/94.5/99.6/58.0/67.2%.
  Projection skipping reduces 0.620710 to 0.468365 ms; adding attention
  skipping increases this to 0.473366 ms.
- T4/Ph: a/m bypass is zero, h/z bypass 98.4/99.7%; projection skipping
  reduces 0.623390 to 0.455867 ms, a 1.36748x gain. This supports the joint
  h/z explanation without claiming separate measured h and z contributions.
- All 35 retained 14M attention-toggle comparisons are slower with the
  attention skip path enabled. The table shows two illustrative endpoints.

## Mechanism and limits

Code sources in Run028: `candidates/k049/joint.cu` (h/z active-group weight
loads and eligible short-row substitution), `k042/projection.cu` (a/m tests
after operands are loaded), and `k035/sparse_gemm.h` plus
`k035/flash_fwd_kernel.h` (FlashAttention tile loads, QK, masking, softmax, PV).
These establish which load requests are avoidable, not measured DRAM traffic.
FlashAttention background is cited to Dao et al., arXiv:2205.14135 (2022).

Softmax remains common work and does not carry zero scores into zero
probabilities. V zeros remain useful in PV. Sparsifying the probabilities
is described only as an untested opportunity, not an implemented improvement
or a guarantee of latency savings. Sparse attention is not declared impossible.

Retained controls do not isolate each site's latency or each overhead source,
and do not establish timing attribution for 70M. The proposed new single-
checkpoint operation-toggle experiment is separate:
[design approved at kappa=0.5](../PER-SITE-LATENCY-DESIGN.md).

## Verification

The builder checks pooled integer counters, source identities and full raw
timing coverage before writing either table copy. Manuscript build and visual
verification are recorded at closeout in the manuscript README. The canonical
34-page PDF compiles with resolved references and citations and no overfull
boxes; subsection and Table 2 are together on page 9, visually checked along
with the new bibliography entry on page 10.
