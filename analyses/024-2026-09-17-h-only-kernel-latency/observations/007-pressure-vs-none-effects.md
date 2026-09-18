# 007 - Pressure versus no pressure: paired distributions

**Question and request.** What are the matched effects of adding all-site OL1
or h-only OL1 to four- and seven-site gated models? The user requested a new
2-by-3 version of Figure06, with OL1(all) minus no pressure in the first row
and OL1(h) minus no pressure in the second. Figure06 is retained.

**Method.** Within each gate topology and kappa in {0, 0.01, 0.05, 0.1, 0.5},
subtract the no-pressure endpoint from the corresponding pressured endpoint.
There are 20 contrasts from 30 unique final checkpoints. The top and bottom
rows reuse the same ten no-pressure references. Initialization, training
order, gate sites/operators, threshold, and training budget are matched.
The active OL1 recipes use lambda=1 and trust budget b=1. All-site targets
are a,m,h,z for four-site gates and a,m,h,q_post,k_post,v,z for seven-site
gates; h-only pressure targets h under either topology.

The script re-pairs Figure05's retained endpoint data, validates its original
source reduction hashes and records that intermediate source hash. All loss
values are ordinary final-validation loss. Sparsity is the pooled integer
zero-product count divided by the full model-product count, displayed as a
percentage; differences are percentage points. Latency is the difference of
geometric means of 1,344 K050 host timings per checkpoint, in microseconds.
For each pressure/topology combination, the mean paired difference is also
the equally weighted pressure coefficient with kappa block intercepts.

**Coverage.** Seed 1234, matched initialization and training-order hashes,
712 updates and 1,493,172,224 training input tokens per model. Quality and
sparsity use all 500 validation documents and 338 complete 2,048-token blocks,
692,224 input tokens; the excluded tail is 1,444 tokens. K050 timing uses
64 common validation inputs, seven passes and three processes, RTX 5090,
BF16, batch 1, sequence length 2,048 and all 50,304 output logits. No new
training or timing measurement is made.

**Caption and legend.** Paired pressure effects relative to no pressure in
Pythia-14M. Top row: OL1(all) minus no pressure. Bottom row: OL1(h) minus no
pressure. Columns show validation-loss difference (nats/token), model-wide
logical-sparsity difference (percentage points) and K050 full-model latency
difference (microseconds). Each panel contains four-site (blue) and seven-site
(orange) distributions, each with five matched kappa contrasts. Boxes span
the 25th to 75th percentiles, bars mark medians and whiskers show the complete
range. All five differences are dots with small fixed vertical offsets.
Linear quantiles give the second/fourth order statistics as box edges for
five observations. Dashed vertical lines mark zero. Axes share identical
scales between the two rows within each metric. Negative loss/latency and
positive sparsity favor pressure on the corresponding metric. Boxes describe
variation across the fixed kappa settings, not confidence intervals.

**Result.** H-only OL1 lowers loss at kappa <= 0.1 under both topologies,
while raising it at kappa=0.5. It increases logical sparsity and lowers
observed K050 latency in all ten pairs. All-site OL1 lowers loss in only
2/10 pairs (four-site kappa=0 and 0.01); it increases logical sparsity in
9/10 pairs and lowers observed latency in 8/10. The two positive all-site
latency differences are tiny, both below 0.07 microseconds, for seven-site
kappa=0 and 0.01. Equal-weight means are below; plotted bars show medians.

| Pressure minus none | Topology | Mean loss difference | Mean sparsity difference (pp) | Mean latency difference (microseconds) |
| --- | --- | ---: | ---: | ---: |
| OL1(all) | 4-sites | +0.108426 | +1.792154 | -29.898175 |
| OL1(all) | 7-sites | +0.036193 | +2.829209 | -1.527357 |
| OL1(h) | 4-sites | -0.176647 | +0.720313 | -43.918587 |
| OL1(h) | 7-sites | -0.182877 | +1.034939 | -33.805337 |

**Caveats.** One shared training seed does not establish seed uncertainty or
statistical significance. The 20 contrasts reuse ten no-pressure endpoints,
so the two pressure rows are not independent samples. Boxes reflect a fixed
five-point kappa grid, whose high-threshold endpoints can dominate means.
Only the seven-site h-only latency comparisons span Run033 and Run029
physical GPU/host sessions; all other pairs share Run029. Therefore that
cross-session timing difference cannot be attributed solely to pressure.
Logical sparsity is not measured acceleration. No manuscript edit or finding
promotion is made.

**Source and output.** `../07_plot_pressure_vs_none.py` reads
`../data/paired-topology-effects.json`, retaining the original quality/sparsity
source from Analysis023 and timing source from Analysis024. Outputs are
`../data/pressure-vs-none-effects.json` and
`../figures/.archive/07-14m-pressure-vs-none-effects.pdf`. The machine-readable file
retains all endpoints, 20 differences, exact counts and 12 box summaries.

**Verification.** Independent joins against the original quality and timing
reductions verify all 30 endpoint identities, fixed gate/pressure scopes,
the 20 contrasts, shared references and GPU-session boundaries. All 60
plotted values are within the shared column scales. Independent sorted-value
checks reconcile the 12 box summaries; regenerated PDF and JSON are
byte-identical. The PDF was rendered and visually inspected and all fonts
are embedded. Figure06 remains unchanged.
