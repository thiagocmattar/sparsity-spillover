# 006 - Paired pressure-scope distributions

**Question and request.** What changes when OL1 pressure expands from h only
to every gated site, holding four- or seven-site gate topology fixed? The user
requested the same blocked comparison and 1-by-3 distribution figure as
Observation005, now with four-site and seven-site rows.

**Method.** At each kappa in {0, 0.01, 0.05, 0.1, 0.5}, subtract the h-only
endpoint from the all-site endpoint within each topology. The ten pairs use
20 retained final checkpoints. Four-site pressure expands from h to a,m,h,z;
seven-site pressure expands from h to a,m,h,q_post,k_post,v,z. Gate sites,
gate operators, kappa, lambda=1, trust budget b=1, initialization, training
order and training budget remain matched. Pressure target composition and
the equal-tensor objective normalization change. Within each topology, the
mean difference is also the coefficient of an all-site indicator in a model
with five kappa block intercepts and equal weight per condition.

Figure05's retained endpoint data supply uniform ordinary final-validation
loss, integer logical-product counts and K050 geometric-mean latency. The new
script verifies the original reduction hashes and records the intermediate
source hash. Sparsity differences are in percentage points; latency differences
are in microseconds per full-sequence forward. Positive loss and latency mean
that all-site pressure has worse quality and longer observed latency; positive
sparsity means more logical zero-product opportunity.

**Coverage.** One shared seed (1234), the matched initialization/data-order
identities, 712 updates and 1,493,172,224 training input tokens per model.
Quality and sparsity cover 500 MiniPile validation documents, all 338 complete
2,048-token blocks and 692,224 input tokens; the excluded tail is 1,444 tokens.
K050 latency uses 64 common validation inputs, seven passes and three processes,
1,344 host timings per checkpoint; RTX 5090, BF16, batch 1, sequence length
2,048 and all 50,304 output logits. No new measurement is made.

**Caption and legend.** Paired effect of broadening OL1 pressure from h to
all gated sites at matched kappa. Rows show four-site (blue) and seven-site
(orange) gate topologies; each has five paired differences. Columns show
validation loss (nats/token), model-wide logical sparsity (percentage points)
and K050 full-model latency (microseconds). Boxes span the middle 50%, bars
mark medians and whiskers span the complete observed range. Dots show all
five kappa contrasts with small fixed vertical offsets. Quantiles use NumPy's
linear method, making the box edges the second and fourth order statistics
for these five observations. Dashed vertical lines mark zero. These boxes
describe variation across fixed kappa settings, not confidence intervals.

**All matched contrasts, OL1(all) minus OL1(h).**

| Gate topology | kappa | Loss difference | Sparsity difference (pp) | Latency difference (microseconds) |
| --- | ---: | ---: | ---: | ---: |
| 4-sites | 0 | +0.242422 | -0.599371 | +29.623984 |
| 4-sites | 0.01 | +0.253774 | +0.000906 | +27.976895 |
| 4-sites | 0.05 | +0.294213 | +1.420816 | +10.878895 |
| 4-sites | 0.1 | +0.319637 | +2.050796 | +4.398661 |
| 4-sites | 0.5 | +0.315321 | +2.486058 | -2.776373 |
| 7-sites | 0 | +0.260357 | -1.345151 | +16.424671 |
| 7-sites | 0.01 | +0.277695 | -1.132669 | +34.708793 |
| 7-sites | 0.05 | +0.267852 | -0.263037 | +59.643029 |
| 7-sites | 0.1 | +0.192102 | +0.893163 | +51.135170 |
| 7-sites | 0.5 | +0.097340 | +10.819044 | -0.521765 |

**Equal-weight blocked summaries.** Means below summarize the five paired
differences per topology; figure bars instead show medians. The relative
latency column uses the geometric mean of all-site/h-only latency ratios.

| Gate topology | Mean loss difference | Mean sparsity difference (pp) | Mean latency difference (microseconds) | Geometric mean latency change |
| --- | ---: | ---: | ---: | ---: |
| 4-sites | +0.285073 | +1.071841 | +14.020412 | +2.336933% |
| 7-sites | +0.219069 | +1.794270 | +32.277980 | +5.547190% |

**Result.** Broadening pressure raises validation loss in all ten comparisons.
It increases logical sparsity in 6/10 pairs: 4/5 under four-site gates and 2/5
under seven-site gates. Under seven-site gates the median sparsity difference
is negative (-0.263037 pp), while the positive mean is driven by the +10.819044 pp
endpoint at kappa=0.5. Observed latency is higher in 8/10 pairs; both kappa=0.5
contrasts are slightly negative. At that threshold, seven-site all-pressure
adds +10.819044 pp sparsity for +0.097340 loss and -0.521765 microseconds of
observed latency. The earlier high-threshold quality improvement from expanding
topology under all-site pressure does not establish a quality benefit from
expanding pressure at fixed topology; these are different contrasts.

**Caveats.** All conditions share a single training seed; these are ten
matched settings, not ten independent seed replicates. No statistical
significance or seed-level confidence interval is claimed. Four-site latency
pairs share the Run029 GPU/host session, while seven-site all/h-only latency
pairs compare Run029/Run033 physical sessions. In particular, the tiny
seven-site negative latency difference at kappa=0.5 is not an established
speed benefit or formal equivalence result. Gate topology is held fixed, but
pressure composition and objective normalization change. Logical sparsity is
not measured acceleration. No manuscript edit or finding promotion is made.

**Source and output.** `../06_plot_paired_pressure.py` reads
`../data/paired-topology-effects.json`, retaining that file's original source
chain through Analysis023 quality/sparsity and Analysis024 timing reductions.
It writes `../data/paired-pressure-effects.json` and
`../figures/06-14m-paired-pressure-effects.pdf`.

**Verification.** Independent checks rejoin all 20 endpoints directly from
the original quality and timing reductions, verify matched gate/pressure
identities, recompute the ten contrasts and six sets of sorted box statistics,
and confirm the source hashes. All ten pairs and 30 plotted values are present
within the axes. Re-running the script reproduces the PDF and JSON bytes.
The PDF was rendered and visually checked; all fonts are embedded.
