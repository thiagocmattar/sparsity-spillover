# Kernel realization, retained matched study

## Question

Can a specialized implementation realize measured inference gains from the observed opportunity?

## Method

Copy Run 029's final 04-kernel-autoresearch-and-rmodel-r03.pdf byte-for-byte and include its matched report. No timing is rerun or historical denominator recombined. The historical search checkpoint is fixed A7-OL1 kappa=.5 (c30); the final implementation is K050, eligible proposal ordinal 42.

## Coverage

One physical RTX5090; BF16, batch 1, uncached T=2048, full vocabulary logits. Native SDPA CUDA-graph same-checkpoint denominator. Timings cover 64 fixed inputs, seven paired passes, three fresh processes. Numerical qualification covers all 338 blocks from 500 validation documents (1,444-token tail excluded). Final cohort: 35 checkpoints. Retrospective: 42 eligible proposals, 1,173 total scheduled process outcomes across phases.

## Figure and caption

[Publication PDF](../figures/07-kernel-realization.pdf)

**Matched kernel development and sparsity-associated acceleration.** Left: gray points show 214 timed candidate/checkpoint comparisons, including 74 numerical failures; they are not all qualified accelerations. The blue curve tracks the best fully qualified incumbent at fixed c30, initialized at native 1x. P0 is a separate comparator; two unsupported comparisons lack timings. Right: final K050 results across all 35 checkpoints with unweighted OLS and an estimated intercept. R_model equals the manuscript's S_model; plotted ticks are percentages, but regression uses fractions. Both axes are linear, with a restricted right-panel speedup range.

## Result

The matched qualified incumbent reaches 1.7830x; final K050 qualifies at 35/35 with geometric mean 1.2502x and R-squared .7813. Relative to native-normalized fused no-skip results, enabling sparse paths yields 1.0564x geometrically averaged acceleration and helps 19/35 checkpoints. Disabling attention skipping improves all 35. P0 qualifies on only five checkpoints; a five-point geomean is not a whole-cohort comparison.

## Caveats

Figure copied unchanged, including its legacy R notation. Ablation ratios compare separately randomized native-normalized fresh-process measurements, not direct within-process toggles. FP16 canonical R_model differs from BF16 executed opportunity. Model quality, topology and weights vary across checkpoints, so this is not causal sparsity-speedup proportionality or equal-quality acceleration. No claim of cached-decoding transfer, larger-scale transfer, or agent superiority over human kernel developers follows.

## Source script and evidence

Copy step in `plots.py:make_figures`, invoked by `01_build.py`. Original source: `runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/12_figures.py`; frozen final r03 PDF and `results/report-001.json` hashes in `figure_data.json`.

Paths above are relative to the analysis root or repository root as named.
Complete count-derived values and source hashes are in [figure_data.json](../figure_data.json).
