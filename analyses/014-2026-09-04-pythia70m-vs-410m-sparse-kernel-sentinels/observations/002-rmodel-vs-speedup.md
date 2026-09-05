# Observation 002 - R_model versus measured speedup

## Question

How does canonical logical opportunity (`R_model`) relate to measured
full-model speedup across all available, verified Pythia sparse-kernel
realizations?

## Method and coverage

This is a visualization of existing results; no model, kernel, or timing was
rerun. It contains all 12 verified sentinel checkpoints from Runs 023 and 024:
A0, A1-H, A4-OL1 at kappa 0 and 0.5, and A7-OL1 at kappa 0 and 0.5, for each
of Pythia-70M and Pythia-410M. Both batch sizes (1 and 32) are shown, giving
24 points. These are one-seed checkpoints (seed 1234), not 24 independent
training realizations. Their source training budget was 712 optimizer
boundaries / 1,493,172,224 input tokens.

The horizontal coordinate is canonical `R_model`, computed directly from
the archived integer `block_zero_product_count / model_product_count` and
checked against `condition-comparison.csv`. This is the complete validation
diagnostic: all 500 MiniPile documents, 338 complete 2,048-token blocks,
692,224 input tokens, and an excluded 1,444-token tail. Counts pool across
blocks and layers before division; the denominator includes the dense LM
head. The same checkpoint-level horizontal coordinate appears in both batch
panels. It is the pinned canonical diagnostic, not a fresh BF16 timing-batch
zero count.

The vertical coordinate is the median of seven paired timing-block ratios,
`native_dense_ms / sparse_linear_ms`; it is not a ratio of independently
aggregated medians. Whiskers show the archived 10th and 90th percentiles of
those same seven ratios. The exported CSV retains all 168 ratios as well as
the medians, intervals, counts, raw-source paths, and SHA-256 hashes.

All timing uses the same physical H100 NVL, BF16, and uncached 2,048-token
prefill, with randomized variant order within paired timing blocks. It
includes packing and sparse-linear execution in the full-model forward path.
A0/A1-H replace only h-to-W2; A4/A7 replace four linear operation families.
Attention remains dense SDPA and the LM head remains dense in every timed
full-model sparse path. The source run READMEs contain the exact benchmark
protocol and validation gates.

### Explicit exclusions

- Run 022's 14M sparse path failed the N=128 correctness gate, so there is no
  valid 14M full-model sparse speedup to plot. It is not plotted as zero.
- Interior A4/A7 kappa values and post-hoc TEAL points have no completed,
  verified full-model sparse timing in these runs. No runtime is inferred.
- Standalone linear primitives and A7 attention compositions have different
  runtime denominators; they are not mixed with full-model speedup.
- The official SparseLM0.5B positive control is not one of the trained Pythia
  realizations and has no matched canonical `R_model` here.
- Failed/recovered infrastructure attempts do not count as new independent
  realizations. Only the verified final run artifacts are used.

## Figure caption and legend

**Canonical logical opportunity versus measured full-model speedup.** Purple
denotes Pythia-70M and blue denotes Pythia-410M. Panels (a) and (b) show batch
sizes 1 (circles) and 32 (triangles), respectively, on identical axes. Every
verified checkpoint is labelled; A4 and A7 abbreviate A4-OL1 and A7-OL1.
Points are medians of seven paired native-dense/sparse-linear timing ratios;
whiskers are their 10th-90th percentiles, sometimes smaller than the markers.
The dashed line marks 1x break-even; values below it are slowdowns. Horizontal
coordinates use complete-validation, integer-pooled canonical `R_model`.
Runtime is BF16 full-sequence prefill on the same H100 NVL, with sparse linear
operations but dense attention and LM head. All 24 valid checkpoint/batch
measurements are shown; 14M has no valid sparse timing after its correctness
gate failed.

## Result

Every median is below break-even. The 70M results span 0.1832-0.9835x at batch
one and 0.2523-0.7066x at batch 32. The 410M results span 0.0485-0.3275x and
0.1108-0.4989x, respectively.

Larger `R_model` does not map to a universal speedup across these endpoints.
For example, at batch one, 70M A4-OL1 kappa=0.5 has `R_model=0.3560` and
speedup 0.9835x, whereas 410M A7-OL1 kappa=0.5 has `R_model=0.8062` and
speedup 0.2805x. This is descriptive evidence about the tested implementation,
not a controlled causal comparison between those two conditions.

## Caveats and nonclaims

- Timing intervals describe seven repeats on one device, not confidence
  intervals over training seeds, hardware, or datasets.
- Model size, topology, quality, sparsity distribution, kernel shapes, and
  sparse-operation coverage differ. No regression or scaling law is fitted.
- Canonical `R_model` includes logical opportunity in operations the timed
  sparse path does not exploit, including attention. It is not removed FLOPs
  or a prediction of speedup.
- The figure does not measure decoding, other GPUs, approximate pruning, or a
  fused sparse-attention integration. It does not establish that other sparse
  implementations cannot exploit the observed zeros.
- The Run-024 A7 high-threshold extended-attention coverage caveat described
  in Observation 001 does not affect canonical `R_model` or these measured
  full-model timings; the extended coverage metric is not plotted.
- No manuscript text or consolidated finding is changed by this visualization.

## Provenance and reproduction

- Script: `../02_plot_rmodel_speedup.py`
- PDF: `../figures/01-rmodel-vs-speedup.pdf`
- Plot data: `../rmodel-speedup-points.csv`
- Cross-check table: `../condition-comparison.csv`, generated by `../01_compare.py`
- 70M raw inputs: `../../../runs/023-2026-09-04-pythia70m-sakana-sparse-kernel-sentinels/results/raw/`
- 410M raw inputs: `../../../runs/024-2026-09-04-pythia410m-sakana-sparse-kernel-sentinels/results/raw/`

From the repository root:

```powershell
python analyses/014-2026-09-04-pythia70m-vs-410m-sparse-kernel-sentinels/02_plot_rmodel_speedup.py
```

The generator checked all 24 coordinate pairs against verified raw sources.
The single-page PDF was rendered with Poppler and visually inspected for
label collisions, clipping, and legibility.
