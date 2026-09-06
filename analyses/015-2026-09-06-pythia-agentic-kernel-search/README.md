# Analysis 015: agentic sparse-kernel search through Pythia-410M

Status: **complete descriptive systems case study, with a paper-planning note;
not promoted to a research finding or result-bearing manuscript claim**.

## Question

On one deployment GPU, after a bounded Sakana-derived implementation search:

1. does canonical `R_model` associate with measured batch-one full-model
   speedup within Pythia-14M, 70M, and 410M; and
2. can implementation changes alter speed at fixed checkpoint and fixed
   `R_model`?

The operational estimands remain distinct. `R_model` is the count-pooled
fraction of logical multiplication opportunities with an exact-zero operand.
Speedup is the paired native/candidate host-time ratio. Neither is treated as
the other.

## Sources and reduction

- Source run: [Run 025](../../runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/README.md).
- Frozen final policies: K013 for 14M, K016 for 70M, and K010 for 410M.
- Workload: Pythia causal-LM prefill, batch 1, sequence length 2,048, BF16,
  complete logits including the dense LM head.
- Hardware: one NVIDIA RTX PRO 4500 Blackwell under CUDA 12.8 and PyTorch
  2.11.0+cu128.
- Final timing: 16 fixed seed-2500 training-cache blocks, five passes, and 80
  paired native/candidate samples per checkpoint. Mode switches and staging
  were outside the timers.
- Numerical/quality coverage: all 338 complete MiniPile validation blocks
  (692,224 input tokens from 500 documents); the 1,444-token tail was excluded.
- Logical coverage: the source checkpoint's complete 338-block canonical pass;
  integer numerator and denominator counts are retained and rechecked.
- Uncertainty: deterministic 20,000-draw bootstrap over the 16 timing-input
  identities. These intervals do not represent fresh-process or training-seed
  variation.

[`01_reduce.py`](01_reduce.py) verifies source identities and recomputes every
reported timing estimate before producing [the complete tables](tables.md),
[`final-results.csv`](final-results.csv), [`regressions.csv`](regressions.csv),
and [`same-rmodel-optimization.csv`](same-rmodel-optimization.csv).
[`02_plot.py`](02_plot.py) creates the initial two PDFs using Analysis 010's
typography and colour family.

The independent-process extension is reduced by
[`03_reduce_replications.py`](03_reduce_replications.py) and plotted by
[`04_plot_replications.py`](04_plot_replications.py). The direct fixed-`R_model`
confirmation is reduced by [`05_reduce_fixed_rmodel.py`](05_reduce_fixed_rmodel.py)
and plotted by [`06_plot_fixed_rmodel.py`](06_plot_fixed_rmodel.py). Those
reducers retain process-level observations and never substitute timing-input
bootstrap intervals for between-process replication.

## Results

The final matrix contains 36 checkpoints. Thirty-two pass the complete
elementwise-logit and validation-loss gates. The failed final deployments are
14M K013 A7-OL1 at `kappa=0.1` and 70M K016 A4-OL1 at `kappa` 0.01, 0.05, and 0.1; they
remain visible in the figure and are excluded from qualified fits.

| Size | Qualified / total | Qualified OLS R2 | Spearman rho | Qualified points faster than native | Best qualified speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 14M | 11 / 12 | 0.235 | +0.273 | 9 | 1.0667x, A7-OL1 `kappa=0.05` |
| 70M | 9 / 12 | 0.003 | +0.067 | 2 | 1.0514x, A4-OL1 `kappa=0.5` |
| 410M | 12 / 12 | 0.611 | +0.755 | 2 | 1.0176x, A4-OL1 `kappa=0.5` |

These are within-size, one-seed descriptive fits. The 410M positive association
does not transfer to 70M, and most 410M checkpoints remain slower than native.
At 14M, eight of the eleven qualified points have an input-cluster interval
entirely above 1; the corresponding counts are one at 70M and two at 410M.

The fixed-`R_model` comparisons demonstrate implementation sensitivity. The
early 14M P0-to-K001 change improves all four development sentinels (median
speedup-ratio gain 1.0536x). The final 14M K012-to-K013 repair makes three
previously invalid deployments valid but generally gives back latency. K016 is
faster than K009 on 9/12 70M conditions, yet three transferred A4 interior
conditions fail the numerical gate. K010 improves all three recorded 410M
development comparisons; its large A0 improvement is a dense native fallback,
while the two sparse high-`kappa` gains are small.

## Independent-process and hardware replication

Three eager-only processes per checkpoint on RTX PRO 4500 and two primary
eager-only processes per endpoint sentinel on H100 NVL strengthen and narrow
the original estimates. Every primary process uses 80 paired timing samples
and passes complete validation unless it reproduces one of the four known
frozen-policy failures.

| GPU | Size | Qualified / total | Slope per +10 pp `R_model` | OLS R2 | Spearman rho | Faster than native |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| RTX PRO 4500 | 14M | 11 / 12 | +0.0347x | 0.347 | +0.573 | 9 |
| RTX PRO 4500 | 70M | 9 / 12 | +0.0036x | 0.0027 | +0.467 | 4 |
| RTX PRO 4500 | 410M | 12 / 12 | +0.0105x | 0.636 | +0.867 | 2 |
| H100 NVL | 14M | 6 / 6 | +0.0649x | 0.321 | +0.429 | 4 |
| H100 NVL | 70M | 6 / 6 | +0.0100x | 0.051 | +0.429 | 2 |
| H100 NVL | 410M | 6 / 6 | +0.0075x | 0.700 | +0.829 | 0 |

Thus `R_model` has a strong positive descriptive association at 410M, a
moderate association at 14M, and a weak/nonlinear association at 70M. The
relation is not a hardware-independent conversion law. In particular, the
high H100 410M R2 orders six implementations that are all slower than native.

Matched fixed-policy transfer also changes relative value: mean H100-minus-RTX
speedup is -0.0104x at 14M, +0.0133x at 70M, and -0.0593x at 410M. All six 410M
sentinels lose relative value on H100. Component probes show 18/18 eligible
attention-projection-only paths above break-even, versus 11/22 FFN-only paths;
the components are not additive and QK/PV remain dense.

## Direct fixed-R_model confirmation

Attempt `rtxpro4500-004` compares implementations in independent processes at
identical checkpoints, workloads, and canonical integer `R_model` counts. All
48 processes pass complete validation. The 14M primary contrast is the
adaptive direct P0-to-K001 confirmation triggered after the final robustness
policy K013 proved slower than P0; the 70M and 410M contrasts are K009-to-K016
and K004-to-K010.

| Size | Qualified process pairs | Median optimized / baseline | Range | Wins |
| --- | ---: | ---: | ---: | ---: |
| 14M | 6 | 1.0152x | 0.9932--1.0287x | 5/6 |
| 70M | 6 | 1.0113x | 0.9897--1.0869x | 5/6 |
| 410M | 6 | 1.0064x | 1.0037--1.0113x | 6/6 |

All architecture medians exceed one and 16/18 process pairs favor the
optimized implementation. Conversely, K013/P0 is 0.9370x and 0.9477x at the
two 14M endpoints. Together these results directly demonstrate that an
implementation search can improve or harm measured throughput while trained
weights and logical opportunity remain fixed.

## Figures

- [Figure 1: `R_model` versus frozen full-model speedup](figures/01-rmodel-vs-full-model-speedup.pdf)
- [Figure 2: fixed-`R_model` implementation transitions](figures/02-same-rmodel-kernel-search-transitions.pdf)
- [Figure 3: fresh-process `R_model` association across GPUs](figures/03-fresh-process-rmodel-vs-speedup.pdf)
- [Figure 4: matched frozen-policy hardware transfer](figures/04-matched-hardware-transfer.pdf)
- [Figure 5: FFN and attention-projection contribution probes](figures/05-component-contributions.pdf)
- [Figure 6: independent-process optimization at fixed `R_model`](figures/06-fixed-rmodel-kernel-optimization.pdf)

The matching figure records are in [observations](observations/INDEX.md).

## Interpretation boundary

This supports a scoped systems statement: executable benefit is specific to
GPU, shape, model scale, site/layer policy, and correctness constraints; a bad
policy can add overhead even at high logical opportunity. It does not support a
universal `R_model`-to-speed mapping or a claim that an agent outperforms human
or non-agent optimizers. There is one search trajectory and one training seed
per checkpoint. Fresh-process replication measures systems variability, not
training uncertainty.

The intended attention result was not obtained. QK-score and probability-value
matmuls remained dense SDPA for every final policy. Sparse attention projection
linears were eligible where stated, but that is not a sparse QK/PV kernel.
There is a frozen-policy H100 sentinel transfer, but no H100 retuning search and
no strongest compiled-dense comparator. Final timing used fixed training-cache
blocks rather than the pre-registered seed-2504 validation timing sample.
Complete quality validation is unaffected, but this deviation narrows the
timing claim.

## Evidence integrity

The local evidence archive is 5,840,881 bytes with SHA-256
`404d02e668c6de6b9cf2e2835bef3c166e82d61c12e04c6bfb8bfcc66466777d`.
It matches the stable remote post-package identity. Of 4,101 inventoried files,
4,100 match their self-inventory; the sole mismatch is the controller's own
Phase-13 launch log, which grew from zero to 46 bytes after self-inventory and
before archival. [`source-verification.json`](source-verification.json) records
that bounded race explicitly.

RunPod was closed before analysis: zero pods and zero endpoints remained. The
pre-existing shared 100 GB volume was intentionally retained.

The RTX three-process archive is 6,334,093 bytes with SHA-256
`74783671c3db7c795de20f19482d4db583868b0f26d781be2358a82cef839cbf`.
The H100 replication/component archive is 15,267,840 bytes with SHA-256
`0a38a9f425eb8bfffa287682ca77acdbc7814200e9b99e8202fa55a49a8462b9`.
The direct fixed-`R_model` archive is 731,849 bytes with SHA-256
`0753655f73ccd2bd8586265c1558641949fc4b6b112684570b891f3b00a815d8`.
Run-local verifiers reconcile the expected process counts, full-validation
coverage, checkpoint hashes, and identical canonical counts within each pair.
