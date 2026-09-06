# Analysis 015: agentic sparse-kernel search through Pythia-410M

Status: **complete, descriptive, and not promoted to a finding or manuscript
claim**.

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
[`02_plot.py`](02_plot.py) creates the two PDFs using Analysis 010's typography
and colour family.

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

## Figures

- [Figure 1: `R_model` versus frozen full-model speedup](figures/01-rmodel-vs-full-model-speedup.pdf)
- [Figure 2: fixed-`R_model` implementation transitions](figures/02-same-rmodel-kernel-search-transitions.pdf)

The matching figure records are in [observations](observations/INDEX.md).

## Interpretation boundary

This supports a scoped systems statement: executable benefit is specific to
GPU, shape, model scale, site/layer policy, and correctness constraints; a bad
policy can add overhead even at high logical opportunity. It does not support a
universal `R_model`-to-speed mapping or a claim that an agent outperforms human
or non-agent optimizers. There is one search trajectory, one GPU, one training
seed per checkpoint, and no matched fresh-process replication.

The intended attention result was not obtained. QK-score and probability-value
matmuls remained dense SDPA for every final policy. Sparse attention projection
linears were eligible where stated, but that is not a sparse QK/PV kernel.
There is also no frozen-kernel H100 transfer matrix and no strongest compiled
dense comparator. Final timing used fixed training-cache blocks rather than the
pre-registered seed-2504 validation timing sample. Complete quality validation
is unaffected, but this deviation narrows the timing claim.

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
