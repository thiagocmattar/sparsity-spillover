# Observation 001: `R_model` is not a hardware-independent speed predictor

## Question

After freezing a separate specialized policy for each Pythia scale, does higher
canonical `R_model` associate with higher batch-one full-model speedup?

## Method and coverage

Figure 1 contains all 36 final Run 025 checkpoint realizations: 12 each for
Pythia-14M, 70M, and 410M. Timing is on one NVIDIA RTX PRO 4500 Blackwell at
batch 1 and sequence length 2,048, including the dense LM head. Each point is
the geometric mean of 80 paired native/candidate host-time ratios from 16 fixed
inputs and five passes. Error bars resample input identities. Canonical
`R_model` is recomputed from the complete 338-block integer logical-product
counters.

Numerical correctness and quality were separately checked over all 338 complete
validation blocks. Thirty-two deployments pass. Four failed deployments remain
plotted with red crosses; qualified regression lines exclude them. Each line is
an unweighted OLS fit with intercept within one model size. Spearman rho is
reported alongside R2 because the checkpoint ladder is small and nonlinear
responses are plausible.

## Legend and caption

Colour identifies model size; marker identifies A0, A1-H, A4-OL1, or A7-OL1.
Filled points pass the complete gate, while red crosses fail. The dashed
horizontal line is native break-even. The plot asks whether the logical
opportunity measure covaries with realized execution under each frozen policy;
it does not equate the two quantities.

## Result

The qualified association differs sharply by scale: 14M has `R2=0.235` and
`rho=+0.273`; 70M has `R2=0.003` and `rho=+0.067`; 410M has `R2=0.611` and
`rho=+0.755`. Thus the 410M frozen policy yields a positive descriptive
association, the 70M policy yields essentially none, and 14M is weak.

Absolute gain is also non-monotone with scale. The best qualified results are
1.0667x at 14M, 1.0514x at 70M, and 1.0176x at 410M. Only two of twelve 410M
points break even despite its stronger fitted association. Logical opportunity
can help rank conditions under one implementation without guaranteeing that
the implementation amortizes its overhead.

## Caveats

There is one GPU, one training seed per checkpoint, one timing process/session
per final matrix, and no frozen H100 transfer. Input-cluster intervals are not
fresh-process confidence intervals, and OLS fits over 9--12 checkpoints are
descriptive rather than causal or scaling-law estimates. Final timing uses
training-cache blocks rather than the planned validation timing sample. QK/PV
attention remained dense, so this figure cannot establish an attention-kernel
benefit.

## Source

- Figure: [`../figures/01-rmodel-vs-full-model-speedup.pdf`](../figures/01-rmodel-vs-full-model-speedup.pdf)
- Generator: [`../02_plot.py`](../02_plot.py)
- Reduced results: [`../final-results.csv`](../final-results.csv)
- Regression audit: [`../regressions.csv`](../regressions.csv)
- Reduction/verification: [`../01_reduce.py`](../01_reduce.py)
