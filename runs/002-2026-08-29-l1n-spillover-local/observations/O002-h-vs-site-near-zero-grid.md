# O002 - Sitewise L1N spillover trajectories

## Question

How do the individual `q_post`, `k_post`, `v`, and `m` near-zero responses vary
with near-zero mass at pressured `h`, and how do those trajectories differ for
GeLU and ReLU?

## Method and coverage

The four panels use the ten matched final Run 002 checkpoints. Every coordinate
is measured over all 500 MiniPile validation documents, 338 complete 2,048-token
blocks, 692,224 input tokens, all six layers, and the declared 1,444-token
excluded tail. The threshold is `abs(x) <= 1e-3`. Integer hits and denominators
are pooled across validation batches and layers before division; there is no
condition, site, or seed averaging.

The original terminal diagnostic supplies `h`, `q_post`, `k_post`, and `v`.
Because `m` was not selected in the approved pre-launch diagnostic set, it was
measured post hoc by reloading each retained, hash-verified final checkpoint and
rerunning the same complete validation pass. Operational `m` is the MLP-branch
LayerNorm/gate output feeding W1. Every post-hoc loss reproduced its stored final
validation loss within `1e-5`, and all ten `m` records contain six layer rows
whose integer counts reconcile to the pooled row.

## Values

| Activation | Pressure label | h (%) | q_post (%) | k_post (%) | v (%) | m (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GeLU | control | 0.462031 | 0.078658 | 0.084344 | 0.102682 | 0.079735 |
| GeLU | lambda=0.1 | 0.866986 | 0.078563 | 0.080600 | 0.144588 | 0.089122 |
| GeLU | lambda=0.5 | 3.293959 | 0.082259 | 0.087731 | 0.157389 | 0.097652 |
| GeLU | lambda=1 | 4.680761 | 0.081033 | 0.079200 | 0.188676 | 0.103057 |
| GeLU | lambda=5 | 18.525665 | 0.079350 | 0.072975 | 0.136680 | 0.156633 |
| ReLU | control | 70.290451 | 0.065143 | 0.076149 | 0.093194 | 0.083518 |
| ReLU | lambda=0.1 | 86.452467 | 0.076135 | 0.085318 | 0.140040 | 0.092960 |
| ReLU | lambda=0.5 | 93.777274 | 0.085382 | 0.087798 | 0.180118 | 0.084718 |
| ReLU | lambda=1 | 96.236050 | 0.079163 | 0.085610 | 0.208567 | 0.084331 |
| ReLU | lambda=5 | 99.456826 | 0.078588 | 0.078670 | 0.132252 | 0.088266 |

## Figure caption and encoding

**Figure 02. Sitewise near-zero mass versus pressured FFN `h` for local
Pythia-14M L1N pretraining.** The 2x2 panels show `q_post`, `k_post`, `v`, and
`m`. Blue circles/solid lines denote GeLU; orange squares/dashed lines denote
ReLU. Point labels are the no-pressure control (`ctrl`) or naive-L1 weight.
Lines connect control through increasing pressure strength as visual guides,
not fitted relationships. All panels use the same zero-based x and y scales and
report percent at epsilon `1e-3`.

## Interpretation and limits

The panels separate the attention-site average shown in Figure 01 and add the
MLP input branch. They are descriptive trajectories from one seed and a short
training horizon. GeLU-versus-ReLU differences include the activation operator
change. Near-zero activation mass is not an exact-zero product count, removable
compute, measured speedup, a causal route, or a long-horizon optimum. The source
attempt evidence is `valid`; the additional `m` measurement is a
checkpoint-reconstructible post-hoc diagnostic, not a training-time gradient
measurement.

## Provenance

- Terminal source: `../artifacts/verification.json`
- Original terminal activations: `../artifacts/attempts/*/diagnostics/activation_statistics.json`
- Post-hoc `m`: `../artifacts/posthoc-m-activation-statistics.json`
- Numerical reduction: `../artifacts/figure_data_sitewise.json`
- Measurement/reduction source: `../06_plot_sitewise.py` and `../sitewise_figure.py`
- Final visual-QA layout: `../07_refine_sitewise_figure.py`
- Output: `../figures/02-h-vs-site-near-zero-grid.pdf`
