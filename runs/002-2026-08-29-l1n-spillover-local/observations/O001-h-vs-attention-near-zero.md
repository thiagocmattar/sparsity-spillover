# O001 - L1N pressure at h versus attention near-zero mass

## Question

As naive L1 pressure at `h` increases, does count-pooled near-zero mass move
toward `h` while the untargeted post-RoPE query, key, and value sites lose
near-zero mass, and does the trajectory differ between GeLU and ReLU?

## Method and coverage

The figure uses ten matched randomly initialized Pythia-14M conditions: one
control and four `l1_naive` weights per activation. Every point is measured at
the reloaded final checkpoint over all 500 MiniPile validation documents, 338
complete 2,048-token blocks, 692,224 input tokens, and the declared 1,444-token
excluded tail. Model/data seeds are both 0 and all conditions share the same
initial-parameter and training-schedule hashes. The near-zero threshold is
`abs(x) <= 1e-3`.

X is formed by pooling integer `h` hits and denominators across validation
batches and all six layers, then dividing once. For Y, each of `q_post`,
`k_post`, and `v` is pooled the same way and the three resulting fractions are
averaged without weights. Their denominators are equal, so the result also
equals a joint count pool across those sites. No condition or seed averaging is
performed.

## Values

| Activation | Pressure label | h near-zero (%) | Attention mean near-zero (%) | Final validation loss |
| --- | --- | ---: | ---: | ---: |
| GeLU | control | 0.462031 | 0.088561 | 5.414689 |
| GeLU | lambda=0.1 | 0.866986 | 0.101251 | 5.269140 |
| GeLU | lambda=0.5 | 3.293959 | 0.109127 | 5.228334 |
| GeLU | lambda=1 | 4.680761 | 0.116303 | 5.243827 |
| GeLU | lambda=5 | 18.525665 | 0.096335 | 5.615236 |
| ReLU | control | 70.290451 | 0.078162 | 5.514373 |
| ReLU | lambda=0.1 | 86.452467 | 0.100498 | 5.238756 |
| ReLU | lambda=0.5 | 93.777274 | 0.117766 | 5.132591 |
| ReLU | lambda=1 | 96.236050 | 0.124446 | 5.198799 |
| ReLU | lambda=5 | 99.456826 | 0.096503 | 5.609286 |

## Figure caption and encoding

**Figure 01. Near-zero mass at pressured FFN `h` versus mean near-zero mass at
untargeted attention sites for local Pythia-14M pretraining.** Blue circles and
a solid line denote GeLU; orange squares and a dashed line denote ReLU. Labels
identify the no-pressure control and naive-L1 weight. Lines connect conditions
in ascending pressure order and are visual guides, not fitted relationships.
Both axes begin at zero and report percent at epsilon `1e-3`.

## Interpretation and limits

The plotted geometry is descriptive. A down-right within-activation trajectory
is the prespecified simple spillover signature; the observed pattern must be
interpreted together with final validation loss and whether `h` responded to
pressure. This one-seed, two-hour local cohort does not establish a causal
route, seed uncertainty, a long-horizon optimum, removable logical products,
or measured speedup. GeLU-versus-ReLU differences intentionally include the
operator change. The attempts are labeled `valid`.

## Provenance

- Source: `../artifacts/verification.json`
- Numerical reduction: `../artifacts/figure_data.json`
- Source script: `../04_plot.py` and `../plotting.py`
- Output: `../figures/01-h-vs-attention-near-zero.pdf`
