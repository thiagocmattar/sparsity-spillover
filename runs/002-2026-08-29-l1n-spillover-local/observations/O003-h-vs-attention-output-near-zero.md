# O003 - L1N pressure and attention-output near-zero mass

## Question

How does near-zero mass at the attention-branch output immediately after the
output projection `W_o` and before residual addition vary with near-zero mass at
pressured `h`, and how do those trajectories differ for GeLU and ReLU?

## Method and coverage

The figure uses the ten matched final Run 002 checkpoints. Every coordinate is
measured over all 500 MiniPile validation documents, 338 complete 2,048-token
blocks, 692,224 input tokens, all six layers, and the declared 1,444-token
excluded tail. The threshold is `abs(x) <= 1e-3`. Integer hits and denominators
are pooled across validation batches and layers before division; there is no
condition or seed averaging.

The x-coordinate comes from the original terminal `h` diagnostic. The output
metric was not selected before launch, so it was measured post hoc from all ten
retained, hash-verified final checkpoints. Hooks capture both the output of each
`attention.dense` (`W_o`) module and the corresponding
`post_attention_dropout` output immediately before the parallel residual sum.
Attention dropout is zero in every layer, and all captured tensors were exactly
equal. Every post-hoc pass reproduced its stored final validation loss within
`1e-5`; six layer rows reconcile to each pooled output row.

## Values

| Activation | Pressure label | h near-zero (%) | Attention output near-zero (%) |
| --- | --- | ---: | ---: |
| GeLU | control | 0.462031 | 0.140630 |
| GeLU | lambda=0.1 | 0.866986 | 0.245113 |
| GeLU | lambda=0.5 | 3.293959 | 0.328246 |
| GeLU | lambda=1 | 4.680761 | 0.429791 |
| GeLU | lambda=5 | 18.525665 | 0.251973 |
| ReLU | control | 70.290451 | 0.114821 |
| ReLU | lambda=0.1 | 86.452467 | 0.239269 |
| ReLU | lambda=0.5 | 93.777274 | 0.386593 |
| ReLU | lambda=1 | 96.236050 | 0.462373 |
| ReLU | lambda=5 | 99.456826 | 0.208396 |

## Figure caption and encoding

**Figure 03. Near-zero mass after attention output projection versus pressured
FFN `h` for local Pythia-14M L1N pretraining.** Blue circles/solid lines denote
GeLU; orange squares/dashed lines denote ReLU. Labels identify the no-pressure
control and naive-L1 weight. Lines connect control through increasing pressure
strength as visual guides, not fitted relationships. Both axes begin at zero
and report percent at epsilon `1e-3`.

## Interpretation and limits

This is a checkpoint-reconstructible post-hoc activation diagnostic, not a
training-time gradient measurement. The trajectories are descriptive results
from one seed and a short training horizon. GeLU-versus-ReLU differences include
the activation operator change. Near-zero activation mass is not an exact-zero
product count, removable compute, measured speedup, a causal route, or a
long-horizon optimum. The source attempt evidence is `valid`.

## Provenance

- Terminal source: `../artifacts/verification.json`
- Original `h`: `../artifacts/attempts/*/diagnostics/activation_statistics.json`
- Post-hoc attention output: `../artifacts/posthoc-attention-output-activation-statistics.json`
- Numerical reduction: `../artifacts/figure_data_attention_output.json`
- Measurement/reduction source: `../08_plot_attention_output.py` and `../attention_output_figure.py`
- Final visual-QA layout: `../09_refine_attention_output_figure.py`
- Output: `../figures/03-h-vs-attention-output-near-zero.pdf`
