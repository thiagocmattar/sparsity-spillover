# O004 - L1N spillover at epsilon 1e-2

## Question

How does the Figure 01 relationship change when near-zero mass uses `abs(x) <= 1e-2`?

## Method and coverage

All coordinates are recalculated from previously stored integer hits and denominators at `abs(x) <= 1e-2`; no checkpoint rerun was required. Counts are pooled across all validation batches and six layers before division. Coverage is all 500 MiniPile validation documents, 338 complete 2,048-token blocks, 692,224 input tokens, and the declared 1,444-token excluded tail. `h`, `q_post`, `k_post`, and `v` come from the terminal diagnostic; `m` and the post-`W_o` attention output come from their verified full-validation post-hoc diagnostics. There is one seed and no condition averaging.

## Values

| Activation | Pressure label | h (%) | Attention mean (%) |
| --- | --- | ---: | ---: |
| GeLU | control | 3.579157 | 0.850081 |
| GeLU | lambda=0.1 | 6.787376 | 0.973581 |
| GeLU | lambda=0.5 | 25.309118 | 1.099355 |
| GeLU | lambda=1 | 41.200635 | 1.160884 |
| GeLU | lambda=5 | 89.690732 | 0.966890 |
| ReLU | control | 70.591584 | 0.796514 |
| ReLU | lambda=0.1 | 86.675570 | 1.029030 |
| ReLU | lambda=0.5 | 93.912799 | 1.195580 |
| ReLU | lambda=1 | 96.330078 | 1.250093 |
| ReLU | lambda=5 | 99.474717 | 0.956160 |

## Figure caption and encoding

**Figure 04. Epsilon-1e-2 version of Figure 01.** X is count-pooled `h` near-zero mass; Y is the unweighted mean of separately count-pooled `q_post`, `k_post`, and `v` fractions. Blue circles/solid lines denote GeLU and orange squares/dashed lines denote ReLU. Labels identify control and naive-L1 weight. Axes begin at zero.

## Interpretation and limits

These are threshold-dependent descriptive trajectories from one seed and a short training horizon. GeLU-versus-ReLU differences include the activation operator change. Near-zero activation mass is not an exact-zero product count, removable compute, measured speedup, a causal route, or a long-horizon optimum.

## Provenance

- Numerical reduction: `../artifacts/figure_data_attention_mean_eps1e-2.json`
- Source script: `../10_plot_eps1e-2_figures.py` and `../eps1e2_figures.py`
- Output: `../figures/04-h-vs-attention-near-zero-eps1e-2.pdf`
