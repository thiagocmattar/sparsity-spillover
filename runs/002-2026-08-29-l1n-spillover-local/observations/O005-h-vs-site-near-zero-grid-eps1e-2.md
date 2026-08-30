# O005 - Sitewise L1N trajectories at epsilon 1e-2

## Question

How does the Figure 02 sitewise view change when near-zero mass uses `abs(x) <= 1e-2`?

## Method and coverage

All coordinates are recalculated from previously stored integer hits and denominators at `abs(x) <= 1e-2`; no checkpoint rerun was required. Counts are pooled across all validation batches and six layers before division. Coverage is all 500 MiniPile validation documents, 338 complete 2,048-token blocks, 692,224 input tokens, and the declared 1,444-token excluded tail. `h`, `q_post`, `k_post`, and `v` come from the terminal diagnostic; `m` and the post-`W_o` attention output come from their verified full-validation post-hoc diagnostics. There is one seed and no condition averaging.

## Values

| Activation | Pressure label | h (%) | q_post (%) | k_post (%) | v (%) | m (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| GeLU | control | 3.579157 | 0.723227 | 0.806253 | 1.020763 | 0.825338 |
| GeLU | lambda=0.1 | 6.787376 | 0.745665 | 0.816006 | 1.359073 | 0.862748 |
| GeLU | lambda=0.5 | 25.309118 | 0.817788 | 0.868815 | 1.611463 | 0.990629 |
| GeLU | lambda=1 | 41.200635 | 0.833393 | 0.814187 | 1.835071 | 1.085434 |
| GeLU | lambda=5 | 89.690732 | 0.785367 | 0.741738 | 1.373565 | 1.559306 |
| ReLU | control | 70.591584 | 0.650777 | 0.790119 | 0.948647 | 0.878814 |
| ReLU | lambda=0.1 | 86.675570 | 0.763833 | 0.867858 | 1.455398 | 0.883395 |
| ReLU | lambda=0.5 | 93.912799 | 0.879968 | 0.895363 | 1.811411 | 0.868319 |
| ReLU | lambda=1 | 96.330078 | 0.825804 | 0.888991 | 2.035484 | 0.875779 |
| ReLU | lambda=5 | 99.474717 | 0.777364 | 0.789138 | 1.301978 | 0.875558 |

## Figure caption and encoding

**Figure 05. Epsilon-1e-2 version of Figure 02.** The 2x2 panels show count-pooled `q_post`, `k_post`, `v`, and `m` near-zero mass versus count-pooled `h`. Blue circles/solid lines denote GeLU and orange squares/dashed lines denote ReLU. Point labels identify control or naive-L1 weight. All panels use the same zero-based x and y scales.

## Interpretation and limits

These are threshold-dependent descriptive trajectories from one seed and a short training horizon. GeLU-versus-ReLU differences include the activation operator change. Near-zero activation mass is not an exact-zero product count, removable compute, measured speedup, a causal route, or a long-horizon optimum.

## Provenance

- Numerical reduction: `../artifacts/figure_data_sitewise_eps1e-2.json`
- Source script: `../10_plot_eps1e-2_figures.py` and `../eps1e2_figures.py`
- Output: `../figures/05-h-vs-site-near-zero-grid-eps1e-2.pdf`
