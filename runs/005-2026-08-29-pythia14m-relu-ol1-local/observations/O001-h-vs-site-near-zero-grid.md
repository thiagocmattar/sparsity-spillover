# O001 - Sitewise OL1 spillover trajectory

## Question

How do the individual `q_post`, `k_post`, `v`, and `m` near-zero responses
change as near-zero mass increases at the OL1-pressured ReLU FFN hidden
activation `h`?

## Method and coverage

The four panels use the five matched final Run 005 checkpoints: a ReLU control
and ReLU OL1 at `h` with `lambda` in `{0.01, 0.1, 0.5, 1.0}` and relative
trust budget 1.0. Every coordinate comes from the terminal activation
diagnostic over all 500 MiniPile validation documents, 338 complete
2,048-token blocks, 692,224 input tokens, all six layers, and the declared
1,444-token excluded tail. The threshold is `abs(x) <= 1e-3`.

For every site and condition, integer threshold hits and element counts are
pooled across validation batches and layers before division. Each plotted
pooled count was reconciled against the six layer rows and the corresponding
hash-verified transfer inventory. There is no batch, layer, condition, or seed
averaging.

## Values

| Pressure label | h (%) | q_post (%) | k_post (%) | v (%) | m (%) |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 74.252905 | 0.121371 | 0.136114 | 0.306624 | 0.073468 |
| lambda=0.01 | 76.982732 | 0.114055 | 0.134881 | 0.302895 | 0.075440 |
| lambda=0.1 | 84.380795 | 0.113054 | 0.139502 | 0.320898 | 0.079937 |
| lambda=0.5 | 90.668530 | 0.115207 | 0.127579 | 0.370035 | 0.086387 |
| lambda=1 | 93.624487 | 0.114117 | 0.122267 | 0.387541 | 0.072892 |

## Figure caption and encoding

**Figure 01. Sitewise near-zero mass versus pressured FFN `h` for local
Pythia-14M OL1 pretraining.** The 2x2 panels show `q_post`, `k_post`, `v`, and
`m`. Orange squares and a dashed line denote the single ReLU series. Point
labels are the no-pressure control (`ctrl`) or OL1 weight. Lines connect the
control through increasing pressure strength as visual guides, not fitted
relationships. All panels use the same zero-based x and y scales and report
percent at epsilon `1e-3`.

## Result and limits

Targeted `h` near-zero mass rises monotonically from 74.253% in the control to
93.624% at `lambda=1.0`. The clearest untargeted response is at `v`, which
rises from 0.307% to 0.388% overall and increases most between `lambda=0.1`
and `0.5`. The `q_post`, `k_post`, and `m` trajectories are small and
non-monotonic: `k_post` peaks at `lambda=0.1`, and `m` peaks at `lambda=0.5`
before falling below control at `lambda=1.0`.

These are descriptive trajectories from one seed, one model scale, and a
short local horizon. Near-zero activation mass is not an exact-zero product
count, removable compute, measured speedup, proof of a causal route, or a
long-horizon optimum. The evidence label is
`valid_with_provenance_limitation` because the detached attempt manifests have
null Git fields; exact input, diagnostic, code-content, schedule, checkpoint,
and artifact identities remain verified.

## Provenance

- Terminal source: `../artifacts/verification.json`
- Terminal activations: `../artifacts/attempts/*/diagnostics/activation_statistics.json`
- Numerical reduction: `../artifacts/figure_data_sitewise.json`
- Completion record: `../artifacts/sitewise_figure_completion.json`
- Source script: `../07_plot_sitewise.py`
- Output: `../figures/01-h-vs-site-near-zero-grid.pdf`
