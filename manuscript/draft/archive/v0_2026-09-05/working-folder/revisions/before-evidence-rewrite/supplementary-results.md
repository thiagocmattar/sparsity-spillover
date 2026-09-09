# Supplementary result inventory

This file is the prose-facing index for the v0 paper bundle. Numerical claims
in `main.tex` should be traceable through the generated Analysis 012 outputs,
not recopied from memory.

## Decision on Pythia-410M

Retain the 410M cohort in the main paper, with its own boundary-result
subsection. It is not a clean third point in a scaling trend:

- all scales received 1,493,172,224 scheduled input tokens, corresponding to
  106.142, 21.202, and 3.684 tokens per parameter;
- the 410M A0 validation loss (4.547456) is worse than the 70M A0 loss
  (4.099766), and its gradient-clipping incidence is 56/712 versus 8/712;
- increasing trained `kappa` from 0 to 0.5 lowers 410M loss by 0.501189 for
  A4-OL1 and 0.305604 for A7-OL1, opposite the 14M and 70M directions;
- post-hoc clipping still raises 410M loss monotonically;
- Run 021 rejects `6e-4` and `1e-3` as upward learning-rate repairs under the
  same one-pass budget, but does not test a longer horizon.

Allowed interpretation: fixed-token regime dependence with unresolved
optimization horizon. Disallowed interpretation: proven undertraining,
compute-optimal scaling, or quality improvement over the dense control.

## Main trained endpoints

Each cell is paired eager validation loss / measured `R_model` percentage.

| Scale | A4 k=0 | A4 k=.5 | A7 k=0 | A7 k=.5 |
|---|---:|---:|---:|---:|
| 14M | 5.458276 / 7.8100 | 6.037982 / 12.7134 | 5.480181 / 7.0542 | 5.829407 / 27.4827 |
| 70M | 4.805361 / 25.6725 | 5.389480 / 35.5962 | 4.941206 / 23.5624 | 5.215925 / 40.6019 |
| 410M | 5.692075 / 38.3387 | 5.190966 / 71.5914 | 5.426296 / 41.1215 | 5.120692 / 80.6155 |

The complete kappa ladder and all 60 post-hoc points are in
`../../analyses/011-2026-09-03-pythia14m-70m-410m-selected-ladder/figure_data.json`.
The paper synthesis and source hashes are in
`../../analyses/012-2026-09-04-paper-synthesis/figure_data.json`.

## Microstructure results

- At 410M and `kappa=.5`, A4 has higher exact-zero mass than A7 at `a`
  (94.0% versus 90.9%) and `m` (92.0% versus 86.3%), yet lower `R_model`
  (71.6% versus 80.6%).
- A7's QK plus PV contributions are 16.7733, 10.4238, and 11.0639 percentage
  points at 14M, 70M, and 410M. These terms are absent from A4 except for
  negligible natural zeros.
- High-dose A7 `(q_post,k_post,v)` exact-zero masses are
  `(93.545,94.541,98.713)%`, `(71.531,72.123,86.939)%`, and
  `(77.806,80.611,93.960)%` across the three scales. They are not monotone in
  parameter count.
- In the separate 14M naive-L1 diagnostic, `h` near-zero mass rises from
  63.5085% to 92.3512%, while `v` rises from 0.1327% to 0.1853%; `q`, `k`, and
  `m` stay tiny and non-monotone. Post-`W_o` near-zero mass peaks at 0.8551%
  before falling to 0.7248% at the largest dose.

## OL1 diagnostic

At matched 14M A1-H endpoints, OL1 minus naive-L1 validation loss is
`-0.008087`, `-0.006083`, `-0.002453`, and `+0.018753` at pressure weights
0.05, 0.1, 0.5, and 1.0. The geometric projection works as implemented, but
the endpoint evidence does not establish general superiority.

## Coverage and units

Every main endpoint covers all 500 validation documents, 338 complete
2,048-token sequences, and 692,224 input tokens. The 1,444-token tail is
excluded. Fractions are formed only after pooling integer counts. `R_model` is
a percentage of zero-containing scalar products in the declared eager,
full-sequence graph with a dense LM-head denominator. It is not removed FLOPs,
kernel utilization, or speedup.

## Bundle inventory

- `main.tex`, `sections/`, and `references.bib`: paper source.
- `main.pdf`: rendered anonymous v0 paper.
- `figures/*.pdf`: vector paper figures copied from their owning run/analysis.
- `tables/*.tex`: generated count-reconciled tables.
- `build-input-hashes.json`: exact hashes of copied figure/table inputs.
- `framing-review.md`: adversarial review and repairs.
- `claim-ledger.md`: claim/evidence/nonclaim crosswalk.
- `handoff.md`: clean implementation-repository specification.

