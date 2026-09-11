# O005 - Activation structure and operation-weighted sparsity

## Question and status

Can a recipe have more FFN zeros yet less model-wide sparsity because its
attention products remain effectively dense? The author requested this composite
figure on 11 September 2026. It combines existing Pythia-14M evidence at
kappa = 0.5, with no new measurements. **Analysis only; no manuscript edits.**

## Method, sources and coverage

- Signed distributions: the two `14M_A4-OL1_0.5.json.gz` and
  `14M_A7-OL1_0.5.json.gz` artifacts under
  [Run 031 histograms](../../../runs/031-2026-09-08-signed-activation-density/results/histograms).
  The source collector is
  [02_evaluate.py](../../../runs/031-2026-09-08-signed-activation-density/02_evaluate.py);
  [Analysis 018 O011](../../018-2026-09-08-results-materials/observations/O011-activation-density-v3.md)
  records the full threshold-grid interpretation.
- Operation counts and validation losses:
  [Analysis 018 figure_data.json](../../018-2026-09-08-results-materials/figure_data.json),
  with [O004](../../018-2026-09-08-results-materials/observations/O004-operation-accounting.md)
  recording the full three-size accounting. These use corrected A4-OL1 from
  Run 015 and A7-OL1 from Run 014, both step-712 checkpoints at kappa = 0.5.
- The source IDs, attempt paths, training/cache identities, and reference losses
  reconcile across the two sources. They share initialization and data-order
  seeds 1234, initial parameters, training schedule, optimizer settings, 712
  steps and 1,493,172,224 training input tokens. Both use lambda = b = 1, with
  pressure and gate sites determined by the complete four-/seven-site recipes.
- Each source pass covers all 500 MiniPile validation documents: 338 complete
  2,048-token sequences, 692,224 input tokens, 691,886 next-token predictions
  and a 1,444-token excluded tail. Distribution recapture and operation counting
  are separate full-validation passes of the same checkpoints. Recapture loss
  differs from retained loss by +0.00000330 / -0.00003169 for A4/A7, within the
  Run 031 tolerance. Bar totals and reported loss retain the original pass.

Pool integer counts across six layers and all validation sequences before
dividing. FFN pools h,m with 80:20 element weights (2,658,140,160 elements);
attention pools post-RoPE q,k and v equally (1,594,884,096 elements).
Exact zeros are a separate point mass. Ten native bins are summed per display
bin, nominal width 0.01, using the actual float32 edges. Density divides by all
elements and actual bin width; it is not renormalized to nonzeros or the view.
Both density axes use symlog with linear threshold 0.01 and the same limits.
The FFN view [0,3] excludes at most 0.003745% of elements; attention [-4,4]
excludes at most 0.239626%. Full native [-8,8] tails remain in the source files.

Each stacked contribution is 100 times its pooled zero-product count divided
by the same 6,363,055,915,008 model products. The denominator includes all six
block operation families and the dense final output projection; future-masked
attention pairs are excluded. QK/PV use measured product counts, not an
independence estimate from activation marginals. The six segments sum to
model-wide sparsity. Projection segments sit below the adjacent QK/PV segments.

## Figure and caption

[Figure PDF](../figures/04-14m-distributions-and-operations.pdf).
Two conceptual panels: two aligned density axes on the left, two stacked bars
on the right. Blue/orange curves and exact-zero labels identify 4-site + OL1
and 7-site + OL1 (A4-OL1/A7-OL1 in the source records).
There are no density fills or baseline curve. Bar colors identify operations,
retaining the earlier accounting palette; QK/PV are the two warm segments at
the top. The polished titles read "Local sparsity at kappa = 0.5 (14M)" and
"Operation-weighted sparsity accounting." A short takeaway states that more FFN
zeros does not imply higher model-wide sparsity. Bar totals explicitly say
"total," and the note reads "QK + PV: 0.06 -> 16.77 pp."

The 11 September polish keeps the same figure dimensions and all data, enlarges
the density axes vertically by 17%, and specifies "exact zeros" in each label.
Only the bars have faint horizontal major-grid lines. The legend has three
columns and two rows: QKV / FFN up / FFN down, then QK / Attention output / PV.

**FFN sparsity alone can misrank recipes; operation accounting explains the
reversal.** Both panels compare 4-site + OL1 and 7-site + OL1 at kappa = 0.5
in Pythia-14M.
**(a)** Nonzero activation densities, with exact-zero masses labeled separately.
FFN pools h,m in an 80:20 element ratio; attention pools post-RoPE q,k and v
equally. Densities divide counts by all captured elements and bin width, without
renormalizing the nonzero mass or displayed range; both vertical axes are
symlog, linear below 0.01. The four-site recipe has more FFN zeros (99.31% versus 93.64%)
but far fewer attention zeros (0.22% versus 95.60%). **(b)** Stacked measured
zero-product contributions from six operation families, each divided by all
model products including the dense final output projection. QK and PV supply
0.06 pp under 4-site + OL1 and 16.77 pp under 7-site + OL1, explaining the reversal in
model-wide sparsity: 12.71% versus 27.48%. The seven-site recipe also has lower validation
loss, 5.8294 versus 6.0380. All quantities cover the full 338-block validation
set. These are complete-recipe comparisons and logical sparsity opportunities,
not isolated causal effects of attention pressure or measured speedups.

## Results

| Recipe | FFN zeros (%) | Attention zeros (%) | QK+PV contribution (pp) | Model-wide sparsity (%) | Validation loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| A4-OL1 | 99.305452 | 0.220322 | 0.056568 | 12.713449 | 6.037982 |
| A7-OL1 | 93.640379 | 95.598892 | 16.773285 | 27.482684 | 5.829407 |

| Operation | A4-OL1 contribution (pp) | A7-OL1 contribution (pp) |
| --- | ---: | ---: |
| QKV projection | 3.173188 | 2.429717 |
| FFN up | 4.139356 | 2.938696 |
| FFN down | 4.275167 | 4.272431 |
| Attention output | 1.069170 | 1.068555 |
| QK | 0.000001 | 8.324157 |
| PV | 0.056567 | 8.449129 |

The QK/PV contribution increases by 16.716717 pp, offsetting a 1.947482 pp
decrease across the four projection contributions. The net model-wide increase
is 14.769236 pp. Pooling only FFN activation zeros therefore orders these two
recipes differently from model-wide operation accounting.

## Interpretation limits and retained supporting figures

- One trained seed and one selected threshold. The comparison does not isolate
  pressure, gates, or the extra Q/K/V pressure targets from the rest of each
  recipe, including the change in pressure-objective normalization.
- Element-weighted activation mixtures and operation-weighted counts are
  different estimands. Neither activation percentages nor these aggregate
  product counts demonstrate profitable kernel skipping.
- Nonzero density omits the exact-zero mass, which is shown explicitly in the
  labels. Symlog display areas are not probabilities. No smoothing, imputed
  bins, confidence bands or independent-replicate interpretation is introduced.
- The existing
  [full threshold density grid](../../018-2026-09-08-results-materials/figures/05-v3-activation-density-grid.pdf)
  and [three-size operation figure](../../018-2026-09-08-results-materials/figures/04-operation-accounting.pdf)
  are retained unchanged. They can support a later appendix revision; no figures
  or text have yet been moved in the manuscript.

## Reproduction

Source script: [04_distributions_and_operations.py](../04_distributions_and_operations.py).
The [figure data](../data/14m-distributions-and-operations.json) retain source
hashes, checkpoint identities, display-bin integer counts, zero/tail masses,
coverage, full operation counters and unrounded contributions.

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/04_distributions_and_operations.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_distributions_and_operations.py -q
```

Three focused tests pass: conservation of zero/nonzero/tail probability mass,
operation contributions using the full model denominator, and the joined
two-checkpoint evidence with all plotted counts. The 7.1-by-3.35-inch PDF was
rendered with Poppler and visually checked. All numeric labels are present,
all text lies within page bounds, and fonts are embedded. The retained source
hashes and observation links resolve. The polish also verifies the revised
visible labels, grid placement and three-column legend order. The figure data
are identical to the preceding version, and the manuscript diff is empty.
Figure SHA-256: `13259bdf520a3540b81953b6088a72acec44ee2c68f534c0010cda8f55b53a33`.
