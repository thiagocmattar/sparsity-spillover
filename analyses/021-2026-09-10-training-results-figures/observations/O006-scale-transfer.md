# Quality costs and architectural ceilings across model sizes

## Question and scope

Does the high-threshold seven-site OL1 recipe remain favorable across model
sizes when both quality cost and architectural ceiling are explicit?

The author requested this analysis-only redesign on 11 September 2026.
The manuscript and the previous two-row scale figure remain unchanged.

## Sources and coverage

- Immediate source: Analysis 018 [figure_data.json](../../018-2026-09-08-results-materials/figure_data.json).
  Its SHA-256 is recorded in the [reduction](../data/scale-transfer.json), together
  with all selected source paths, endpoint identities, losses and integer counts.
- Thirty trained endpoints: A4-OL1 and A7-OL1 at kappa = 0, 0.01, 0.05, 0.1,
  0.5 in each of Pythia-14M, 70M and 410M. Three untreated A0 evaluations supply
  the size-specific loss references. Thirty A0 clipping evaluations supply the
  three dotted curves (ten target fractions, p = 0, 0.1, ..., 0.9, per size).
- Trained sources: Runs 014/015 at 14M, Run 018 at 70M and Run 019 at 410M;
  14M A0 is retained from Run 004. Clipping sources are Analysis 006 at 14M
  and the retained `teal_frontiers.json` files in Runs 018/019.
- Each evaluation covers all 500 validation documents packed into 338 complete
  2,048-token sequences: 692,224 input tokens and 691,886 prediction tokens.
  The excluded tail is 1,444 tokens. Counts pool sequences, layers and operation
  families before division; future-masked attention pairs are excluded.
- One seed, 1234; random initialization. Within each size, initialization hash,
  data-order schedule, seeds, validation cache and training-token budget match.
  The selected OL1 runs use lambda = b = 1. Each model trains for 712 optimizer
  steps and 1,493,172,224 input tokens.

## Reduction and display

Model-wide sparsity is `100 * block_zero_product_count / model_product_count`.
The denominator includes dense block matrix products and the final output
projection. The analytic ceilings use the same workload and denominator.

Every vertical coordinate is `loss - same_size_A0_loss`, calculated before
rounding. The A0 losses are 5.2085730123096665, 4.09976596627715 and
4.547456437666741. The last rounds directly to **4.547**, rather than the
double-rounded 4.548 in the proposed subtitle. The measured 14M clipping p=0
is retained with its small positive loss difference, 0.000021047493410364382;
it is not forced to the unmodified A0 origin.

All panels share the displayed loss range [-0.05, 1.35], with separate sparsity
ranges [0, 32], [0, 52] and [0, 90]. All thirty trained endpoints fit. Four
clipping evaluations per size lie above the displayed range (p = 0.6 to 0.9).
Their coordinates remain in the reduction and plotted paths; the axes clip
them at the display boundary. No values are dropped and reconnected.

Endpoint utilization always uses the **same-size A7 ceiling**:
`100 * pooled_zero_products / (338 * A7.reachable_product_count)`.
It does not normalize A4 by its own narrower ceiling. These annotations replace
the old second normalized row. Only boundary trained thresholds are labeled.
The companion table beneath the panels reports the seven-site kappa = 0.5
endpoints and their total loss cost relative to A0.

## Figure and proposed caption

[Publication PDF](../figures/05-scale-transfer.pdf)

**High-threshold seven-site OL1 retains a favorable quality-sparsity trade-off
across model sizes when architectural ceilings are explicit.** Panels show
Pythia-14M, 70M and 410M. Blue diamonds and orange triangles denote four-site
and seven-site OL1, respectively, at trained thresholds kappa = 0, 0.01, 0.05,
0.1, 0.5. The vertical axis is validation loss minus the untreated same-size A0
loss; all panels share its scale, while horizontal ranges differ for readability.
Vertical blue and orange guides mark the four-site and seven-site architectural
ceilings. Endpoint labels and the companion table report seven-site kappa = 0.5
sparsity and the fraction of the same-size seven-site ceiling used. At that
threshold, seven-site OL1 has both greater sparsity and lower loss than four-site
OL1 at all three sizes. Thin gray dotted curves with open circles show A0
post-hoc clipping at the four projection-input sites. Higher-loss clipping
evaluations continue outside the displayed range; complete trajectories are
retained in the supporting full-range figure. Lines connect evaluated settings,
not training trajectories or interpolated models. Comparisons are matched within
size for initialization, data order and training budget, with one seed and
complete validation. Sparsity counts zero-operand multiplication opportunities,
not measured runtime savings.

For eventual manuscript adoption, the full clipping trajectories are already
available in the draft appendix; this task adds no manuscript reference.
The retained analysis counterpart is Analysis 018
[full-range scale figure](../../018-2026-09-08-results-materials/figures/03-scale-transfer-and-ceilings.pdf).

## Results

| Size | Seven-site S_model at kappa = 0.5 | Seven-site ceiling used | Loss change vs. A0 |
| --- | ---: | ---: | ---: |
| 14M | 27.48% | 91.8% | +0.621 |
| 70M | 40.60% | 82.2% | +1.116 |
| 410M | 80.62% | 92.4% | +0.573 |

The corresponding A7-OL1 minus A4-OL1 high-threshold differences are
+14.769 pp / -0.2086 loss, +5.006 pp / -0.1736 loss and
+9.024 pp / -0.0703 loss. Thus the seven-site points lie **lower and farther
right** at kappa = 0.5. This ordering is not universal across thresholds:
at kappa = 0 it holds only at 410M. The 14M and 410M endpoints both use about
92% of the seven-site ceiling despite their very different raw sparsities.

## Caveats and nonclaims

- These are comparisons of complete trained recipes. The 70M/410M cohorts lack
  pressure-free A4/A7 controls, so this figure does not establish transfer of
  the isolated marginal OL1 effect. Site expansion also changes the pressure
  objective and its equal-tensor normalization.
- The table's +0.621 loss at 14M is relative to untreated A0. It is distinct from
  the +0.1265 loss incurred by adding pressure to A7 at the same threshold.
- Clipping acts at four projection inputs, whereas seven-site recipes also
  target q/k/v. Their different computational reach is visible in the guides.
- Ceilings and operation weights depend on architecture and sequence length.
  Common A7 normalization is not a size-independent quality or speedup metric.
- Equal training tokens are not equal tokens per parameter. The 410M cohort
  uses a 3e-4 peak learning rate, compared with 1e-3 at 14M/70M.
- Single-seed endpoints provide no replicated uncertainty estimates. Connecting
  lines only order evaluated thresholds and do not define fitted frontiers.

## Reproduction and verification

Source script: [05_scale_transfer.py](../05_scale_transfer.py).
The [companion table](../tables/scale-transfer-endpoints.md) is generated from
the same unrounded endpoint reduction as the embedded PDF table.

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/05_scale_transfer.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_scale_transfer.py -q
```

Three focused tests cover the approved endpoint values, complete validation,
within-size loss references, common A7 normalization and preservation of
off-scale clipping coordinates. The one-page 7.4-by-4.2-inch PDF was rendered
and visually checked; fonts are embedded and text remains inside the page.
