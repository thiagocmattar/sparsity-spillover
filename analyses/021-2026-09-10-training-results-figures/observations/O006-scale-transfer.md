# Quality costs and architectural ceilings across model sizes

## Question and scope

Does the high-threshold seven-site OL1 recipe remain favorable across model
sizes when both quality cost and architectural ceiling are explicit?

The author requested this analysis-only redesign on 11 September 2026.
The manuscript and the previous two-row scale figure remain unchanged.
The main figure is frozen after the author's final label-spacing and
endpoint-annotation polish in this analysis.

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

Every vertical coordinate is the **absolute validation loss** from its paired
loss/count evaluation. The retained sidecar table still computes
`loss - same_size_A0_loss` before rounding; these differences are not plotted.
The A0 losses are 5.2085730123096665, 4.09976596627715 and 4.547456437666741.
The measured 14M clipping p=0 remains at its actual loss 5.208594059803077,
rather than being replaced by the separately evaluated untreated A0 loss.

All panels share the displayed loss range [4.0, 6.4], with separate sparsity
ranges [0, 32], [0, 52] and [0, 90]. All thirty trained endpoints fit. Four
clipping evaluations at 14M/410M lie above the displayed range (p = 0.6 to 0.9),
as do three at 70M (p = 0.7 to 0.9). Their coordinates remain in the reduction
and plotted paths; the axes clip
them at the display boundary. No values are dropped and reconnected.

Endpoint utilization always uses the **same-size A7 ceiling**:
`100 * pooled_zero_products / (338 * A7.reachable_product_count)`.
It does not normalize A4 by its own narrower ceiling. These annotations replace
the old second normalized row. Only boundary trained thresholds are labeled.
The figure has no embedded table, raw-sparsity endpoint annotation or repeated
ceiling text blocks. Colored percentages at the tops of the vertical guides
identify the ceilings. The separate Markdown table retains endpoint sparsity
and total loss cost relative to A0. Only the tightly clustered 14M kappa = 0
labels retain short leaders; other labels sit directly beside their markers.
Only the seven-site kappa = 0 label is kept in the 410M cluster. Both high-
threshold endpoints remain labeled, and ceiling-utilization text is placed
close to the corresponding orange endpoint. The 410M ceiling labels are
separated horizontally and the vertical guides are lighter than the data.
The subsequent author-requested annotation polish centers each utilization
label on two lines (percentage above "of ceiling"), with a borderless white
background on in-plot threshold and utilization text. At 70M, utilization
moves slightly right, the four-site kappa = 0.5 label moves up and left, and
the seven-site kappa = 0.5 label moves right. Data, axes and normalization
are unchanged.
The final position adjustment moves utilization up/right at 70M and down/right
at 14M/410M. At 410M, the seven-site kappa = 0.5 label moves upward, while the
four-site label sits below/right of its marker, above the utilization note.

## Figure and proposed caption

[Publication PDF](../figures/05-scale-transfer.pdf)

**High-threshold seven-site OL1 retains a favorable quality-sparsity trade-off
across model sizes when architectural ceilings are explicit.** Panels show
Pythia-14M, 70M and 410M. Blue diamonds and orange triangles denote four-site
and seven-site OL1, respectively, at trained thresholds kappa = 0, 0.01, 0.05,
0.1, 0.5. All panels share the absolute validation-loss scale, while horizontal
ranges differ for readability.
Vertical blue and orange guides mark the four-site and seven-site architectural
ceilings. Seven-site kappa = 0.5 annotations report the fraction of the same-size
seven-site ceiling used: 91.8%, 82.2% and 92.4%, respectively. At that
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
The separate [companion table](../tables/scale-transfer-endpoints.md) is generated
from the same unrounded endpoint reduction; it is not embedded in the PDF.

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/05_scale_transfer.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/test_scale_transfer.py -q
```

Three focused tests cover the approved endpoint values, complete validation,
absolute plotted losses, retained within-size loss differences, common A7
normalization and preservation of off-scale clipping coordinates.
The one-page 7.4-by-2.95-inch PDF was rendered
and visually checked; fonts are embedded and text remains inside the page.
