# O001 - R_model versus final validation loss for naive L1 and OL1

## Question

At matched lambda after one full MiniPile pass, where do Run 004's naive-L1
and Run 009's OL1 endpoints lie in final validation loss versus measured
`R_model`?

## Sources and coverage

- Run 004 `artifacts/verification.json` and each selected attempt's
  `metrics.json`, `diagnostics/activation_statistics.json`, and
  `diagnostics/logical_products.json`.
- Run 009 equivalents for its four OL1 attempts. Run 009 reuses Run 004's GeLU
  and ReLU controls, which are plotted once.
- One seed, Pythia-14M, topology `A1-H`, ReLU at `h`, pressure only at `h`, and
  lambda `{0.05, 0.1, 0.5, 1.0}`.
- Each endpoint follows 712 optimizer boundaries and 1,493,172,224 training
  input tokens. Final validation covers all 338 complete blocks and 692,224
  input tokens; the 1,444-token tail is excluded.

`01_compare.py` rejects unverified evidence or unmatched initialization,
schedule, condition, training, validation, topology, or lambda identities. It
recalculates activation and logical fractions from stored integer numerators
and denominators before writing `comparison.json`. `02_plot.py` reads only that
reduced artifact.

## Figure caption and legend

**Figure 1. Measured model-level logical opportunity versus final validation
loss after one full MiniPile pass.** Blue circles and a solid line show Run
004's ReLU `h`-only naive-L1 conditions; orange squares and a dashed line show
Run 009's matched OL1 conditions. Text beside every pressure endpoint gives its
lambda. The gray diamond and black triangle are the reused Run 004 GeLU and
ReLU controls, respectively. Horizontal position is measured exact-zero
`R_model` as a percentage; it is a logical-product opportunity, not speedup.
Vertical position is final loss from the reloaded checkpoint over all 338
complete validation blocks. Lower loss is better. The validation-loss axis uses
conventional ascending numerical order, is explicitly labeled as an endpoint
detail, and does not start at zero. Lines connect
lambda-ordered endpoints for readability and do not imply interpolation. There
are no uncertainty bars because only one seed was evaluated.

## Observed pattern

Across the four matched lambdas, naive L1 and OL1 have nearly coincident
`R_model` coordinates. OL1 minus naive-L1 `R_model` is `+0.003632`, `+0.002272`,
`-0.019581`, and `-0.010971` percentage points at lambda `0.05`, `0.1`, `0.5`,
and `1.0`. The corresponding validation-loss changes are `-0.008087`,
`-0.006083`, `-0.002453`, and `+0.018753`.

The lowest naive-L1 endpoint is `5.102276` loss at `3.949334%` `R_model`
(lambda `1.0`). The lowest OL1 endpoint is `5.110235` loss at `3.767998%`
`R_model` (lambda `0.5`). ReLU without pressure moves from the GeLU control's
negligible `R_model` to `2.714130%`, while worsening final loss from `5.208583`
to `5.269646` in these endpoints.

## Caveats and nonclaims

This is a descriptive single-seed, single-scale endpoint comparison. It does
not estimate realization variance, establish that the small paired loss
differences are reproducible, identify a causal gradient route, or measure
runtime acceleration. The runs used independently scheduled Pods and are not
bitwise matched. Exact arithmetic ordering is not promoted to a consolidated
finding or manuscript claim.

## Provenance

- Reduction: `../01_compare.py`
- Plot: `../02_plot.py`
- Reduced artifact: `../comparison.json`
- Generated tables: `../tables.md`
- Figure: `../figures/01-r-model-vs-final-validation-loss.pdf`
