# O002 - Peak learning rate versus final validation loss

## Question

How did final validation loss vary across the four approved peak learning rates
after the same short local training budget?

## Sources and coverage

- Source: `../artifacts/verification.json`, which reconciles the four immutable
  attempt manifests and their terminal metrics.
- Conditions: peak learning rates `5e-4`, `1e-3`, `2e-3`, and `4e-3`.
- Matched inputs: model seed `0`, data-order seed `0`, identical initial
  parameter hash, and identical 449-update block schedule.
- Budget: 58,851,328 training input tokens per condition.
- Each plotted loss is from the reloaded final checkpoint evaluated over all
  500 MiniPile validation documents: 338 complete blocks, 692,224 input tokens,
  and the declared 1,444-token excluded tail.

## Numerical reduction

The plot reads the one `final_validation_loss` scalar for each condition from
the terminal verification summary and orders the four points by peak learning
rate. It does not average across conditions or seeds, and no uncertainty bars
are available because each learning rate has one seed.

## Figure caption and encoding

**Figure 01. Final validation loss by peak learning rate for the local
Pythia-14M calibration.** The x-axis is logarithmic base 2 so the factor-of-two
grid is evenly spaced. Circular markers are the four observed conditions; the
solid connecting line is only a guide across the ordered grid. Values above
the markers give the final validation losses. The y-axis starts at zero.

## Observed pattern

Final validation loss decreased monotonically from 6.257393 at `5e-4` to
5.418346 at `4e-3`, an absolute difference of 0.839047 across the tested grid.
The lowest observed value is therefore at the upper grid boundary.

## Uncertainty, nonclaims, and possible confounds

This is a one-seed, 449-update, half-reference-batch calibration. It does not
locate an interior optimum, establish that `4e-3` remains preferable at a
longer horizon or another batch size, or quantify seed uncertainty. Sequential
execution may retain order or thermal effects. Evidence remains provisional
because the detached attempt manifests have null Git fields, although exact
code, config, initialization, data, schedule, and checkpoint hashes remain
available.

## Provenance

- Source script: `../04_plot_peak_lr_vs_final_val_loss.py`
- Output: `../figures/01-peak-lr-vs-final-val-loss.pdf`
