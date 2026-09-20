# Retained 14M structural diagnostics

## Question

What detailed evidence adds to the main sparsity/latency figure without mixing
instruction counts from one implementation with timings from another?

## Method and coverage

`02_figures.py::diagnostic_14m` selects the 42 retained 14M records from
Analysis024 `data/kernel-appendix.json`: 22 trained endpoints and 20 Base/ReLU
clipping settings. Coordinates and speedup references are unchanged. The updated
figure omits the historical 70M row; new 70M implementation comparisons belong
in the matched Run045 tables. This is a presentation change, not a new measurement.

The output PDF is `figures/20-kernel-structure-native-base-speedup.pdf`. Its
corresponding data JSON records the input hash and every selected point ID.

## Caption and result

**Scalar sparsity, instruction bypass and native-base speedup at 14M.** Left:
projection scalar sparsity versus projection MMA bypass. Middle/right:
projection/attention bypass versus full-model speedup. Every speedup uses the
same native Base reference (0.655442398 ms). Lines connect measured thresholds;
dotted paths are post-hoc clipping. Counts pool all 338 complete validation
sequences; latency uses 64 fixed sequences, seven passes and three processes.

Projection instruction bypass tracks much of the measured speedup, while
post-hoc scalar sparsity can remain poorly represented by bypass. Attention
bypass is nearly constant for the trained recipes except one aggressive setting;
this does not establish a general causal attention-speedup relation.

## Caveats and verification

MMA bypass includes padding and scalar substitution; it is not the fraction of
all arithmetic or memory traffic removed. Logical sparsity remains a separate
estimand. Historical sessions are retained, including the clipping session.
This 22-endpoint diagnostic subset is smaller than the 41-endpoint main figure.
The standalone PDF was rendered with Poppler and visually checked: three aligned
panels, readable shared legend, and no clipped labels.
