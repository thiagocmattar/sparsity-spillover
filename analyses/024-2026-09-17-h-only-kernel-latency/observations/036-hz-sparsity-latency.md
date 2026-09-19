# Figure 15: h/z contribution to model-wide sparsity versus latency

PDF: [15-14m-hz-sparsity-latency.pdf](../figures/15-14m-hz-sparsity-latency.pdf).
Builder: [26_plot_hz_sparsity_latency.py](../26_plot_hz_sparsity_latency.py).
Exact coordinates, integer counts and source hashes:
[14m-hz-sparsity-latency.json](../data/14m-hz-sparsity-latency.json).

## Question and scope

How does full-model latency vary with the combined h/z contribution to
model-wide sparsity when other sites' contributions are omitted from the x-axis?
The user selected 14M only and the same trained recipes as manuscript Figure 1
(Analysis024 Figure 08). This is a new scatter plot of retained measurements;
no experiment, timing, kernel change or manuscript edit is involved.

## Method and coverage

Each of the 22 points represents one final trained checkpoint: Base model,
GeLU -> ReLU, and T4/Ph, T4/Pall, T7/Ph, T7/Pall at kappa = 0, .01, .05,
.1, .5. Colors match Figure 1. Circular markers are unconnected; the two
nearly coincident T7 high-threshold points overlap. Post-hoc clipping and
other recipe families are outside this requested matched cohort.

The horizontal coordinate is

`100 * (zero products in MLP W2 + zero products in attention output projection)
/ full-model product count`.

MLP W2 is fed by h; the attention output projection is fed by z. The common
denominator includes every measured block operation and the dense vocabulary
head, exactly as in S_model. Thus the unit is percentage points contributed
to S_model, not the sum or average of the two activation zero percentages,
and not sparsity renormalized over h/z alone. Contributions from a, m, QK
and PV are omitted only from the numerator; all operations still execute
in the measured model. Natural zeros remain included at unpressured sites.

Canonical FP16 logical counts pool all six layers and all 338 complete
2048-token blocks from 500 MiniPile validation documents: 692224 input tokens,
with a 1444-token excluded tail. Integer numerators are added before division.
All 22 original diagnostic files are hash-checked and their counts compared
with the retained paper export.

The vertical coordinate preserves Figure 1's final K050 full-model latency
in milliseconds per 2048-token sequence. Timings use RTX5090, BF16, batch one,
all 50304 vocabulary logits, 64 fixed inputs, seven passes and three fresh
processes (1344 timings per checkpoint), summarized by the geometric mean.
T7/Ph uses Run033; the other 17 settings use Run029.

## Caption

**h/z sparsity contribution and full-model latency on Pythia-14M.** Each
point is one trained recipe-threshold setting; colors identify the six
Figure 1 recipes. The x-axis sums the h-fed MLP-down and z-fed attention-output
zero-product counts and divides by the full-model product count, including
the dense output head. It therefore shows their combined contribution to
S_model in percentage points. The y-axis is the measured K050 full-model
latency per 2048-token sequence (RTX5090, BF16, batch one), averaged
geometrically over 1344 timings. Sparsity uses full-validation FP16 counts.
Axes are linear, and no trend or interpolation is fitted.

## Result and interpretation limits

The h/z contribution ranges from effectively zero at the base model to
5.34434 pp. Full-model latency ranges from 0.45948 to 0.65157 ms. Within
each pressure family, higher thresholds increase the h/z contribution and
generally reduce latency, with a small low-threshold reversal in T7/Pall.
At kappa=.5 the four pressure recipes have almost the same h/z contribution
(5.34099--5.34434 pp) while latency spans 0.45948--0.47389 ms. Their scalar
h/z contribution alone therefore does not determine latency.

This is a descriptive relationship among different trained checkpoints and
quality levels, not an isolated timing effect of h/z. It does not measure
tile occupancy or instruction bypass. FP16 logical counts and BF16 runtime
measurements remain distinct. Timings span two sessions, so small differences
are not evidence of reliable winners. Each setting uses one training seed;
no seed uncertainty, confidence interval, regression or causal claim is added.
No manuscript adoption or finding promotion is made.

## Verification and reproduction

- Exact membership and all 22 latencies match Figure 1's retained export.
- Original pooled h/z counts, full denominators, coverage and source hashes
  match for every checkpoint. Latencies also match the geometric means of
  the three retained process means to relative tolerance 1e-12.
- Every plotted coordinate lies inside the chosen axis limits; no point is
  shifted or jittered. All existing analysis PDFs remain byte-identical.
- The one-page PDF was rendered at 1700 pixels and visually inspected for
  label/legend clearance and clipping. All fonts are embedded.

Reproduce from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/26_plot_hz_sparsity_latency.py
```
