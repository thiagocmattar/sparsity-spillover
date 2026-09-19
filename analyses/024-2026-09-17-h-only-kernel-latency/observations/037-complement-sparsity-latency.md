# Figure 16: Sparsity contributions outside h/z versus latency

PDF: [16-14m-complement-sparsity-latency.pdf](../figures/16-14m-complement-sparsity-latency.pdf).
Builder: [27_plot_complement_sparsity_latency.py](../27_plot_complement_sparsity_latency.py).
Exact coordinates, integer counts and source hashes:
[14m-complement-sparsity-latency.json](../data/14m-complement-sparsity-latency.json).
Companion: [Figure 15, h/z contributions](036-hz-sparsity-latency.md).

## Question and method

At the user's request, replace the h/z contribution on the horizontal axis
with the complementary set of sites, in a separate figure. The 20 trained
14M T/P settings, measured latencies, colors, marker styles and connecting
lines are identical to Figure 15. The original figure is preserved.

The new horizontal coordinate is

`100 * (QKV + MLP W1 + QK + PV zero-product counts) / full-model product count`.

These operations correspond to a, m, q/k jointly, and v. QK is counted once,
including zero-operand products from either q or k, without double-counting
their intersection. The denominator remains the full-model product count,
including the dense vocabulary head. Only the numerator changes; the plotted
value is neither 100% minus h/z sparsity nor a renormalization over these sites.
Its unit is percentage points contributed to S_model. The exact integer
identity is checked for every checkpoint: complementary zero-product counts
plus h/z zero-product counts equal all block zero-product counts. Consequently,
the x-coordinates of Figures 15 and 16 sum to S_model for each setting.

The builder verifies all 20 original logical diagnostic files against source
hashes and the paper export, and retains the latencies from Figure 15 exactly.
This is a plot of existing measurements, with no new timing or model changes.

## Coverage and style

The four recipes are T4/Ph, T4/Pall, T7/Ph and T7/Pall, each with
kappa = 0, .01, .05, .1, .5. Colors follow Figure 1; dashed Ph and solid Pall
lines connect increasing threshold settings within each recipe. Base model,
GeLU -> ReLU and post-hoc clipping settings are omitted.

Canonical FP16 logical counts pool all six layers and 338 complete 2048-token
blocks from 500 MiniPile validation documents, totaling 692224 input tokens
and excluding a 1444-token tail. Counts are added before division.
Full-model K050 latency uses RTX5090, BF16, batch one, 2048-token sequences
and all 50304 vocabulary logits. The geometric mean pools 64 inputs, seven
passes and three fresh processes: 1344 timings per setting. T7/Ph uses Run033;
the other 15 points use Run029. These are the same qualified checkpoints and
latencies as the corresponding points in manuscript Figure 1.

The y-axis retains Figure 15's 0.45--0.65 ms range; the linear x-axis spans
2.8--23.0 pp and includes every point. Typography, canvas and two-column
legend follow Figure 15. No regression, uncertainty band or annotation is added.

## Caption

**Sparsity outside h/z and full-model latency on Pythia-14M.** Each point
represents one T/P recipe-threshold setting. The x-axis sums zero-product
contributions from a, m, q/k and v (QKV, MLP-up, QK and PV) using the full-model
denominator, including the dense output head. QK is counted once. These
contributions complement the h/z contributions in Figure 15, summing to
S_model at every setting. The y-axis is K050 full-model latency per 2048-token
sequence on RTX5090 (BF16, batch one), averaged geometrically over 1344 timings.
Sparsity uses full-validation FP16 counts. Dashed Ph and solid Pall lines
connect increasing thresholds within each recipe as visual guides.

## Result and caveats

The complementary contribution ranges from 3.67625 to 22.14170 pp. At
kappa=.5, T7/Pall contributes 22.14170 pp outside h/z, versus 11.31950 pp for
T7/Ph, yet their full-model latencies are approximately equal (0.47337 and
0.47389 ms, respectively, in different sessions). T4/Pall and T4/Ph have
smaller complementary contributions (7.36911 and 4.88333 pp) and lower
latencies (0.45948 and 0.46225 ms). More sparsity outside h/z therefore does
not consistently order the observed full-model latencies.

This relationship does not isolate the runtime effect of the complementary
sites: h/z sparsity, activation structure and model quality also vary across
these checkpoints. It is not instruction bypass, removed FLOPs, or a causal
latency decomposition. Cross-session differences and one training seed per
setting limit small-effect interpretation. Lines are not fitted trends or
claims of attainable intermediate models. No manuscript or finding is changed.

## Verification and reproduction

All 20 source hashes/counts and complement identities pass; checkpoint
membership and latency values match Figure 15 exactly. Every point lies
inside the plotted bounds. All pre-existing analysis PDFs, including Figure
15, remain byte-identical. The new PDF was rendered at 1700 pixels and visually
checked; labels and legend fit cleanly, and all fonts are embedded.

```powershell
.venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/27_plot_complement_sparsity_latency.py
```
