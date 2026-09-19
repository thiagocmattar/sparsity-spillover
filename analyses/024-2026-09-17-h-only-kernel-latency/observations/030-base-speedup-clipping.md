# Figure 13: Base-model speedup with post-hoc clipping

**Subsequent placement:** the unchanged PDF is now manuscript Figure 4 on
page 6, after the 14M/70M quality overview (Figure 3). The agentic-development
discussion moves with it unchanged. See
[observation 031](031-quality-main-appendix.md); the original adoption
record below retains its earlier placement and numbering.

PDF: [13-14m-70m-sparsity-base-speedup.pdf](../figures/13-14m-70m-sparsity-base-speedup.pdf).
Source: [22_plot_base_speedup_clipping.py](../22_plot_base_speedup_clipping.py).
Exact coordinates, normalization references, source/output hashes and curves:
[14m-70m-sparsity-base-speedup.json](../data/14m-70m-sparsity-base-speedup.json).

**Manuscript adoption, 18 September 2026:** the unchanged PDF is now Figure 3
on page 5, directly after the post-hoc calibration paragraph in Experimental
Study. The accompanying text introduces the configured GPT-6 Astra agent,
execution mechanisms, within-recipe linear trends and the limited clipping
gains, including the positive high-dose 70M exception. See the
[adoption and claim audit](../../../manuscript/draft/reviews/2026-09-18-base-speedup-adoption/README.md).
The creation notes below describe the preceding analysis-only task.

## Question and method

How does full-model speedup relative to each size's base model vary with
model-wide sparsity, including the measured post-hoc control paths?
The plot uses the normalization from [Appendix A3](020-dense-speedup-caption.md)
and the matched recipes and style of [Figure 08](025-matched-quality-latency.md).
The two panels are 14M and 70M. Both axes are linear, with independent panel
ranges that include every point. Circular markers, colors, line types, fonts
and the shared six-entry legend follow Figure 08; dotted paths have one subtle
Post-hoc annotation per panel. No point labels or fitted trend are added.

For every setting, speedup is the unrounded optimized A0 latency divided by
the optimized setting latency at the same model size. The numerators are
0.651573035007824 ms for 14M (Run029 K050) and 3.313247016348893 ms for 70M
(Run035 k050-70m-v2). These exact A3 references are also used for the Run036
clipping paths; the paths are not separately normalized or shifted to place
their zero-clipping measurements on historical control points. Recurring
clipping work is included in the measured latency.

## Coverage

There are 22 trained checkpoints per size: Base model, GeLU -> ReLU, and
T4/Ph, T4/Pall, T7/Ph, T7/Pall at kappa = 0, .01, .05, .1, .5. This is exactly
Figure 08's 44-checkpoint cohort. The ten 14M pressure-free multisite conditions
in the older A3 are not part of Figure 08's matched cohort.

All 40 measured control clipping settings are included, with p = 0, .1, ...,
.9 for each base/ReLU control at each size. The existing Figure 08 join verifies
checkpoint identity, clipping target, retained loss, integer counts, kernel
identity and timing qualification. These paths join all measured doses in
order; they are not a Pareto selection or a claim of attainable interpolation.

Timing uses RTX 5090, BF16, batch one, 2,048-token full-sequence inference and
all 50,304 logits. Each latency is a geometric mean over 64 inputs, seven
passes and three processes (1,344 samples). Qualification and integer-pooled
FP16 logical sparsity cover all 338 complete blocks from 500 validation
documents; the 1,444-token tail is excluded. The full-model logical denominator
includes the dense output head. One training seed/final checkpoint per recipe
is represented.

## Publication caption

**Model-wide sparsity and full-model speedup relative to the base model.**
Panels show Pythia-14M (a) and Pythia-70M (b), with 22 trained checkpoints and
20 control clipping evaluations each. Speedup is the optimized base-model
latency divided by the optimized setting latency at the same size; the base
model is therefore 1x. Both terms use K050 at 14M and its qualified shape-specific
port at 70M. Colors identify the six recipes; dashed h-only and solid all-site
pressure curves connect independently trained threshold settings. Dotted
post-hoc paths apply clipping at a, m, h and z to the fixed base and ReLU
checkpoints, retaining all ten measured targets and their own zero-dose
measurements. Timing includes recurring clipping costs. Model-wide sparsity
uses full-validation pooled logical counts, whereas speedup uses measured
full-model geometric-mean latency. The panel scales differ. Normalization
does not remove differences between timing sessions or model quality.

## Result and associated proposed manuscript writing

The initial suggested placement was the runtime subsection or its appendix,
alongside the quality/sparsity overview. The original plotting task did not
edit manuscript TeX; the subsequent author-requested adoption is recorded above.

> Relative to the optimized base model, the trained recipes reach up to
> 1.418x speedup at 14M and 2.047x at 70M. The measured clipping paths have a
> different execution trade-off: positive clipping doses remain below 1x at
> 14M, while sufficiently high doses reach up to 1.222x at 70M. These are
> comparisons among models or clipped evaluations with different quality;
> they should be read together with the validation-loss panels. A larger
> scalar sparsity value does not guarantee a larger speedup.

## Caveats and verification

Trained 14M times use Run029 except for T7/Ph in Run033; 70M trained times
use Run035. Clipping times use Run036 on another GPU/host session. Small
differences across sessions are descriptive. The 70M port is shape-specific,
and equal optimization effort across sizes is not established. Base-normalized
speedup is not each recipe's native-versus-optimized speedup and does not
isolate the causal contribution of skipping. No seed uncertainty is shown.
Clipping can incur substantial quality costs, retained in the source data
and Figure 08's quality analysis.

All 84 ratios were checked against their original latencies and common
size-specific references. The 44 trained ratios equal the existing A3 metric;
the trained and clipping memberships exactly match Figure 08. Source hashes,
pooled sparsity counts and output hashes were checked. Four existing Figure 08
tests pass, covering the clipping identity/count/timing join and retained
coordinates. Every existing analysis PDF remains byte-identical. The one-page
PDF was rendered at 2,000 pixels and visually checked; all fonts are embedded.
No experiment, benchmark, cloud work or manuscript modification was performed.

Reproduce from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/22_plot_base_speedup_clipping.py
```
