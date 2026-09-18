# Figure 8: Matched 14M/70M quality, sparsity and latency with measured clipping

PDF: [08-14m-70m-quality-sparsity-latency.pdf](../figures/08-14m-70m-quality-sparsity-latency.pdf).
Builder: [16_plot_matched_quality_latency.py](../16_plot_matched_quality_latency.py).
Exact plotted records, joins, limits and hashes:
[14m-70m-quality-sparsity-latency.json](../data/14m-70m-quality-sparsity-latency.json).

## Question and requested design

How do the matching trained recipes and control clipping paths compare in
quality-sparsity and sparsity-latency views at 14M and 70M? This is a new 2-by-2
extension of [Figure 6](../figures/06-14m-quality-sparsity-latency.pdf), with 14M
above 70M, quality on the left and full-model latency on the right. The title
is "Sparsity vs. Quality trade-off and Latency on Pythia 14M and 70M". Each panel
title begins with its model size. One shared six-entry legend follows Figure 6.

The shared recipe labels are Base model, GeLU -> ReLU, T4/Ph, T4/Pall, T7/Ph,
and T7/Pall. Control markers remain open gray and filled olive circles;
h-only pressure uses teal/purple dashed curves and all-site pressure uses
blue/orange solid curves. Dotted control paths receive one subtle Post-hoc
note per panel. Quality panels retain the T4 and T7 ceiling guides.
DejaVu Sans and 14/11.5/11/10/11-point title/panel/axis/tick/legend sizes follow
Figure 6, on a 10.8-by-6.8-inch canvas.

The final annotation polish shifts the quality-panel Post-hoc notes right:
panel (a) from 2% to 4% sparsity and panel (c) from 10.5% to 15.5%.
Their vertical positions and rotations, and all plotted data, are unchanged.

## Method and coverage

Both panels in each row contain the same 22 trained checkpoints: the two
controls and T4/T7 with h-only or all-site OL1 at κ=0,.01,.05,.1,.5.
There are 44 unique trained checkpoints across sizes. No pressure-free
multisite or historical local-pressure recipes are included. All 14M trained
coordinates exactly preserve Figure 6. Trained losses use the ordinary
reloaded final-checkpoint evaluation; logical sparsity uses integer-pooled
canonical FP16 counters. One seed and final checkpoint per setting are used.

All 40 control clipping settings from Run036 are joined to the retained
quality table by exact checkpoint identity and target p. The join also verifies
identical retained losses, integer counts, model size, control family, and
kernel name. Each of four control/size combinations has p=0,.1,...,.9.
Clipping zeroes abs(x)<=t at a,m,h,z in all six layers, after the trained
activation, using the original per-site/layer thresholds. ReLU is the trained
control topology; its post-hoc clipping still covers all four sites.

The latency values are the qualified candidate geometric means from
[Run036](../../../runs/036-2026-09-18-controls-clipping-final-kernel/results/clipping-final-kernel.json),
the same source as [Figure 7](../figures/07-controls-posthoc-final-latency.pdf).
They include recurring external PyTorch clipping in the timed graph; p=0
identity masks are elided. Each setting pools three fresh processes, 64 common
inputs and seven timing passes, or 1,344 synchronized host timings. The workload
is RTX 5090, BF16, batch 1, T=2,048, uncached full-sequence inference and all
50,304 logits. Every setting qualifies against its matching clipped reference.
The final kernels are K050 at 14M and the qualified k050-70m-v2 port at 70M.

All 40 latency settings are visible. Quality panels retain a focused view
around the trained recipes: 5.12-6.19 at 14M and 4.0-5.58 at 70M. Eight 14M
clipping records (both controls at p=.6-.9) and seven 70M records (Base at
p=.6-.9; ReLU at p=.7-.9) lie above those quality limits. Their exact IDs and
values are retained, and the paths are plotted through the view boundary.
The complete control clipping loss range is visible in Figure 7's right column
and [Appendix A1](../figures/A1-complete-quality-sparsity.pdf). No trained points
are outside the displayed limits. Sparsity scales match within each row;
quality and latency scales are independent between sizes.

All canonical quality/count evaluations and numerical qualification use the
338 complete 2,048-token blocks from 500 MiniPile validation documents, with
692,224 input tokens and a 1,444-token excluded tail. Retained FP16 clipping
quality/counts remain distinct from the BF16 runtime evaluation; no loss or
latency has been shifted to align its zero-dose point with another session.

## Publication caption

**Sparsity vs. quality trade-off and latency on Pythia 14M and 70M.**
Rows show 14M (a,b) and 70M (c,d); columns show validation loss and full-model
latency versus model-wide logical sparsity. Each row uses the same 22 trained
checkpoints and the shared recipe legend. Pressure curves connect separately
trained κ=0,.01,.05,.1,.5 settings, not training trajectories. Dotted Post-hoc
curves clip the fixed Base and ReLU control checkpoints at a,m,h,z, with
p=0,.1,...,.9. All 40 measured clipping latencies are shown and include the
recurring clipping operators. Quality views focus on the trained-model range;
eight 14M and seven 70M high-loss clipping points continue above the view,
with full curves in the control-clipping figure. Vertical quality-panel guides
denote topology-specific analytic ceilings, not observed sparsity or speedup.
Validation covers all 338 complete blocks from 500 documents, excluding the
1,444-token tail. Latencies are geometric means of 1,344 timings per setting
on RTX 5090, BF16, batch 1, 2,048-token full-sequence inference and the full
50,304-way logit head. Clipping quality/counts are the retained FP16 measurements.
The 14M kernel is K050; 70M uses its qualified shape-specific port. Trained
timings use Run029/Run033 at 14M and Run035 at 70M; clipping uses Run036 on a
different GPU/host session. Small cross-session differences do not establish
reliable winners. Each training condition has one seed and one final checkpoint.

## Result and associated proposed manuscript writing

Proposed placement: extension of the introduction overview, or the paired
[quality and execution discussion](../../../manuscript/draft/training-results.tex)
and [kernel section](../../../manuscript/draft/kernel-autoresearch.tex).
This new figure is not adopted into manuscript TeX by this task.

> The same recipe comparisons yield different quality and latency orderings.
> At κ=.5, T7/Pall has greater logical sparsity than T7/Ph at both sizes:
> 27.483% versus 16.664% at 14M and 40.602% versus 32.233% at 70M. The two
> 14M latencies are both approximately 0.474 ms across different timing
> sessions. At 70M, the all-site setting has lower loss (5.216 versus 5.337)
> but higher latency (1.748 versus 1.618 ms). Control clipping follows a
> different path: every positive clipping setting is slower than its own
> zero-dose control at 14M, whereas the largest 70M clipping targets reduce
> final-kernel latency at substantial quality cost. Logical sparsity alone
> therefore does not determine the quality-latency ordering.

## Caveats and verification

Clipping uses a separate timing session and externally applied masks, so it
is not a matched same-session isolation of trained gates versus clipping.
The zero-clipping endpoints retain their own measured timings rather than
being attached to the historical control marker. High-p clipping latency
improvements are not necessarily improvements over the native graph; Run036
retains that separate comparison. The 70M port is shape-specific, not unchanged
14M machine code or an equal-budget optimization study. No kernel skipping
benefit is inferred from scalar sparsity. The curves imply neither interpolated
attainable models nor uncertainty over independent training seeds.

Four focused tests cover source/output hashes, preservation of Figure 6's
14M coordinates, matched row membership, the exact 40-point clipping join,
integer-count sparsity, latency geometric means, full validation coverage and
declared quality tails. All four pass, along with the seven existing paper
evidence checks. The one-page PDF was rendered at 2,000 pixels and visually
inspected; all three fonts are embedded. Figure 6, Figure 7, all other existing
PDFs and manuscript files are preserved. No new training or benchmarking ran.

Reproduce from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/024-2026-09-17-h-only-kernel-latency/16_plot_matched_quality_latency.py
.venv/Scripts/python.exe -X utf8 -m unittest discover -s analyses/024-2026-09-17-h-only-kernel-latency -p test_matched_quality_latency.py -v
```
