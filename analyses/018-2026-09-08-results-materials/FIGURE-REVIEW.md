# Critical figure review — 8 September 2026

The first package carried too much information inside the artwork. Wide source
canvases hid the problem: scaling them to paper width made labels small and
made annotations compete with the data. The revised figures are all exactly
5.5 inches wide, with ordinary lettering at least 8 pt (math subscripts and
superscripts scale conventionally). Numerical detail belongs in captions and tables.

The [official ICLR template](https://github.com/ICLR/Master-Template/blob/master/iclr2026/iclr2026_conference.tex)
specifies a 5.5-inch text width and asks for clean, legible, reproducible artwork
that remains understandable in black and white. The 8-pt threshold is our
review criterion, not a claimed numeric ICLR rule. This review checks clarity
and formatting; it does not certify conference acceptance or final page fit.

| Figure | Critical assessment of the first version | Revision and decision |
| --- | --- | --- |
| 01: 14M overview | Two views, recipe lines, frontier lines, rings, clipping curves and a historical cohort competed in one graphic. Dose detail was unreadable after reduction. | One scatter plot; 30 included training conditions, eight consistent recipe symbols; no connecting lines or frontier overlay. Full numerical frontier remains a table. Clipping has its own figure. Keep in main text. |
| 02: matched effects | The aligned rows worked, but “child”, “parent” and “expand gates” obscured the scientific change. Twenty-five full tick labels crowded the axis. | Seven concrete action labels, 25 aligned points, one shared dose key; treatment minus reference throughout. Points progress left to right with dose within a block. No additive-chain interpretation. Keep in main text. |
| 03: scale transfer | Both rows centered loss on A0, hiding absolute quality. Missing clipped controls weakened the comparison. Per-point labels and ceiling lines added clutter. | Top row: complete absolute loss versus raw sparsity. Bottom: A0-relative loss versus ceiling utilization. Both rows include all ten clipping targets for A0/A1-H and five doses for A4-OL1/A7-OL1. One shared legend; ceiling curves move to Figure 08. Keep in main text. |
| 04: operation accounting | Stacks, ceiling markers, numeric stack labels and a second heatmap repeated the same evidence. | Six stacked bars at κ=.5; operation color and hatch key shared once. Full dose and per-operation rates remain in tables. Move to appendix unless needed to explain the activation case. |
| 05: activation mass | Twelve panels, four categorical magnitude bands, a symlog scale and per-panel RMS annotations required too much decoding. | Four panels: κ=0/.5 columns and exact-zero/small-nonzero rows, linear 0–100% scales. Five shared sites h,m,q,k,v, one recipe legend, no RMS text or connecting lines across sites. Keep as a focused case study. |
| 06: clipping coverage | Fifteen clipping trajectories plus all trained curves and frontier overlays duplicated Figure 01. | A single full-range scatter of all 150 post-hoc evaluations, grouped by five source families. Source dose and clipping target remain in the table. Appendix only. |
| 07: runtime | Copying the previous 35-checkpoint artwork preserved clutter and violated the requested cohort exclusion. | Qualified incumbent curve only on the left; final K050 scatter and descriptive fit for 30 included checkpoints on the right. Recompute every cohort-dependent statistic. Recipe symbols follow Figure 01; no giant equation, repeated legend or failed-timing cloud. Keep in systems section. |
| 08: ceiling versus size | Missing entirely; embedding ceiling markers inside other figures did not answer the structural question directly. | Dedicated four-topology plot at the three actual parameter counts, with a log size axis and a fixed full-sequence workload. OL1 variants share the same ceilings, so duplicate curves are unnecessary. Introduce before scale normalization. |

## Final visual decisions and remaining limits

- All eight figures have a single reading task; no narrative headline is embedded
  above the artwork. Captions supply coverage, dose interpretation and limitations.
- Recipe color and marker identity are consistent. Dose symbols in Figure 02 have
  a separate explicit key; operation hatches in Figure 04 preserve grayscale distinctions.
- Paper-width rendering caught cropped right-edge ticks in Figures 03/07 and an
  activation value above the first proposed Figure 05 axis range. Both were fixed.
- The dense low-threshold cluster in Figure 01 is real. It is not jittered or
  expanded in an inset; exact endpoint identification is available in the table.
- Full clipping losses compress the trained differences in Figure 03. This is the
  cost of the requested complete quality view; Figure 02 and the scale-pair table
  carry the fine differences. Do not reintroduce another zoomed panel.
- Figure 05 shows measured mass summaries, not a reconstructed density. A0 lacks
  a/z near-zero measurements. Post-Wo attention output is omitted because it is
  not the manuscript's pre-Wo z site.
- A single seed supplies no between-seed intervals. No error bars are invented.
  Connecting scale/threshold curves show evaluated order, not a fitted scaling law.

The main text should use Figures 01, 02, 08 and 03 as the central sequence;
include 05 if the mechanism discussion has space, followed by the systems
figure. Figures 04/06 and complete tables provide supporting detail.
