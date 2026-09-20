# Figure 3: 70M quality and latency with a native PyTorch reference

## Question and approved scope

The author requested a 70M counterpart of Figure 1, with quality versus
model-wide sparsity in (a) and full-model latency versus sparsity in (b).
The latency reference must be the native PyTorch base model, making the
transferred kernel's overhead visible. The author also requested an honest
transfer/optimization discussion, the missing 410M appendix reference, and
removal of the separate main-text speedup graph (formerly Figure 4).

## Method and coverage

[36_plot_70m_native_reference.py](../36_plot_70m_native_reference.py) reuses the
22 canonical 70M endpoints and 20 fixed-control clipping settings from
`data/14m-70m-quality-sparsity-latency.json`. The six available recipes are
Base, GeLU -> ReLU, T4/Ph, T7/Ph, T4/Pall and T7/Pall. Each pressure recipe
has kappa 0, 0.01, 0.05, 0.1, 0.5. No unavailable no-pressure, T1/Ph or T2/Ph
70M checkpoints are invented. Colors, line styles, 90% typography and the
8.6 x 3.85 inch canvas follow Figure 1; the six recipes use a 2 x 3 legend.

Panel (a) preserves all quality/sparsity coordinates and both clipping paths.
Seven high-loss clipping points extend above the focused view. Panel (b)
preserves the 21 non-base original-port latencies and substitutes the
native PyTorch/SDPA latency only for the Base marker. The same value is drawn
as a horizontal reference. The original specialized base measurement remains
in the data export, explicitly distinct from the plotted latency.
Clipping timings are not plotted.

Canonical quality and pooled logical counts cover all 338 complete validation
blocks from 500 documents, excluding the 1,444-token tail. Latency is the
geometric mean of synchronized host times on RTX5090: BF16, batch one,
2,048 tokens and all 50,304 logits, CUDA graphs, 64 inputs x seven passes x
three processes. The entire 70M trained grid and native reference use the same
Run035 GPU session. Recurring gates and preprocessing are included; static
preparation, capture and equal input staging are excluded.

## Results and interpretation

| Comparison | Retained value |
| --- | ---: |
| Original-grid native T0/P0 reference | 1.663883297 ms |
| Same base with original specialized port | 3.313247016 ms |
| Specialized/native dense latency ratio | 1.991274 |
| Best original-grid native-base speedup | 1.028123x |
| Non-base grid checkpoints faster than native base | 3 of 21 |
| Separate Run040 native base | 1.588591605 ms |
| Run040 selected T7/Ph, kappa 0.5 | 1.328911792 ms |
| Run040 selected native-base speedup | 1.195408x |

The complete Run040 result is not inserted into the original-session curves.
Its final implementation was selected on training inputs and tested on the
base and one sparse checkpoint, in three fresh full-validation processes each.
The appendix reports this separate experiment and its own native reference.
It retained sparse h/z, widened output-column tiles from 128 to 256, and used
native a/m projections and attention. The dense path remained slower than
PyTorch. Native h/z replacement localized most dense-path overhead, but the
conditional replacement does not separate the costs of padding, inspection
and memory traffic. Widening output columns does not change the empty-input
grouping rule and cannot establish a tile-size/skipping-frequency trade-off.

The original dense-kernel-normalized 2.047x ratio is not a native-base gain.
The old main-text speedup figure is removed, along with the stronger scale
claim. The remaining execution paragraph uses absolute latency examples.
Post-hoc statements now use the fixed native reference: maxima 1.025019x
(14M) and 0.613449x (70M). The small 14M difference spans GPU sessions and is
not evidence of a reliable clipping benefit. Historical PDFs/data are kept.

## Figure and caption

[PDF](../figures/23-70m-quality-sparsity-native-latency.pdf), copied to
`manuscript/draft/figures/23-70m-quality-sparsity-native-latency.pdf`.

**Quality, sparsity and execution on Pythia-70M.** (a) Validation loss and
(b) full-model latency versus model-wide sparsity for 22 trained checkpoints:
Base, GeLU -> ReLU, and T4/T7 with Ph or Pall at kappa 0, 0.01, 0.05, 0.1, 0.5.
Colors and line styles match Figure 1. Dotted paths in (a) clip the fixed
Base/ReLU controls; high-loss tails extend beyond the view. Vertical guides
mark sparsity ceilings. In (b), the Base marker and horizontal reference use
native PyTorch/SDPA (1.664 ms); all other markers use the original 70M port of
the 14M kernel. All trained timings use the same RTX5090 session, batch one,
2,048 tokens and full output logits. Later optimization results are reported
separately in the kernel appendix.

## Manuscript writing and limits

`experimental-study.tex` introduces the three model-size roles, the shared
protocol, quality costs and the limited transfer result before
`training-results.tex` begins From sparsity to speedups. The 410M stress-test
discussion points to `app:410m-results` and the retained training trajectories.
`kernel-appendix.tex` adds `app:kernel-transfer` and the two-checkpoint table
`tab:70m-kernel-transfer`. The introduction and 14M clipping appendix reference
the appropriate new/existing figure. Later paired-pressure text is preserved.

This is one training seed per recipe and one inference shape/device class;
neither the original grid nor the targeted follow-up proves a scaling law.
Losses remain canonical checkpoint losses, not the BF16 qualification losses.
No provisional optimization results are used, and no experiment was launched.

## Verification and provenance

[37_verify_70m_native_reference.py](../37_verify_70m_native_reference.py)
recomputes the native reference from all 1,344 primary Run035 native timings,
checks unchanged coordinates and explicit backend selection, checks the
separate optimization session and table rounding, verifies manuscript copy
hashes and embedded fonts, and confirms the removed figure reference is absent.
The builder checks source hashes and all six final Run040 full-validation gates.
See [verification](../data/70m-native-reference-verification.json) and
[figure data](../data/70m-quality-sparsity-native-latency.json).

A temporary 21-page manuscript build has resolved references/citations, no
overfull boxes, and three existing underfull horizontal-box warnings. All pages
and the standalone figure were visually checked. Figure 3 appears on page 5;
the transfer evidence is in Appendix E. `main.pdf` is not replaced by this edit.
