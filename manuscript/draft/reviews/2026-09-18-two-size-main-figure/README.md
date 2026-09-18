# Two-size main Figure 1 adoption

## Request and result

At the author's request, Analysis024 Figure08 replaces the preceding 14M-only
overview as Figure 1 in the introduction, under the existing
`fig:quality-sparsity-overview` label. It appears on page 2 of the
[rebuilt 35-page draft](../../main-14m-70m.pdf). The open canonical `main.pdf`
is locked against replacement; it remains the older build. The TeX sources
and new figure asset are current.

The [paper figure](../../figures/08-14m-70m-quality-sparsity-latency.pdf) is
byte-identical to the [approved source](../../../../analyses/024-2026-09-17-h-only-kernel-latency/figures/08-14m-70m-quality-sparsity-latency.pdf).
The approved title, labels, colors, markers, annotation positions and all data
coordinates are preserved. The caption defines the shared T4/T7 and Ph/Pall
legend, Base/A0 and ReLU/A1-H aliases, and the four panel references.

## Evidence and coverage

The source [observation](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/025-matched-quality-latency.md)
and [builder](../../../../analyses/024-2026-09-17-h-only-kernel-latency/16_plot_matched_quality_latency.py)
document the question, methods, caption, results and limits. The unchanged
evidence export is copied as
[figure1-quality-latency.json](../../supplementary-data/figure1-quality-latency.json).
Figure and data manifests record source paths and SHA-256 hashes.

Each model-size row contains the same 22 trained checkpoints: Base, ReLU and
T4/T7 with h-only or all-site OL1 at five thresholds. All 40 fixed-control
clipping settings have measured latency; 15 high-loss clipping settings lie
above the focused quality ranges and are disclosed in the caption, with
references to the complete appendix clipping curves. Validation uses all 338
complete blocks; the excluded tail is 1,444 tokens. FP16 quality/counts remain
distinct from BF16 timing. The 70M kernel is the qualified shape-specific port.

The introduction, study coverage, cross-size results, execution discussion,
conclusion and evaluation scope now reflect available 70M h-only and clipping
results. An additional ten-row appendix table supplements the preceding
64-condition table, giving 40/22/12 conditions at 14M/70M/410M (74 total).
The historical table retains its values and loss-pass convention; Figure 1
uniformly uses ordinary final-checkpoint loss for trained points.

At the highest threshold, expanding T7 pressure increases sparsity at both
sizes but raises loss at 14M and lowers it at 70M. The more sparse 70M model
has higher final-kernel latency. These are descriptive single-seed comparisons.
No 70M no-pressure control or 410M h-only result is inferred. Historical
35-checkpoint skipping ablations remain separate, and small cross-session
timing differences are not treated as reliable rankings. Post-hoc timings
include external clipping at positive targets; zero-dose identity masks are
elided. No training, benchmarking, other figure adoption or cloud work ran.

## Verification

Four focused Figure08 evidence tests pass. The paper asset and evidence copy
match their source hashes; all ten new appendix rows match the source records
after declared rounding. Numerical comparisons in the revised text were
checked against those records.

The draft was rebuilt with BibTeX and repeated pdflatex passes in an isolated
directory, with resolved references/citations and no overfull boxes. All
35 pages were rendered and visually checked; the final caption pages 2 and 10
were checked again after the zero-dose wording correction. All fonts are
embedded. Six underfull vertical boxes, one underfull horizontal box and the
existing longtable's ignored glue-shrinkage diagnostic remain; affected
pages have no clipping or overlapping content.

[verification.json](verification.json) records hashes, coverage, table checks,
build diagnostics and output identity. Rebuild from `manuscript/draft` with
`pdflatex main`, `bibtex main`, then two further `pdflatex main` passes once
the canonical output file is writable. The review build uses that same source
with an isolated output directory to avoid the viewer lock.
