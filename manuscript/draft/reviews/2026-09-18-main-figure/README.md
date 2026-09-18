# Main Figure 1 adoption

## Current readability revision

The author's follow-up adds the requested title, T/P legend names, a wider
8.6-by-3.85-inch layout, and one shared post-hoc note. The final axis/legend
font is 11 points (about 7 points at the paper's 5.5-inch width), with 10-point
ticks and a 14-point title. The two controls use larger open-gray/filled-olive
circles at their original coordinates. All measurements, membership, limits,
colours, and pressure line styles remain unchanged.

The current paper asset matches the analysis PDF and `figures/SOURCES.json`.
The open `main.pdf` could not be overwritten, so the rebuilt 32-page draft is
[main-readability.pdf](../../main-readability.pdf), with Figure 1 on page 2.
All manuscript TeX files remain unchanged. References resolve; no overfull
boxes occur. Four underfull vertical boxes, one underfull horizontal box, and
the existing appendix longtable's ignored glue diagnostic remain. The figure
and pages 2-13, 17, and 30 were rendered and visually reviewed.
See [readability-verification.json](readability-verification.json) for the
current hashes and checks. The initial adoption record below is historical;
its hashes and PDF layout describe commit `4282dca`.

## Initial adoption

At the author's request, the approved Analysis024 Figure06 is now Figure 1
on page 2 in the introduction. It replaces the earlier quality-only overview
under the existing `fig:quality-sparsity-overview` label.

## Artwork and evidence

At initial adoption, the [paper copy](../../figures/06-14m-quality-sparsity-latency.pdf) was byte-identical
to the [approved analysis PDF](../../../../analyses/024-2026-09-17-h-only-kernel-latency/figures/06-14m-quality-sparsity-latency.pdf), with
SHA-256 `0712761cc05b1295015a5e14cb9323895d2ffe158cc8d96bbdbc22c5f3442859`.
Its source is pinned in [SOURCES.json](../../figures/SOURCES.json).
The [source observation](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/023-14m-quality-latency-variant.md),
[plot script](../../../../analyses/024-2026-09-17-h-only-kernel-latency/14_plot_14m_quality_latency.py),
and [point provenance](../../../../analyses/024-2026-09-17-h-only-kernel-latency/data/14m-quality-sparsity-latency.json)
retain the scientific definitions and checkpoint identities.

Both panels show the same 22 final 14M checkpoints: the base and GeLU-to-ReLU
controls, plus four threshold/pressure recipes at five thresholds each.
The caption describes validation coverage, pooled sparsity, post-hoc paths,
analytic ceilings, the BF16 K050 timing protocol, and the separate physical
GPU/host session for seven-site h-only timing. The quality panel retains the
approved view: eight high-loss post-hoc points are outside its limits and the
caption directs readers to the complete appendix figure. These clipped controls
have no matched timing measurements in this adopted evidence snapshot.

The introduction now distinguishes its 40-condition training study from the
22 checkpoints in Figure 1. Stale seven-site h-only timing statements in the
kernel section, appendix, and conclusion now point to Figure 1. Historical
35-checkpoint kernel figures and their skipping ablations retain their scope.
This adoption adds no training, timings, 70M results, or new statistical analysis.

## Verification

The draft was built from `manuscript/draft/` with `pdflatex
-interaction=nonstopmode -halt-on-error main.tex`, `bibtex main`, and two further
identical pdflatex passes. All commands completed successfully. The final PDF
has 32 pages; Figure 1 resolves to page 2. References and citations resolve,
all PDF fonts are embedded, and no overfull boxes occur.

The final log contains one underfull horizontal box and five underfull vertical
boxes. It also reports an ignored infinite-glue-shrinkage diagnostic at the
appendix longtable on page 17. That page's extracted text is unchanged from the
pre-adoption PDF, and its rendered layout was checked. These diagnostics were
not silently described as a warning-free build.

Rendered pages 1-13, 17, 28, and 30 were visually reviewed, covering every page
whose normalized extracted text changed and the pages with layout diagnostics.
No clipping or overlapping content was found. Bibliography and ICLR style files
remain unchanged. [verification.json](verification.json) records the source and
output hashes, checkpoint checks, build diagnostics, and review coverage.
