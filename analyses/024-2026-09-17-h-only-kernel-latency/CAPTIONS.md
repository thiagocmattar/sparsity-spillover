# Paper figures, captions, and associated manuscript writing

[High-threshold operation bypass](figures/12-operation-bypass-summary.pdf)
is manuscript Figure 7: two panels, T4/Pall versus T7/Pall at kappa=0.5,
six operations per size. The [observation](observations/029-operation-bypass-summary.md)
defines the count-based bars and the rewritten subsection's explanation of
projection savings and attention implementation limits. Figure 10 remains
the complete diagnostic view.

[Paired pressure analysis](figures/03-pressure-scope-threshold.pdf) is now
manuscript Figure 4, with a shorter caption and two-paragraph Section 4.2.
The [observation](observations/015-pressure-scope-caption.md) and
[adoption record](../../manuscript/draft/reviews/2026-09-18-paired-pressure/README.md)
retain the subtraction convention, supporting contrasts and timing exceptions.

New all-model overview:
[quality-sparsity at 14M/70M/410M](figures/11-all-model-quality-sparsity.pdf),
with [caption, evidence and adopted writing](observations/028-all-model-quality-sparsity.md).
It includes all 74 trained paper conditions and the sixty fixed-control clipping
settings. Adopted as manuscript Figure 3 in Section 4.1; the common loss axis,
uniform final-loss pass, clipping visibility and cross-size limits are explicit.

New operation-bypass analysis:
[scalar sparsity and matrix-instruction bypass](figures/10-operation-bypass.pdf),
with [caption, metric and hypothesis check](observations/027-operation-bypass.md)
and the [complete per-setting table](TABLE_OPERATION_BYPASS.md).
It reuses saved counters for all 62 trained and 40 control-clipping settings;
the twelve size-specific panels now follow Figure 08's style. The observation
includes a paragraph explaining the implementation and runtime attribution;
its focused four-checkpoint summary is now adopted as manuscript Figure 7.

New Figure 4 extension:
[operation contributions across size and threshold](figures/09-14m-70m-operation-contributions.pdf),
with [caption, evidence and proposed writing](observations/026-operation-grid.md).
The 2-by-3 grid covers 24 retained Ph/Pall checkpoints at 14M/70M and κ=0,.05,.5,
using a separate contribution scale for each model-size row.

New combined Figure 6 extension:
[14M/70M quality-sparsity and full-model latency](figures/08-14m-70m-quality-sparsity-latency.pdf),
with [caption, evidence and proposed writing](observations/025-matched-quality-latency.md).
It uses 44 trained checkpoints, a shared legend and all 40 measured clipping
latencies. **This is now the selected manuscript Figure 1**, adopted unchanged
with a revised caption and associated results/scope text; see the
[adoption record](../../manuscript/draft/reviews/2026-09-18-two-size-main-figure/README.md).
The original Figure 6 is preserved.

New control-clipping measurement figure:
[final-kernel latency at 14M/70M](figures/07-controls-posthoc-final-latency.pdf),
with [caption, coverage and results](observations/024-controls-posthoc-final-latency.md).
The [implementation note for later text](observations/024-controls-posthoc-final-latency.md#implementation-note-for-later-text)
describes the tiling, short-row path and clipping overhead included in latency.
This Run036 figure is an additional analysis artifact; no manuscript adoption
or renumbering is implied.

**Previous main paper figure:** [Figure 6, side-by-side 14M quality and latency](figures/06-14m-quality-sparsity-latency.pdf),
with [caption and proposed writing](observations/023-14m-quality-latency-variant.md).
It preserves the original Figure 1 and uses the revised recipe naming and styling.
The readability revision adds the Pythia-14M title, wider layout, T/P legend,
open/filled control markers and a single post-hoc note.
This variant was the preceding manuscript Figure 1; Figure 8 now replaces it.
The original analysis Figure 1 is still retained. See the earlier
[adoption record](../../manuscript/draft/reviews/2026-09-18-main-figure/README.md).

Rebuilt from [task.md](task.md) on 18 September 2026. Each linked caption file
is also the figure's observation: it contains the question, method, coverage,
publication caption, result, proposed manuscript paragraph, caveats, and provenance.
Proposed paragraphs remain suggestions except for the manuscript adoptions
explicitly recorded above.

| Figure | Publication PDF | Caption and associated writing | Proposed placement |
|---|---|---|---|
| 1 | [Quality-sparsity trade-offs](figures/01-quality-sparsity-tradeoffs.pdf) | [Caption / writing](observations/013-quality-sparsity-caption.md) | Introduction; Section 4.1 |
| 2 | [Intervention sites](figures/02-intervention-sites.pdf) | [Caption / writing](observations/014-intervention-sites-caption.md) | Methodology |
| 3 | [14M/70M paired loss, sparsity and latency](figures/03-pressure-scope-threshold.pdf) | [Caption / writing](observations/015-pressure-scope-caption.md) | Section 4.2 |
| 4 | [Absolute operation contributions](figures/04-operation-sparsity-changes.pdf) | [Caption / writing](observations/016-operation-changes-caption.md) | Section 4.4 |
| 5 | [Quality, sparsity, and latency](figures/05-quality-latency.pdf) | [Caption / writing](observations/017-quality-latency-caption.md) | Section 4.5 |
| A1 | [Complete quality-sparsity results](figures/A1-complete-quality-sparsity.pdf) | [Caption / writing](observations/018-complete-quality-caption.md) | Results appendix |
| A2 | [Pressure versus none](figures/A2-pressure-versus-none.pdf) | [Caption / writing](observations/019-pressure-none-caption.md) | Results appendix |
| A3 | [Dense-reference speedup](figures/A3-dense-reference-speedup.pdf) | [Caption / writing](observations/020-dense-speedup-caption.md) | Results appendix |
| A4 | [Instruction-level explanation](figures/A4-instruction-level-explanation.pdf) | [Caption / writing](observations/021-instruction-caption.md) | Results appendix |
| Table 2 | [Fixed-pressure threshold contrast](TABLE-2-threshold-placement.md) | [Caption / writing](observations/022-threshold-table-caption.md) | Section 4.3 |

Figure numbering is proposed by task.md, not an automatic renumbering of the draft.
The scope/pressure legend is consistent across recipe plots. Figure 4 instead
uses operation colors, and the preserved A4 instruction cohort includes local
naive L1/OL1 controls (explicitly stated in its caption).

Manuscript adoption needs three explicit updates: replace old pending 70M/h-only
statements with the verified coverage; retain the distinction between the primary
54-checkpoint comparison and the historical 30-checkpoint instruction cohort;
use ordinary-final loss consistently for these new comparisons. The older draft
tables mixed ordinary-final h-only losses with logical-diagnostic losses for other
recipes, producing small numerical differences. Both loss passes are retained
separately in [the checkpoint table](data/paper-checkpoints.json).

All thirteen previous PDFs are preserved, byte-for-byte, in
[figures/.archive](figures/.archive/README.md). Historical observation links now
point there; their scientific claims and historical source-hash records are retained.
