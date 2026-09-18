# Paper figures, captions, and associated manuscript writing

New combined Figure 6 extension:
[14M/70M quality-sparsity and full-model latency](figures/08-14m-70m-quality-sparsity-latency.pdf),
with [caption, evidence and proposed writing](observations/025-matched-quality-latency.md).
It uses 44 trained checkpoints, a shared legend and all 40 measured clipping
latencies, while preserving the original Figure 6 and the manuscript.

New control-clipping measurement figure:
[final-kernel latency at 14M/70M](figures/07-controls-posthoc-final-latency.pdf),
with [caption, coverage and results](observations/024-controls-posthoc-final-latency.md).
The [implementation note for later text](observations/024-controls-posthoc-final-latency.md#implementation-note-for-later-text)
describes the tiling, short-row path and clipping overhead included in latency.
This Run036 figure is an additional analysis artifact; no manuscript adoption
or renumbering is implied.

**Selected main paper figure:** [Figure 6, side-by-side 14M quality and latency](figures/06-14m-quality-sparsity-latency.pdf),
with [caption and proposed writing](observations/023-14m-quality-latency-variant.md).
It preserves the original Figure 1 and uses the revised recipe naming and styling.
The readability revision adds the Pythia-14M title, wider layout, T/P legend,
open/filled control markers and a single post-hoc note.
This variant is now adopted as manuscript Figure 1 in the introduction; the
original analysis Figure 1 is still retained. See the
[adoption record](../../manuscript/draft/reviews/2026-09-18-main-figure/README.md).

Rebuilt from [task.md](task.md) on 18 September 2026. Each linked caption file
is also the figure's observation: it contains the question, method, coverage,
publication caption, result, proposed manuscript paragraph, caveats, and provenance.
The proposed paragraphs have not been inserted into manuscript TeX.

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
