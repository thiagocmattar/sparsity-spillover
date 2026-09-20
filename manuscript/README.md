# Living Manuscript Draft

20 September 2026, requested rebuild with the AI-use statement:
`draft/main.pdf` contains the current author-edited sources and the disclosure
on page 9. The 20-page build has resolved references/citations and no overfull
boxes; three underfull-box warnings remain. A dangling introduction reference
to the now-commented paired-pressure figure points to Figure 1 instead.
All pages were visually checked, and the installed PDF matches the checked
build. Other author source edits remain preserved and uncommitted.

20 September 2026, approved AI-use disclosure:
`draft/main.tex` includes an unnumbered AI use statement after the conclusion
and before the references. It records the approved research, coding, analysis
and writing assistance, references kernel validation, and states author
responsibility. This follows the [ICLR 2027 AI policy](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).
Only the source is updated; `draft/main.pdf` has not been recompiled.

20 September 2026, Run044 integration: Figure 1 now includes the missing
14M T2/Ph kappa=0.5 endpoint (41 checkpoints). The complete-results table and
data also restore the four existing Run041 T2 points, giving 79 endpoints
(45/22/12 at 14M/70M/410M). Historical measurements and clipping data are
unchanged. The compact table preserves the 21-page layout; Figure 1 is on
page 2 and the complete 14M table on page 16. The installed PDF has resolved
references/citations, no overfull boxes and three pre-existing underfull-box
warnings. A label alias fixes an existing dangling kernel-appendix reference.
Other author edits remain preserved and uncommitted. See
[Analysis027](../analyses/027-2026-09-20-run044-manuscript/README.md).

20 September 2026, requested rebuild after the 70M manuscript revision:
`draft/main.pdf` now contains the current author-edited sources, the revised
Figure 3 with the native PyTorch reference, and the separate 70M optimization
discussion. The former speedup figure is removed. LaTeX confirmed the checked
21-page build is up to date; references/citations resolve, with no overfull
boxes and three existing underfull horizontal-box warnings. The installed
PDF matches the reviewed build byte-for-byte. No TeX source changed during
this rebuild; other author edits remain uncommitted.

20 September 2026, 70M execution comparison and experimental-study revision:
Figure 3 now shows 70M quality and latency versus model-wide sparsity, using
the native PyTorch base latency (1.664 ms). The former Figure 4 speedup graph
is removed. The text distinguishes the original transfer (best 1.028x native
base speedup) from Run040's separate two-checkpoint optimization (1.195x),
adds its controlled diagnosis to Appendix E, and explicitly links the 410M
stress-test results in Appendix D.3. See [Observation 046](../analyses/024-2026-09-17-h-only-kernel-latency/observations/046-70m-native-reference.md).
The temporary 21-page build has resolved references/citations, no overfull
boxes and three existing underfull horizontal-box warnings; all pages were
visually checked. This source/figure edit does not replace `draft/main.pdf`.
Other author edits are preserved and remain uncommitted.

20 September 2026, requested rebuild after moving Table 1:
`draft/main.pdf` is updated to the current author-edited sources. Table 1
appears directly above Figure 2 on page 4, in Sparsification interventions.
The 20-page build has resolved references/citations and no overfull boxes;
three existing underfull horizontal-box warnings remain. Changed pages
were visually checked and all other pages match the previous reviewed PDF.
The installed PDF hash matches the checked build. No TeX source was changed;
the existing author edits remain uncommitted.

20 September 2026, Table 1 placement:
the intervention table now appears in Methodology, under Sparsification
interventions, directly before Figure 2. The text introduces the table there
and Experimental study refers back to it. All table entries and its caption
are unchanged; the figure is kept after the table. Other author edits are
preserved. A temporary 20-page build confirms Table 1 directly above Figure 2
on page 4, with resolved references and no overfull boxes. This source edit
does not replace `draft/main.pdf`.

20 September 2026, requested rebuild after the Figure 1 frontier restoration:
`draft/main.pdf` now includes the latest author edits and the revised
Figure 1, with post-hoc clipping only in panel (a). The 20-page build has
resolved references and citations, no overfull boxes, and three underfull
horizontal-box warnings. All pages were visually checked. The installed
PDF matches the checked build; no TeX source was edited during this rebuild.
Existing author edits remain uncommitted. Earlier notes describe prior builds.

20 September 2026, Figure 1 intervention focus:
all 40 trained checkpoints and ten recipe styles are preserved. Post-hoc
frontiers are restored only in panel (a); panel (b) spans 0.445--0.665 ms. The caption and figure
reference are updated. See [Observation 045](../analyses/024-2026-09-17-h-only-kernel-latency/observations/045-main-figure-ten-recipes.md).
`draft/main.pdf` is unchanged and has not been recompiled.

20 September 2026, Figure 1 and introduction scale correction:
Figure 1 is now a 14M-only quality/sparsity/latency view with 22 trained
checkpoints and 20 clipping settings, preserving all corresponding values.
The caption describes the new two-panel figure, and the introduction makes
14M the main study and 70M a scale check. The third observation now states
that skipping can reduce latency at both sizes but kernels require adaptation.
See [Observation 044](../analyses/024-2026-09-17-h-only-kernel-latency/observations/044-14m-main-figure-scale-check.md).
The standalone figure and data are checked. `draft/main.pdf` is unchanged and
has not been recompiled, as explicitly requested. Other sections await the
author's later guidelines.

20 September 2026, requested rebuild of the author's latest abstract:
the two remaining sparsity placeholders are filled with the previously checked
27.48% (14M T7/Pall) and 12.71% (T4/Pall). Other author wording is preserved.
The rebuilt 20-page `draft/main.pdf` has resolved references/citations and no
overfull boxes; all pages were visually reviewed. Three existing underfull
horizontal-box warnings remain. Only the comparison sentence, PDF and this
build note are committed; the other author edits remain in the working tree.

20 September 2026, author-guided abstract revision: P/T and the 14M/70M scope
are explicit, the measures sentence is simplified, and the conclusions now
include checked data for ceiling attainment and quality cost, the sparsity/
latency distinction, T4/Ph and nonlocal effects, and h/z execution savings.
The abstract ends with the requested training/kernel co-design takeaway.
The introduction is unchanged. The rebuilt 20-page PDF has resolved references
and checked layout, including the introduction/Related Work transition.
See the [revision and evidence record](draft/reviews/2026-09-20-introduction-abstract/README.md).

20 September 2026, surgical revision of the introduction and abstract:
the previous compressed rewrite is superseded. The author's original five
observations, explanatory flow, metric definitions and closing sentences are
restored, with targeted cuts, bold findings, supporting references and a few
evidence updates. See the [revision record](draft/reviews/2026-09-20-introduction-abstract/README.md).
The rebuilt 20-page `draft/main.pdf` has resolved references/citations and
no overfull boxes. The entire findings paragraph fits below Figure 1 on page 2,
with Related Work following normally. Earlier entries describe prior versions.

20 September 2026, introduction and abstract rewrite: the introduction closes
with four bold findings supported by representative values and figure/table
references. The abstract is approximately half as long and emphasizes local
pressure, placement-dependent execution gains, and joint training/kernel design.
The reference-dependent scale claim and older regression statistics are omitted
from these summaries. See the [editorial review](draft/reviews/2026-09-20-introduction-abstract/README.md)
for evidence and scope. The rebuilt 20-page `draft/main.pdf` has resolved
references/citations and no overfull boxes; the opening and remaining page
layouts were visually checked. Earlier author prose remains unchanged.

20 September 2026, compact results appendix and whitespace correction:
`draft/results-appendix.tex` now presents all 74 endpoints in three grouped
T/P tables: 14M-only ablations, side-by-side 14M/70M results, and 410M.
The old contrast/density/accounting displays and full 70M/410M clipping
sweeps are removed from the paper. A restyled two-panel 14M clipping figure
retains all 300 measured settings. The 410M stress-test artwork and its two
discussion paragraphs are preserved. See [Observation 043](../analyses/024-2026-09-17-h-only-kernel-latency/observations/043-compact-results-appendix.md)
for the editorial rationale, loss-convention reconciliation and provenance.
Tables 6--7 share page 16; Figure 9 is on page 17, and Table 8/Figure 10 are
on page 18. Normal vertical spacing and targeted figure/table placement remove
large gaps; Related Work now follows the introduction on page 2.
Eight focused tests pass. The 20-page `draft/main.pdf` has resolved references
and citations, no overfull or underfull vertical boxes, and visually checked
layout. Three underfull horizontal-box warnings remain. Existing author prose
is preserved; only the requested layout edits are committed in the affected
main-text files. Earlier entries describe previous versions.

20 September 2026, requested rebuild after the appendix revisions:
`draft/main.pdf` now reflects the current author-edited sources, including
the kernel appendix and OL1/accounting/setup revisions. The 30-page build
has resolved references and citations and no overfull boxes. All pages were
reviewed in contact sheets; pages 12--13 and 28--29 were checked at higher
resolution. Six underfull-box warnings remain. The installed PDF's SHA256
matches the reviewed build. No TeX source was changed; existing author edits
remain uncommitted in the working tree. Earlier rebuild notes describe
previous versions.

20 September 2026, kernel-appendix refactor: `draft/kernel-appendix.tex` now
contains one section covering execution rules, numerical checks, timing,
instruction counts and unsuccessful alternatives. The old search figure,
native-per-checkpoint fit and detailed historical runtime tables are removed
from the paper, with their source artifacts retained. A new 14M/70M diagnostic
uses one native base latency per size and no fitted lines. The audit identifies
the material 70M reference difference: maximum native-base gain is 1.028x,
versus 2.047x relative to the slower specialized base implementation.
The main figure and its scale claim are pending separate author direction;
the appendix makes both definitions explicit. See [Observation 041](../analyses/024-2026-09-17-h-only-kernel-latency/observations/041-kernel-appendix.md).
Seven focused checks pass and the standalone figure is visually checked.
`draft/main.pdf` is unchanged, awaiting authorization to recompile.

20 September 2026, second appendix prose pass: the OL1, sparsity-accounting
and experimental-setup sections use shorter explanations and consistent
terminology, preserving equations, numerical results and table entries.
The L1/OL1 comparison distinguishes a single-seed observation from evidence
about variation across seeds; the geometry interpretation does not attribute
the effect to individual attention sites. The OL1 figure no longer permits
ordinary float-only placement, which produced the stretched table/figure page
13. Following appendix sections can share its page. Source checks pass;
`draft/main.pdf` is unchanged and recompilation/layout verification awaits
the user's instruction. Other author edits are preserved.

20 September 2026, experimental-setup rewrite: `experimental-appendix.tex`
now contains the compact architecture, ceiling, training and evaluation
details. T/P names replace the rendered A* labels; explicit per-recipe counts
replace the operation-set notation. The training table retains its numerical
entries and adds effective batch size and sequence length. Shared settings
were checked across all 74 retained runs. Base-model trajectory panel titles
now read T0/P0, with unchanged data. See [Observation 040](../analyses/024-2026-09-17-h-only-kernel-latency/observations/040-experimental-setup.md).
After the user's follow-up authorization, `draft/main.pdf` was rebuilt at
34 pages, with resolved references/citations and no overfull boxes. The revised
appendix and training table/figure were visually checked on pages 15--16.
The subsequent complete-results section and unrelated author edits are preserved.

20 September 2026, sparsity-accounting rewrite: `app:accounting` now uses the
main text's model-wide sparsity, ceiling and lowercase q/k/v/p notation. The
section explains counts for linear projections and causal attention, the
shared denominator and ceiling propagation with a concrete v-to-z example.
Unused site/block sparsity symbols and graph/workload terminology were removed;
counting definitions and cross-reference labels are preserved. TeX structure
and references were checked without compilation. `draft/main.pdf` is unchanged,
as requested; unrelated author edits remain in the working tree.

20 September 2026, requested rebuild: `draft/main.pdf` reflects the current
author-edited sources. The 35-page build has no undefined references/citations
or overfull boxes; pages 4, 12 and 13 were visually checked. The installed
PDF hash matches the reviewed build. No TeX source was changed; author edits
remain uncommitted in the working tree.

19 September 2026, OL1 appendix revision: `app:pressure` now introduces the
code-checked dual update before diagnostics, using the main-text ell_1 notation.
Table 3 compares L1/OL1 at every retained 14M T1/P1 lambda. Figure 7 extends
the geometry view to all 20 14M Figure 1 OL1 conditions, using the same
recipe styles in two panels. The requested existing PDF path is retained. The old 14M-only
diagnostic subsection is consolidated here. The main-method paragraph now
distinguishes ordinary L1 from OL1 and qualifies saturation invariance.
See [Observation 039](../analyses/024-2026-09-17-h-only-kernel-latency/observations/039-ol1-appendix.md)
for sources and limits. All 16 focused tests pass; the 35-page build has resolved
references/citations and no overfull boxes. The new figure and appendix pages
12--13 were visually checked. Other author edits remain in the working tree.

19 September 2026, requested rebuild: `draft/main.pdf` now reflects the current
author-edited manuscript. A missing closing brace in Figure 1's caption was
repaired in `draft/introduction.tex`; no wording changed. The 34-page build has
no undefined references/citations or overfull boxes. Pages 1, 2 and 9 were
visually checked, and the installed PDF's hash matches the reviewed build.
The author's source edits, including the one-brace repair within the revised
caption, remain in the working tree; this rebuild records only the PDF and note.

19 September 2026, Table 2 caption clarification: timing units are now explicit
as microseconds per complete 2048-token sequence. Brackets give the minimum and
maximum savings across all pairs of independent off/on timing-run means, not
confidence intervals or variation across tokens/sequences. The shorter caption
retains the sign convention, repetitions and non-additivity; measurements are
unchanged. See [Observation 034](../analyses/024-2026-09-17-h-only-kernel-latency/observations/034-operation-latency-manuscript.md).
The 34-page `draft/main.pdf` is rebuilt, with resolved references/citations,
no overfull boxes and a visually checked table on page 9. Author edits are preserved.

19 September 2026, a/m implementation justification: the first paragraph of
`draft/kernel-autoresearch.tex` now reports the user-approved Run039 follow-up.
The h/z strategy increases latency at a/m despite unchanged outputs; the text
connects the measured empty-tile fractions and extra padded matrix instructions
to the different activation structure at h/z. It identifies this as a follow-up
test, without attributing an unperformed test to the original agentic search.
Evidence: [Observation 035](../analyses/024-2026-09-17-h-only-kernel-latency/observations/035-am-load-avoidance-port.md)
and its Run039 sources. The current working draft builds at 34 pages with no
undefined references/citations or overfull boxes; the revised paragraph and
adjacent table were visually checked on pages 8--9. Existing author edits are
preserved. After the viewer was closed, the checked build replaced
`draft/main.pdf`; the installed PDF's SHA256 matches the reviewed copy.

19 September 2026, mechanism evidence table: Section 4.3 now defines instruction
bypass, distinguishes the h/z weight-load avoidance from later a/m and attention
checks, and explains the FlashAttention/softmax sequence. Sparsifying attention
probabilities is an untested opportunity. Table 2 replaces the bypass chart,
using 12 operation counts and six grouped timing means checked against original
diagnostics and 8064 raw candidate timing samples. Individual-site attribution
is a separately approved experiment, not a claim drawn from these timings.
Evidence: [Observation 033](../analyses/024-2026-09-17-h-only-kernel-latency/observations/033-kernel-mechanism-table.md).
The canonical 34-page `draft/main.pdf` is rebuilt with resolved references and
citations, no overfull boxes, and a visually checked subsection/table on page 9
and new FlashAttention bibliography entry on page 10. Older entries describe
previous versions.

19 September 2026, focused attention-latency explanation:
`draft/kernel-autoresearch.tex` is reduced to two paragraphs and one figure,
with about 30% less prose. It explains that QK/PV skipping retains input reads
and softmax/output work, while the added skipping overhead exceeds the arithmetic
savings in the matched 14M controls. The instruction-bypass artwork is unchanged;
the operation-contribution figure is removed from the document, with its data
and PDF preserved. Appendix references now describe those retained counters
without referencing the removed figure. Evidence:
[Observation 029](../analyses/024-2026-09-17-h-only-kernel-latency/observations/029-operation-bypass-summary.md).
The verified 34-page build now replaces `draft/main.pdf`, with resolved
citations/references, no overfull boxes, and a visually checked Section 4.3
on page 9. The installed PDF hash matches the checked build.

19 September 2026, nonlocal pressure response: the final paragraph of
`draft/training-results.tex` now describes the retained 14M h/z comparisons
at kappa=0.05 and distinguishes the possible OL1-cap explanation from an
identified mechanism. The approved two-page viridis heatmap is included as
Figure 6 on page 8, immediately after the discussion on page 7. Evidence and
caption: [O017](../analyses/021-2026-09-10-training-results-figures/pressure-scope/observations/O017-layer-sparsity-style.md)
and its linked O016 pressure diagnostics. `draft/main.pdf` is rebuilt at
35 pages with resolved references and citations, no overfull boxes, and the
affected pages visually checked. The author's other paragraph edits are retained.

19 September 2026, author revision and rebuild: the experimental setup,
methodology, captions, and sparsity-to-speedup discussion incorporate the
author's edits; the tile explanation is retained as a source comment.
The Table 1 reference and the distinction between 64 timing sequences and
338 correctness-check sequences are corrected. `draft/main.pdf` is rebuilt
at 33 pages with resolved references and citations, no overfull boxes, and
the revised pages visually checked. Earlier notes describe prior versions.

19 September 2026, tile explanation: `draft/training-results.tex` now gives
the activation-tile sizes (tokens by features) and distinguishes skipped
matrix operations from avoided weight reads at h/z. Checked against
[K042 input projections](../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k042/projection.cu),
[K049 output projections](../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k049/joint.cu),
and the corresponding Run035 70M port. The shorter paragraph retains the
nearly empty row path and the cost of detecting zeros.

19 September 2026, protocol wording: the agentic-kernel paragraph now
explains the comparison with the original model, preset limits on output
differences, and tests on the saved model from each training recipe.
It distinguishes the fixed training-text subset used during development
([Run028](../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/README.md))
from the 64 validation sequences used for timing and all 338 used for
output checks ([Run029](../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/README.md)).
The same per-model checks apply after the
[70M adaptation](../runs/035-2026-09-18-pythia70m-k050-port/README.md).

19 September 2026, paragraph polish: the agentic-kernel introduction in
`draft/training-results.tex` now describes the human-guided workflow in
plain language and uses `\citet{cetin2026sparser}` for the released starting
kernels. It states numerical agreement within the prescribed tolerances,
consistent with Appendix D.4 and the Run029 implementation audit, rather
than exact output equality. Other ongoing author edits are preserved.

19 September 2026, citation correction: `openaigpt6astra` now cites the
[dated OpenAI announcement](https://openai.com/index/gpt-6-astra/) with year
2026, replacing the undated API-documentation entry. OpenAI's
[release index](https://openai.com/research/index/release/?page=2) lists
3 September 2026. The inline author-year label is now `(OpenAI, 2026)`.

19 September 2026, figure polish: Figure 3's PDF now removes naive-L1,
shortens the T1/P1 label, uses independent loss scales, and restores the
three-row legend in a 10% narrower, 14.3% taller canvas. The include uses
90% line width; caption and overview counts are 36/22 (58 total). See
[observation 032](../analyses/024-2026-09-17-h-only-kernel-latency/observations/032-quality-ol1-layout.md).
The figure PDF and provenance are updated; canonical `main.pdf` remains
unchanged while the author is editing.

19 September 2026: [experimental-study.tex](draft/experimental-study.tex)
merges the matching/protocol and post-hoc setup paragraphs. Table 1's caption
now associates threshold kappa and pressure lambda/b with their executed
settings, preserves the historical T1/P1 lambda sweep, and motivates fixed
multisite settings using the conflict projection and norm cap in Appendix A.2.
It does not claim loss preservation. A temporary build and caption layout
check pass, with no overfull boxes; the author's existing unresolved
`\ref{tab}` remains. The canonical PDF and other ongoing author edits are
unchanged by this source-only revision.

The quality overview is now Figure 3, limited to 14M/70M, directly after
Table 1 on page 5. Base-model speedup follows as Figure 4 on page 6. The
complete 410M panel moves to appendix Figure 15 (page 26), beside the
unchanged full post-hoc trajectories in Figure 16 (page 27). The main setup
retains the controlled-study, targeted-replication and fixed-token stress-test
roles. [main.pdf](draft/main.pdf) is rebuilt at 33 pages; see the
[placement and preservation audit](draft/reviews/2026-09-18-quality-scope-placement/README.md).
Notes below retain their earlier revision states and figure numbering.

Analysis024 Figure13 is now Figure 3 immediately after the post-hoc calibration
paragraph in Experimental Study (page 5). The new discussion explains the
human-guided GPT-6 Astra workflow, how sparse execution can save work, the
approximately linear within-recipe trend and the limited clipping gains.
The clipping statement retains the 70M high-dose exception. The rebuilt
[main.pdf](draft/main.pdf) has 33 pages and resolved references. See the
[adoption and claim audit](draft/reviews/2026-09-18-base-speedup-adoption/README.md).
Notes below describe earlier revision states and numbering.

The threshold-placement and cross-size pressure subsections are removed.
The local-sparsity discussion is shortened and merged into Section 4.3,
"When model-wide sparsity translates to speedup." It leads with Analysis024's
operation-contribution figure (now Figure 5), then connects the high-threshold
QK/PV sparsity increase to instruction bypass and the tested kernel's timing
limitations (Figure 6). [main.pdf](draft/main.pdf) is rebuilt at 32 pages.
See the [integration and verification record](draft/reviews/2026-09-18-operation-sparsity-speedup/README.md).
Entries below describe earlier revision states.

Section 4.6 now gives a lean, first-principles explanation of avoidable
arithmetic, projection savings and the tested attention kernel's limitation.
Its replacement Figure 7 uses two grouped-bar panels: 14M/70M, T4/Pall versus
T7/Pall at kappa=0.5, six operations per size. Both are on page 9 of the rebuilt
33-page [main.pdf](draft/main.pdf). The subsection distinguishes instruction
bypass from time saved and does not claim optimal attention execution or
extend the 14M timing attribution to 70M. See the
[revision and verification record](draft/reviews/2026-09-18-kernel-explanation/README.md).
Entries below describe their earlier revision states.

Section 4.2 now uses Analysis024's six-panel paired-pressure figure as
Figure 4, comparing loss, sparsity and latency at 14M/70M. Its two paragraphs
state the h-only advantage at moderate thresholds and the two small 14M
latency exceptions at kappa=.5. The OL1 budget diagnostics remain in
Appendix D.6. The author's latest Section 4.1 edits are preserved, and
[main.pdf](draft/main.pdf) is rebuilt at 34 pages. See the
[paired-pressure revision](draft/reviews/2026-09-18-paired-pressure/README.md).
Entries below describe their earlier revision states.

The quality-sparsity subsection now includes a three-panel overview of all
74 trained paper conditions at 14M/70M/410M (Figure 3, page 6). It quantifies
the sparsity/quality cost, states the common 712-step budget, and explains the
limited 410M sweep and focus on 14M/70M without assuming a scaling law.
The canonical [main.pdf](draft/main.pdf) has been rebuilt and visually checked
(35 pages). Figure 1 remains unchanged. See the [revision and evidence record](draft/reviews/2026-09-18-all-model-quality-overview/README.md).
The notes below retain their earlier revision scope.

The current draft is [main.pdf](draft/main.pdf), with the approved
Analysis024 Figure08 as Figure 1 on page 2. The four panels compare quality,
sparsity and latency at 14M/70M: 44 trained checkpoints and 40 measured control
clipping settings. The caption defines the shared T/P legend and panel labels.
The introduction, results, scope statements and appendix now include the ten
70M h-only endpoints, for 74 trained conditions overall. Historical ablations
retain their original cohort. The source artwork is byte-identical.

Table 1 now uses T/P terminology, nine OL1/control recipes, horizontal rules
between threshold groups, and all three model-size ceiling columns. All-site
pressure means the pressure and thresholding sites coincide. Its caption and
setup text are shortened. The canonical 35-page PDF has been rebuilt, and the
two superseded preview PDFs have been removed. See the
[Table 1 revision record](draft/reviews/2026-09-18-table1-terminology/README.md)
and [Figure 1 adoption record](draft/reviews/2026-09-18-two-size-main-figure/README.md).
Entries below describe earlier revisions.

The 17 September pressure-placement revision integrates ten verified h-only
14M endpoints, giving 64 trained conditions, and restores five historical
kernel records for a 35-checkpoint runtime cohort. Threshold and pressure
scope are now explicit throughout the updated figures and results. The
[32-page draft](draft/main.pdf) and [verification record](draft/reviews/2026-09-17-pressure-placement/README.md)
retain the original ICLR format and bibliography. Seven-site h-only timing
and h-only scaling remain pending. The entries below describe earlier revisions.

The redundant main-text boundary-contrast table (formerly Table 3) is removed.
Section 4.5 states the high-threshold ordering and zero-threshold reversal
directly, referring to Appendix Table 7 for all 15 matched contrasts. Figure 6
and the appendix evidence are unchanged. The rebuilt draft remains 28 pages.

The frozen Analysis 021 cross-size figure is now Figure 6 on page 9, with
Section 4.5 focused on the recurring high-threshold complete-recipe ordering
and architectural ceilings. A0 optimization histories are Figure 8 on page 18
in Appendix C.4, following the realized-protocol table. The caption specifies
nine-step smoothing; the text and Discussion describe realized optimization
regimes without judging convergence. Sources and adoption checks are in
[O006](../analyses/021-2026-09-10-training-results-figures/observations/O006-scale-transfer.md)
and [O007](../analyses/021-2026-09-10-training-results-figures/observations/O007-a0-optimization.md).
The rebuilt draft remains 28 pages. Entries below describe earlier revisions.

Section 4.4 now states "Local sparsity does not determine model-wide sparsity,"
using the approved composite as Figure 5 on page 9. The full threshold density
grid is Figure 8 in Appendix D.1; three-size operation accounting remains in
Appendix D.2 as Figure 9. The Q/K/V comparison remains Section 4.3.
[Analysis 021 O005](../analyses/021-2026-09-10-training-results-figures/observations/O005-distributions-and-operations.md)
records the adoption and source evidence. The rebuilt draft has 28 pages.

Section 4.3 now isolates adding Q/K/V thresholding without pressure at matched
thresholds. Table 2 on page 7 reports the ten raw A4/A7 endpoints; the text
highlights +5.17 pp sparsity for +0.043 loss at kappa = 0.5. Evidence and
verification are recorded in
[Analysis 021 O004](../analyses/021-2026-09-10-training-results-figures/observations/O004-qkv-thresholding.md).
The rebuilt draft has 27 pages. The entries below record earlier revisions.

The frozen paired-pressure figure from
[Analysis 021 O003](../analyses/021-2026-09-10-training-results-figures/observations/O003-paired-interventions.md)
is now Figure 3 on page 6. Section 4.2, "Paired effect of adding pressure,"
focuses on OL1 minus no pressure at matched thresholds. It preserves the
target-set normalization caveat and geometry diagnostic; the local L1N/OL1
comparison is one sentence in Section 4.1 with an Appendix D table reference.
The rebuilt draft remains 26 pages.

On 11 September, the frozen OL1 mechanism figure from
[Analysis 021 O002](../analyses/021-2026-09-10-training-results-figures/observations/O002-ol1-geometry.md)
was adopted as Figure 4 on page 7. The methods, paired-effect discussion,
appendix and conclusion now distinguish the four-/seven-site cap regimes;
the ideal invariance to increasing lambda is conditional on the cap binding.
The rebuilt draft has 26 pages.

The 10 September quality-sparsity overview now uses
[Analysis 021](../analyses/021-2026-09-10-training-results-figures/README.md):
short intervention labels, both baseline and ReLU clipping, and explicit
four-/seven-site ceiling guides. The approved polished figure is now Figure 1
on page 2, in `draft/introduction.tex`, with a brief reference in the Pythia
paragraph. The experimental section refers back to it. The current draft has
26 pages; earlier artwork is retained.
The revision records below describe their historical snapshots.

This directory is the paper-facing scientific layer shipped inside the
bootstrap at its final `manuscript/` path. Copying the bootstrap into a new
repository therefore carries the manuscript and its workflow integration in
the same operation.

## Role

The draft sharpens the research question, terminology, formal definitions,
metric contracts, and intended contributions. It is allowed to lead the
research: a manuscript idea may motivate a new run, analysis, metric, or method.

It is not:

- a frozen experiment plan;
- proof that a contribution claim has been established;
- a substitute for an executed config, code, tests, or artifacts;
- required to describe every implementation detail before exploratory work can
  proceed.

The workflow does not try to keep the draft and implementation textually
identical. The currently documented manuscript/operational differences are an
accepted baseline, and operational contracts execute by default. A new
substantive discrepancy is surfaced during experiment design only when it would
materially change the experiment or the user requests a manuscript-specific
variant.

## V2 workspace

On 9 September 2026, the user enabled Git tracking for `manuscript/draft/`
to collaborate on the paper. The current TeX, bibliography, ICLR template,
figure PDFs, tables, supplementary measurements and compiled `main.pdf` are
tracked, together with manuscript history and review records. Rebuildable
LaTeX files, backup ZIPs and raster QA previews remain ignored; the official
ICLR template ZIP is a small provenance exception. The
[draft README](draft/README.md) documents the build and editing workflow.
Earlier references below to a local-only draft describe the previous policy.

At the author's request, `manuscript/draft/` has been restored exactly to its
first tracked version, commit `1e14471bd4763dbfef84a162d4429021d2f7e2e6`
("Track manuscript draft for collaborative editing"). The current
[27-page ICLR draft](draft/main.pdf), sources, figures and supporting files match
that snapshot. The later task-driven manuscript revisions remain in Git history;
their analysis records remain under `analyses/`. The restored
[draft README](draft/README.md) describes this version and its build commands.

## Earlier draft revisions

The official ICLR 2027 conversion used unmodified conference and bibliography
styles, anonymous authors, review headers and line numbers. Its then-current
27-page layout and verification are preserved in the
[conversion record](draft/reviews/2026-09-09-iclr2027-template/README.md).
The entries below describe earlier versions; their page counts, unchanged-asset
statements and terminology do not describe the current reading draft.

The 8 September revision added the fixed-threshold rationale and the kernel
specialization argument requested by the user. It distinguishes early search
gains from convergence, a descriptive sparsity-speedup association from a
scaling law, and zero-fragment skipping from profitable execution. The
[argument review](draft/reviews/2026-09-08-kernel-argument/revision-log.md) and
[O015 snapshot](../analyses/018-2026-09-08-results-materials/provenance/manuscript-20260908-threshold-kernel/README.md)
retain that revision's source checks and verified reading copy. Figures,
numerical tables, bibliography and introduction were unchanged in that revision.

Technical-reader, scientific-reviewer and literature sub-agents reviewed every
figure, table and result. The [revision log](draft/reviews/2026-09-08-polish/revision-log.md)
records their resolutions and the [verification](draft/reviews/2026-09-08-polish/verification.json).
All 26 pages of that earlier revision were inspected, all citations and labels resolve, and all fonts
are embedded. That review preceded the ICLR template conversion above.

Six main and four appendix figures retain their source artwork. On 9 September,
the user replaced the experimental setup's recipe table with the combined
[Pythia architecture map and sparsification ladder](artifacts/pythia-architecture-sparsification-ladder.pdf).
It is now Figure 1 on page 5, with the architecture above the eight-recipe
site/pressure map and analytic ceilings. The former appendix occurrence was
removed, its cross-reference updated, and all figure/table numbers resolved.
The caption preserves pressure targets and model-size coverage; the artwork
is byte-identical to the source composite. The scale-contrast table now stays
with its discussion after the reflow. The 26-page PDF builds without LaTeX
or box warnings, and the revised layout was visually checked.

Earlier on 9 September,
the user selected the [all-variant overview](draft/figures/01-v2-14m-overview.pdf)
for the quality--sparsity overview, now Figure 2. The reading copy embeds all 30 trained checkpoints, and
its caption and source comments now follow Analysis 018
[O013](../analyses/018-2026-09-08-results-materials/observations/O013-overview-all-variants.md)
and `04_overview_v2.py`; the original overview is retained. The build and affected
pages were checked. Draft sources and the PDF were local-only at that point. Seven
generated tables preserve every numerical row. The supplementary directory
retains all 540 clipping evaluations, seven signed histograms and a newly
sourced preparation/environment protocol.

The earlier full polish has a version-controlled
[source/PDF snapshot](../analyses/018-2026-09-08-results-materials/provenance/manuscript-20260908-polished/README.md).
[O014](../analyses/018-2026-09-08-results-materials/observations/O014-manuscript-polish.md)
records this revision. The original
[results snapshot](../analyses/018-2026-09-08-results-materials/provenance/manuscript-20260908-results/README.md)
and [figure plan](../analyses/018-2026-09-08-results-materials/MANUSCRIPT-PLAN.md)
remain historical records. No new experiment was launched.

### Earlier V2 revisions

The user requested a clean start for V2 on 2026-09-05 and then authorized an
argument-first introduction and related work, followed by a compact
methodology and a separate experimental-study opening. That revision contained
these four sections, general-method and experimental-detail appendices,
a checked bibliography, and a lightweight reading wrapper. The
framing concerns how pressure, gate nonlinearity, and site placement interact
in the quality--sparsity trade-off. The text develops a candidate explanation
and asks how it transfers. The experimental opening defines setup and
comparison logic; empirical findings were deferred until the results adoption above. The user's `draft/supplementary.md` remains unchanged.

[Draft positioning notes](draft/positioning.md) record the literature checks,
length budget, and evidence boundaries, including the combined gate/pressure
change in A4-OL1 versus A7-OL1, OL1's lack of a loss-convergence guarantee, and
the separation between logical sparsity and measured execution. The earlier
spillover claim is not reused as the paper's central argument. The existing
architecture/ladder artifact is included as a setup reference. Draft
sources and the reading copy were local-only at that point.

The subsequent user-requested adversarial review used three independent
sub-agents with explicit five-criterion scoring rubrics for writing, scientific
rigor, and literature/systems positioning. The revised introduction centers
separately controllable interventions whose effects need not be separable,
and states the reader's design decisions before introducing measurement.
Related work now contains three paragraphs on sparsification interventions
and two on inference speedup, with workload-specific execution reasoning and
no dedicated gradient-surgery strand. Detailed reports, initial/final source
snapshots, scores, and the response to criticisms are indexed in
[the review record](draft/reviews/2026-09-05-adversarial/README.md).
The review strengthens positioning; the empirical insight and its transfer
were obligations for the later results stage, now addressed within the stated one-seed limits.

The methodology defines independent gate and pressure choices, exact-zero
model-wide sparsity, and the all-zero selected-site reach ceiling, with most
math inline. The paper now uses calligraphic S while preserving operational
`R_*` fields and fractional units. Exact gates, OL1 mechanics, integer counters,
architecture derivations, and validation coverage are in the appendix;
[provenance notes](draft/methodology-notes.md) identify the implementation
contracts and immutable architecture configurations. The setup artifact uses
the same notation and larger lettering without changing its analytic values.

The [methodology review record](draft/reviews/2026-09-05-methodology/README.md)
retains three independent source audits, initial and revised scored reports,
source/PDF snapshots, a final visual follow-up, and exact ceiling verification.
Final editorial scores are 20/25 for writing, 23/25 for scientific rigor, and
22/25 for literature/presentation; these assess the methods draft, not paper
acceptance or empirical evidence. That reading copy built cleanly, its eight
pages were inspected, and all twelve analytic architecture/topology count
pairs matched the existing implementation. No scientific code or result was
changed and no experiment was launched.

The subsequent user-approved section split keeps Methodology focused on
gate/pressure definitions, sparsity, and structural reach (about 380 words).
The separate [Experimental Study](draft/experimental-study.tex) opening
contains Pythia/MiniPile context, A* recipes, matched-comparison logic, selected
larger-model scope, and the existing setup figure. Pythia-specific counts,
configuration links, and validation coverage moved into an experimental
appendix. The four numbered equations and the figure artwork are unchanged.

The [split review](draft/reviews/2026-09-05-methods-experiments-split/README.md)
includes an independent writing proposal, source audits, and two scored review
rounds. Final scores are 20/25 for writing and 23/25 each for scientific rigor
and literature/presentation; all substantive review issues were resolved.
The nine-page reading copy builds cleanly, with no unresolved references or
box warnings. No result or scientific code was changed and no run was launched.

The 4 September matched-intervention paper is preserved in
[v0_2026-09-05](draft/archive/v0_2026-09-05/README.md), including its PDF,
sources, review notes, and complete former working directory. That paper covers
detailed 14M contrasts, selected 70M recipes, operation-level explanations,
fixed-budget 410M appendix results, and negative 70M execution evidence.

[Analysis 013](../analyses/013-2026-09-04-matched-intervention-manuscript/README.md)
owns its five figures, generated tables, verified reduction, and observations.
The draft was local-only at that point. The older TeX
and report files below preserve earlier proposals and evidence cutoffs; their
410M-unobserved descriptions predate the completed cohorts.

## Draft archive

The local [archive index](draft/archive/README.md) lists snapshots named
`vN_YYYY-MM-DD`, starting at v0 and incrementing for each archive. The date is
the archive date; `draft/` is the clean workspace for the next manuscript.

| Version | Main argument / framing |
| --- | --- |
| [v0_2026-09-05](draft/archive/v0_2026-09-05/README.md) | Matched gate-placement and activation-pressure interventions improve parts of the quality-sparsity frontier; broader pressure can increase the quality cost. |

Each snapshot preserves the PDF, manuscript sources, bundled figures/tables,
style files, review notes, and a SHA-256 inventory. The archive retains the
draft's former local-only Git policy at creation; manuscript sources and PDFs
are now tracked, while temporary outputs remain local. This index records its location
and framing. v0 precedes the proposed pressure-allocation-by-nonlinearity study.
Its `working-folder/` preserves all 102 files moved out of `draft/`, including
earlier revisions, QA renders, and build outputs, with verified hashes recorded
in `working-folder-manifest.json`.

## Historical sources and artifacts

- `introduction.tex` states the paper motivation, sparsity-spillover question,
  proposed contribution structure, and architecture-wide research direction.
- `methodology.tex` gives the formal Pythia graph, sites, activation operators,
  L1/OL1 definitions, exact product counters, `R_block`, `R_model`, and the
  draft architecture ceiling `R_model_max`.
- `experiment-control/README.md` records the paper-intent, execution, analysis,
  and finding status of the intervention ladder. Its A4-OL1 Pythia-14M cell is
  completed by corrective Run 015 because Run 012 actually applied OL1 only at
  `h`; Finding F001 remains discarded and Run 015 is not yet finding-backed.
  Its A7 cell is backed by Run 013,
  Analysis 008, and Finding F002; its completed A7-OL1
  cell is backed by Run 014 and reported descriptively through Analysis 008.
  Its selected Pythia-70M A0, A1-H, A4-OL1, and A7-OL1 cells are complete in
  Run 018 and compared with 14M in Analysis 010; 410M remains unobserved.
- `reports/01-2026-08-30-status-update/` contains the user-requested TeX source,
  PDF, and report-local provenance notes for Status Report Number 1. This
  numbered report is an explicit tracked exception to the directory-wide
  ignore rule; LaTeX auxiliary files remain untracked.
- `reports/02-2026-09-01-status-update/` contains Status Report Number 2's
  generated TeX, compiled PDF, focused verifier, current Pod-ID-reconciled
  billing provenance, and report-local notes. It carries the verified selected
  70M promotion through Analysis 010 without promoting a finding.
- `artifacts/pythia-architecture-map.tex` and its compiled PDF provide the
  intervention-neutral, paper-ready map of one shared Pythia block; the
  adjacent Markdown file records its notation, scope, caption, and caveats.
- `artifacts/sparsification-ladder.tex` and its compiled PDF define the
  paper-facing eight-step intervention ladder, visually separating gate sites,
  gate families, pressure methods, and topology-conditioned architecture
  ceilings; the adjacent Markdown file records its terminology and assumptions.
- `artifacts/pythia-architecture-sparsification-ladder.tex` and its compiled PDF
  combine the architecture map above the intervention ladder as a single
  publication-ready vector figure; the adjacent Markdown file records its
  layout, caption, sources, and scope.

The methodology records upstream provenance in its header. The lean handoff was
separately distilled from `orthogonal-sparsity-pressure` commit
`a5171e2c13d279ec97bdf7b7ef77139fa776e149`. Preserve both identities when a
definition is reconciled.

## Earlier research direction

1. Establish whether pressure at targeted FFN/branch sites produces a
   systematic opposing response at untargeted attention sites: *sparsity
   spillover*.
2. Measure the full executed graph with exact site counters and actual
   zero-operand product counters rather than treating local sparsity as model
   compute.
3. Validate architecture- and sequence-dependent logical reach ceilings for
   candidate intervention topologies.
4. Test whether architecture-wide gates plus conflict-aware pressure improve a
   quality--logical-compute frontier relative to local pressure alone.
5. Keep logical opportunity, sparse-kernel realization, and measured speedup as
   separate claims.

These are directions and hypotheses. Their evidence status lives in run and
analysis observations and in user-approved findings.

## Authority and evidence

Use this order when answering different questions:

1. An immutable run config, code identity, and artifacts say what executed.
2. Tested shared code and compact `research/` contracts say how the current
   workflow operationalizes a method or metric.
3. The manuscript states the current scientific framing and formal proposal.
4. Observations record measured patterns; user-approved findings determine
   which empirical statements may be promoted into the paper.

None of these layers silently rewrites another. Record a mismatch and decide it
explicitly.

The compact list of currently known manuscript/operational differences is in
[`research/MANUSCRIPT.md`](../research/MANUSCRIPT.md).

## Adding future TeX

Add a `.tex` section or generated table only when it serves a current paper
need. Every result-bearing fragment must identify its source run/analysis,
observation or finding, generating script when applicable, coverage, and whether
the file is generated or hand-edited.

Figures and numerical reduction scripts remain owned by their numbered run or
analysis. The manuscript may reference those PDFs and may contain a generated
table fragment, but it does not become a second results store.

Do not build a generalized manuscript pipeline yet. Add a parent document,
bibliography, build command, or generated-fragment directory only when the paper
actually needs it.

## Result evidence crosswalk

- On 11 September 2026, the user approved Analysis 021's frozen kernel v2
  figure and a rewrite around where sparsity becomes useful computation.
  Draft Section 4.6 and Figure 7 now separate native-relative speedup from
  matched projection-skipping gain; Discussion follows the same argument.
  Appendix D.4 retains search/compatibility evidence and D.5 adds the kernel
  predicates, stratified fits and all 30 native/control/optimized latencies.
  [O010](../analyses/021-2026-09-10-training-results-figures/observations/O010-kernel-manuscript-adoption.md)
  records sources, qualifications and verification. No new measurement was made.

- On 8 September 2026, the user requested an explicit attention-skipping
  discussion in the new kernel subsection. It now reports the executed QK/PV
  skips and their negative runtime contribution at T=2048. Longer sequences
  are discussed only as an untested hypothesis: both attention work and
  recurring skip overhead grow, and the current fixed-length kernel requires
  adaptation and fresh qualification. Run029 Observation03 supplies the
  unchanged counters/ablations; its manuscript revision-2 provenance retains
  the updated source snapshots. No new benchmark or finding is added.

- On 8 September 2026, the user requested an implementation/claim audit and
  conditional manuscript integration of Run029. The audit independently
  reconciles all 1,173 outcomes and 522,816 timing pairs and confirms the
  scoped Pythia-14M/RTX5090 result. `draft/kernel-autoresearch.tex`, included
  by the experimental section, presents the existing two-panel PDF;
  `draft/kernel-implementation.md` supplies an actual CUDA excerpt, source
  map, numerical contract, ablations, a non-cohort gate-threshold bug and
  model-identity limitations. The text claims a positive association and
  realizable sparse-path contribution, not universal monotonic scaling,
  equal-quality gains, or autonomous/model-specific agent superiority.
  Run029 Observation06 and `19_audit_evidence.py` own the audit evidence.
  The draft was local-only at that point; run-owned provenance snapshots
  retain the requested new section and companion without changing that policy.

- On 4 September 2026, the user requested the full manuscript rewrite using
  current evidence and adversarial-review-v3. Analysis 013 reconstructs the
  full 35-condition 14M study, 12 selected conditions each at 70M and 410M,
  all 60 clipping points, Run 021's learning-rate screen, and six completed
  Run 023 execution sentinels. The draft reports dose-dependent paired
  effects, common-loss operating points, architectural denominator effects,
  and the runtime limitation. The pressured A4/A7 comparison remains a
  combined gate-and-pressure intervention. The 410M results remain visible
  in the appendix with a main-text disclosure. This is authorized descriptive
  integration; the finding registry and immutable runs remain unchanged.

- On 1 September 2026, the user requested Status Report Number 2 after Run 018
  and Analysis 010 completed. The report retains Report 1's architecture
  diagram and 14M result structure, updates the execution scope and RunPod
  billing, adds matched 14M/70M A4-OL1 and A7-OL1 tables with count-pooled
  exact-zero masses, reports all ten 70M A0/A1-H post-hoc TEAL targets, and
  embeds Analysis 010's single validation-loss--`R_model` frontier. The result
  is descriptive one-seed evidence across two sizes; no scale-independent
  claim, runtime-speedup claim, or finding is promoted, and 410M is unobserved.

- On 31 August 2026, the user approved tentative Finding F002 from the five
  matched Pythia-14M A4/A7 full-pass pairs. Adding symmetric post-RoPE Q/K/V
  gates is a near-null topology expansion at `kappa=0`, improves both
  validation loss and measured `R_model` at `kappa=0.01`, and yields increasing
  logical opportunity with increasing validation-loss cost at larger
  thresholds. The finding is limited to one seed, one 14M scale, one MiniPile
  pass, and logical product opportunity rather than measured speedup. Analysis
  008 owns the expanded frontier, count-pooled eight-site table, and
  machine-readable reduction; Runs 011 and 013 own the verified endpoints.
- On 31 August 2026, Finding F001 was discarded after implementation audit.
  Run 012 declared four-site pressure but reused a training capture fixed to
  `h`; its endpoints are A4-Z + OL1@h, not A4-OL1. Analysis 007 remains a
  historical reduction of those endpoints. Corrective Run 015 subsequently
  completed all five matched four-site conditions with a verified 24-tensor
  pressure identity. Its run-local observation finds paired loss/`R_model`
  improvement at `kappa=0` and `0.01`, followed by increasing logical
  opportunity at increasing loss for larger thresholds. Existing F001-backed
  result prose in `introduction.tex` remains known stale: Run 015 does not
  restore F001 or authorize TeX changes without a new analysis and explicit
  user approval.
- On 30 August 2026, the user approved reporting Analysis 005's uniform
  TEAL-style post-hoc GeLU/ReLU controls as a strong descriptive result in
  `reports/01-2026-08-30-status-update/status-update.tex`. The report references
  Analysis 005 Observations O001 and O002, includes the analysis-owned combined
  trained/post-hoc PDF, and retains the one-seed, intervention-semantics, and
  logical-opportunity caveats. No centralized finding was promoted. Analysis
  005's tracked observations and figures remain the numerical evidence record.
- On 30 August 2026, the user approved updating Status Report Number 1 with
  what was then interpreted as Analysis 007's paper-scale A4-OL1 result.
  The report replaces the prior combined trained/post-hoc plot with Analysis
  007's augmented frontier PDF, replaces the incomplete pilot-only A4-OL1
  subsection with all five Run 012 endpoints, and links Observation O001. It
  retains the one-seed, joint-intervention, logical-opportunity, and
  no-runtime-speedup caveats. The report update preceded the later approval of
  tentative Finding F001. The 31 August implementation audit supersedes that
  interpretation: the report's Run 012 labels are stale, F001 is discarded,
  and Analysis 007 retains numerical provenance only for A4-Z + OL1@h.
- On 31 August 2026, the user approved updating Status Report Number 1 with
  Run 014's completed A7-OL1 cohort. The report now contains the ten-row matched
  A7/A7-OL1 count-pooled table and embeds Analysis 008's corrected single-panel
  frontier with plain OL1 labels. A7-OL1 is marked complete at the report
  cutoff, and Run 014's settled cost is included. The A7/A7-OL1 comparison
  remains descriptive and is not added to Finding F002. The report's README,
  TeX source, and compiled PDF are explicitly tracked for this requested
  update even though `manuscript/reports/` stays ignored by default.
