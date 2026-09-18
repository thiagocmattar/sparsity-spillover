# Figure 3: The benefit of broader pressure depends on threshold strength

PDF: [03-pressure-scope-threshold.pdf](../figures/03-pressure-scope-threshold.pdf).
Manuscript placement: [Section 4.2](../../../manuscript/draft/training-results.tex),
`sec:paired-results` (proposed replacement for the pressure-scope portion of
`fig:paired-intervention-effects`).

## Question, method, and coverage

When does broadening pressure from h to all thresholded sites help or hurt?
Twenty exact pairs compare two sizes, two threshold scopes, and five κ values.
Subtract ordinary-final loss and pooled logical sparsity at fixed size/scope/κ.
All pairs share initial-parameter and training-schedule hashes within size.
Each column shares its Y limits across sizes; κ is categorical and explicitly
shown in every panel. Validation covers 338 blocks from 500 documents, excluding
the 1,444-token tail. [Figure 1's observation](013-quality-sparsity-caption.md)
defines the common loss/count protocol.

## Publication caption

**The benefit of broader pressure depends on threshold strength.** Rows show
14M and 70M; columns show changes in validation loss and model-wide sparsity.
Every point is OL1(all) minus OL1(h), with threshold placement and κ fixed;
blue diamonds and orange triangles distinguish 4-site and 7-site thresholding.
Positive loss changes are worse; positive sparsity changes mean a larger fraction
of logical zero-operand products, expressed in percentage points. The horizontal
line is zero. At 70M with 7-site thresholding, κ=.05 gives −0.371 pp and +0.297
nats/token; κ=.5 gives +8.369 pp and −0.121 nats/token. Expanding the target set
also changes the equal-tensor normalization of the pressure objective. Lines
connect contrasts between separately trained settings in increasing κ, not a
training trajectory or independent replicates. One final checkpoint per setting
is used; no independent-seed uncertainty is represented.

## Result and proposed manuscript writing

> Broader pressure has a threshold-dependent trade-off. At 14M it raises loss
> at every tested κ, while the 7-site sparsity increment changes from negative
> at moderate thresholds to +10.819 pp at κ=.5. At 70M, 7-site broad pressure
> similarly gives both less sparsity and higher loss at κ=.05, but gives more
> sparsity and lower loss at κ=.5. This reversal shows that the direction of
> the pressure-scope trade-off depends on the operating point and model size.

## Caveats and provenance

These contrasts broaden pressure jointly to a,m,z and, for T7, q,k,v. They do
not isolate direct Q/K/V pressure or attribute the effect to normalization versus
target selection. Five κ values are five settings, not five seed replications.
The existing manuscript's pending-70M statement is superseded by these retained
measurements; its blanket loss-increase statement must remain scoped to 14M.

Source: [paper_effect_figures.py](../paper_effect_figures.py),
[13_rebuild_paper_figures.py](../13_rebuild_paper_figures.py),
[checkpoint table](../data/paper-checkpoints.json), and
[20 unrounded contrast pairs](../data/paper-derived.json).
Proposed text is retained here for manuscript adoption, not silently inserted.
