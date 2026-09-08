# Paper summary: kernel auto-research and sparsity-dependent acceleration

## Question and requested presentation

Can one figure connect matched kernel-development progress to the final
historical kernel's relationship between logical sparsity and full-model
speedup? On 8 September 2026 the user requested a one-row, two-column paper
figure, ordinary points in place of crosses, no progress annotations or
variant legend, percentage R_model ticks, and a black dashed regression.
This is a display-only reuse of the completed Run029 evaluation: no new
timings, numerical qualifications, candidate selections or regression fits.

## Method and coverage

The left panel retains all 214 timed history comparisons: 140 qualified and
74 numerically unqualified, displayed identically as gray circles. Two
unsupported combinations remain untimed and unplotted. Each point pools
three fresh processes for one candidate configuration and checkpoint.
Four fixed checkpoints and all 53 eligible historical configurations are
retained; K011 mask configurations share an iteration. Eligible proposals
are numbered 1-42; the separately benchmarked P0 adapter occupies a reference
position at zero and is not counted as a search iteration. No points are
jittered, rescaled, deleted for being slow, or replaced by bars.

The blue step line is unchanged: the cumulative best numerically qualified
speedup on fixed checkpoint c30, with native execution as the 1x pre-search
incumbent. The line is not the maximum over all plotted checkpoints; giving
failures ordinary markers does not make them eligible to update it.

The right panel retains all 35 qualified K050 checkpoint comparisons and
the original unweighted, estimated-intercept OLS. Canonical R_model remains
the retained FP16 count-pooled logical-product fraction. Only its tick
display changes to percentages; the equation still takes fractional R_model
(for example, 10% means 0.10). The speedup axis remains linear and cropped
to the observed/fitted range, without the empty 0-1 region. The entire fitted
line, including the endpoint above the largest observed speedup, is visible.

Both panels use the same native PyTorch/SDPA CUDA-graph timing denominator,
one physical RTX5090, BF16 B1/T2048 uncached causal execution, and the full
50,304-logit output. Each point pools 1,344 paired timing ratios from 64 fixed
validation inputs, seven paired passes, and three fresh processes. Numerical
qualification uses all 338 complete blocks of the 500-document validation
split; the 1,444-token tail is excluded and recorded. Existing checkpoints
share training seed 1234; no model is retrained for this figure.

## Figure and paper caption

Current combined asset:
[04-kernel-autoresearch-and-rmodel.pdf](../figures/04-kernel-autoresearch-and-rmodel.pdf)
(byte-identical to retained revision3).

**Caption:** Matched retrospective kernel auto-research and final-kernel
acceleration for Pythia-14M on an RTX5090. (a) Gray circles show all 214 timed
candidate/checkpoint comparisons, including 74 that fail numerical
qualification; these timings are not all valid accelerations. The blue line
tracks the best qualified speedup on a fixed checkpoint, initialized at the
native 1x reference. Two unsupported combinations have no timings. P0 is a
separate pre-search comparator, not an iteration. (b) The best historical
kernel, K050 at final iteration 42, passes complete numerical qualification
on all 35 checkpoints. Each point is a geometric mean of native/candidate
CUDA-graph timing ratios across three fresh processes. The black dashed line
is unweighted OLS with an estimated intercept; the equation uses fractional
R_model, displayed as percentages on the axis. The positive cross-checkpoint
association is conditional on this model, hardware and workload, not a
causal isolation of sparsity or a quality-matched model comparison.

## Result and interpretation limits

The qualified progress line reaches 1.783029x at iteration 42, from the
unmodified native 1x reference. The final K050 checkpoint sweep gives

    S_hat = 0.9531843003 + 3.8586397669 R_model
    R_squared = 0.7812571630; n = 35; R_model in [0, 1].

K050 is the last eligible historical proposal and the best historical
candidate under the fixed-c30 progress criterion. "Best" does not include
the later diagnostic ablations: disabling attention skipping is slightly
faster, as documented in [Observation03](03-matched-rmodel-speedup.md).
The final-sweep c30 speedup, 1.783175x, is a separate matched measurement
from the history score, not a forced match or rescaled endpoint.

The figure supports successful specialization and a positive sparsity-speedup
association. It does not attribute the entire gain to sparse multiplication;
fusion also contributes, and individual sparse paths can incur overhead.
Checkpoints differ in topology, weights, sparsity and loss. This is one
training seed, architecture and GPU, without a held-out kernel-selection
test. Removing failure-specific markers is solely a visual simplification:
the caption and underlying records preserve their status. No manuscript
TeX or centralized finding is changed.

## Source, reproduction and verification

- Frozen data: `results/matched-retrospective-001.json`, identity checked
  against the prior `results/figures-002.json` receipt before plotting.
- Iteration map: `provenance/candidates.json`.
- Generator: `18_paper_summary.py --revision 2`.
- Figure/data/catalog/script identities: `results/figures-summary-002.json`.
- Tests: `tests/test_paper_summary.py` verifies the 1x2 layout, labels,
  regular markers and all 214 coordinates, unchanged qualification-aware
  progress, all 35 final points, percentage formatting without fit rescaling,
  independently recomputed OLS coefficients, and the black dashed fit.

For a new reproduction, use a fresh `--revision` value (3 or higher here);
the generator refuses to overwrite an existing figure or manifest. No GPU
is needed. Existing single-panel PDFs and their observations remain intact.
The initial combined proof is retained but superseded by revision2, which
embeds TrueType fonts, aligns panel titles, and uses five-percentage-point
ticks. Neither revision changes measurements or numerical qualification.

Revision2 is one 511.2 x 234 point page (7.1 x 3.25 inches). Poppler confirms
embedded, subset Unicode CID TrueType fonts and no raster images. The full
page was rendered at 160 dpi and visually checked for label legibility,
alignment, clipping and fit visibility. Focused and full-suite test results
are recorded in the README's paper-summary closeout.

### Revision3: single-line panel titles

The user subsequently requested the exact unsuffixed PDF filename with the
right title shortened to `(b) Best kernel (final iteration 42)`. Revision3
removes the second title line and the left title's matching blank line;
measurements, points, fit and remaining styling are unchanged from revision2.
`18_paper_summary.py --revision 3` generates the retained `*-r03.pdf` and
`results/figures-summary-003.json`. The unsuffixed PDF is a byte-identical
published copy, SHA-256
`9f65a78bfe68f96cab04aae9a44349ad460c682584fb91745b44e15ffc39356b`.

Before replacing the requested file, its original proof bytes were copied
and hash-verified as `figures/04-kernel-autoresearch-and-rmodel-r01.pdf`.
The old `figures-summary-001.json` figure identity now refers to that archived
proof (same bytes and SHA-256, different filename), not to the published copy.
Revision2 and its receipt remain unchanged; historical generator versions
remain available in Git. Future reproductions use a fresh revision number
(4 or higher) and do not automatically replace the unsuffixed published copy.

All three focused tests pass (0.43s), including exact one-line titles and
unchanged measured coordinates/OLS. The single page was rendered and visually
checked; Poppler confirms embedded Unicode TrueType fonts and no raster images.

### User-authorized manuscript use after audit

The subsequent implementation/claim audit is recorded in Observation06.
The unchanged PDF is now included by `manuscript/draft/kernel-autoresearch.tex`,
with qualification, units and causal limits retained in its caption and text.
The companion `kernel-implementation.md` contains the exact source excerpt,
implementation contract and audit limitations. This scoped integration was
requested by the user; it does not promote a centralized research finding.
