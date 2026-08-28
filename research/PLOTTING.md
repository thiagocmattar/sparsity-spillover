# Plotting Standards

Every figure answers a stated research question and is owned by the numbered run
or analysis that created it.

## Required practice

1. Put the plotting script beside the owning run/analysis as a numbered script.
2. Save publication output as PDF only: `figures/NN-short-description.pdf`.
3. Create a matching observation Markdown file containing:
   - the question;
   - exact sources and coverage;
   - numerical reduction/aggregation;
   - full figure caption and legend explanation;
   - observed pattern;
   - uncertainty, nonclaims, and possible confounds;
   - source script and output path.
4. Add the observation to that folder's `observations/INDEX.md`.

If a figure is later referenced by the manuscript, keep the PDF and reduction
script in the owning run/analysis. Record the manuscript section and approved
claim in the observation; do not create a second untracked copy of the result.

## Honest presentation

- Label axes and units; state n, seeds, filters, and validation coverage.
- Do not truncate axes to exaggerate effects. If a detail view is necessary,
  label it explicitly.
- Prefer distributions and uncertainty to means alone.
- Pool counts before fractions and disclose weighting.
- Comparable panels share scales unless the caption explains why not.
- Use colorblind-safe colors plus redundant line/marker encodings.
- Label near-zero thresholds numerically.
- Call `R_block` and `R_model` logical opportunities, never speedups.
- Do not show undefined or incompatible sites as zero; use `N/A`.

One small shared style helper may enter `src/` after two figures need it. Cohort
selection, reduction, and captions remain local to their owning analysis.
