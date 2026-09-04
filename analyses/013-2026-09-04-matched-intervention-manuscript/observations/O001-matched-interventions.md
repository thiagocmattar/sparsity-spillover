# O001 - Gate location and pressure scope in matched 14M comparisons

## Question and method

How does validation quality and operation-weighted opportunity change when
we vary the gate set, pressure sites, or complete pressure-update recipe?
Read all 30 Analysis 008 endpoint identities and five Analysis 009 realized
h-only conditions from their immutable logical-product and activation files.
Pair loss with counts from the same eager pass. Hold the realized seed,
initialization, data order, schedule, and training budget fixed.

Coverage: 35 checkpoints; each covers all 338 complete validation sequences
(692,224 input tokens), excluding the documented 1,444-token tail. Fractions
are rebuilt from pooled integer counts.

## Figure and caption

[Figure PDF](../figures/01-matched-14m.pdf).
Left: ReLU with ordinary or task-aware L1 pressure. Middle: fixed A4 gates
with no pressure, h-only pressure, or four-site pressure. Right: A4/A7 with
and without pressure at their respective sites. Lines order the evaluated
doses. Smaller loss and larger opportunity are favorable. Local dose is
lambda = .05, .1, .5, 1; A4/A7 dose is kappa = 0, .01, .05, .1, .5.

## Result

- Local OL1 and L1 yield similar opportunity; their quality ordering changes
  at the largest coefficient. Neither recipe consistently dominates.
- Expanding pressure from h to all four A4 sites raises loss at every
  threshold. At .01 opportunity stays effectively fixed (8.5858% to 8.5867%)
  while loss rises from 5.2045 to 5.4583.
- At kappa=.5, adding Q/K/V gates without pressure adds 5.171 pp opportunity
  with +.0432 loss. The zero-threshold identity comparison is -.0021 loss
  and +.006 pp opportunity.
- At .5, pressure adds 12.096 pp / +.1265 loss under A7, versus 2.498 pp /
  +.3783 under A4. The pressure-effect difference is +9.598 pp / -.2518 loss.
  Its sign and magnitude depend on dose; the complete tables expose this.

## Caveats and provenance

Each dose is a within-seed matched contrast. Doses are correlated treatment
levels; seed variability is unmeasured. A7 pressure includes Q/K/V whereas
A4 pressure does not. The difference between these effects compares two
configuration-specific recipes. A7 gates with four-site pressure remain
the missing fixed-pressure condition. The high-threshold pattern is
exploratory, identified after observing the grid.

The original Run 012 declared wider pressure but executed h-only capture.
Analysis 009 audited this realization. It is explicitly labeled A4+OL1@h;
the immutable run metadata is preserved.

Source scripts: [evidence.py](../evidence.py), [plots.py](../plots.py), and
[01_build.py](../01_build.py). Sources and hashes: [figure_data.json](../figure_data.json).
Tables: paired-effects.tex, pressure-differences.tex, trained-14M.tex.
