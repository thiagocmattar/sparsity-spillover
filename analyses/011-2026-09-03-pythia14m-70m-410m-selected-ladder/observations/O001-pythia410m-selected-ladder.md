# O001 - Pythia-410M selected-ladder frontier

## Question

Where do the verified Pythia-410M trained A4-OL1/A7-OL1 ladders lie relative
to the A0/A1-H post-hoc TEAL controls on paired validation loss versus measured
`R_model`?

## Method and coverage

Source: verified Run 019 attempt diagnostics and consolidated TEAL artifact.
Every point evaluates all 338 complete 2,048-token validation blocks from 500
MiniPile documents (692,224 input tokens); the 1,444-token tail is excluded and
reported. Trained endpoints pair loss and `R_model` from the same eager logical
diagnostic. TEAL uses evaluation-only clipping at `a,m,h,z` over ten target
sparsities. Counts are pooled before division.

## Figure caption and legend

Pythia-410M validation loss versus measured model-wide logical zero-product
opportunity. Magenta X and black downward-triangle curves are post-hoc clipping
on A0 and A1-H. Purple hexagons and blue pentagons are trained A4-OL1 and
A7-OL1 ladders. Lines connect target/dose order only. Points above loss 6 are
retained in the data and their trajectories visibly exit the clipped panel.

## Result

A0 begins at paired loss 4.547456 and `R_model=0.0110%`; A1-H begins at
4.651294 and 21.0497%. At `kappa=0.5`, A4-OL1 reaches 71.5914% at loss
5.190966, while A7-OL1 reaches 80.6155% at the lower loss 5.120692. At 410M,
A7-OL1 also dominates A4-OL1 at `kappa=0`, so A4's smaller topology is not
preferred at the zero-threshold endpoint on either plotted axis.

## Caveats and nonclaims

The result is one-seed descriptive evidence. TEAL points are post-hoc and are
not trained models. `R_model` is a logical opportunity count, not speedup.
Lines are not fitted response curves, and the y-axis clipping is disclosed.

Source script: `01_build.py`.

Output: `figures/01-pythia410m-selected-ladder.pdf`.
