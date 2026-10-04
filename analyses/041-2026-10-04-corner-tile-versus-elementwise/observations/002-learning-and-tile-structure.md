# Learning under the temporal tile restriction

Question: does learning continue under the corner gate, and how does empty-tile
structure change during training?

Sources: the verified Run060 events and activation diagnostics at steps
0/32/128/712, with Run058 scalar final-loss references. All source identities
are in `../data/source-identity.json`; numerical curves are in
`../results/summary.json`. Generator: `../01_analyze.py`.

Method/coverage: random Pythia-14M, T2 at h/z, orthogonal L1 at h, matched seed 1234
and 712-update budget. Every plotted loss/activation pass uses all 338 complete
2048-token validation blocks from all 500 documents, excluding the 1444-token
tail. Tile percentages pool integer counts across six layers. Full-validation
loss is measured at 0/1/32/128/256/512/712; no unmeasured points are estimated.
The first optimizer boundary has LR=0 under the matched pre-step schedule, so
step 0 and step 1 losses coincide; neither boundary is an overflow/skipped update.

| kappa | Loss at step 0 -> step712 | H empty tiles: 0 / 32 / 128 / 712 | Z empty tiles: 0 / 32 / 128 / 712 |
|---:|---:|---:|---:|
| 0 | 11.025221 -> 5.553770 | 51.38 / 69.67 / 65.07 / 57.51% | 49.51 / 44.31 / 50.53 / 44.93% |
| .1 | 11.026661 -> 5.596746 | 57.54 / 77.38 / 79.71 / 84.31% | 94.70 / 47.34 / 61.84 / 70.96% |

Result: both models learn substantially and complete all 712 optimizer boundaries
without overflow or skipped updates. Loss still improves from 512 to 712 by
.045679/.036993. Every final h/z layer has surviving activation tiles: layer
emptiness ranges 48.46-61.85%/27.15-61.07% at 0 and67.84-89.07%/45.66-85.68% at .1
for h/z, respectively. No entire h or z layer branch is empty over validation.
This supports viability of learning with the restriction, while Observation 001
documents its quality cost. It does not prove representational redistribution.

Caption: Run060 full-validation learning and aligned 8x16-tile evolution.
Blue solid lines/circles denote kappa 0; orange dashed lines/squares denote .1.
Markers are actual measurements and connecting lines are visual guides.
The left panel's dotted horizontal lines are Run058 final scalar loss references,
not scalar learning trajectories. The other panels show pooled h/z empty tiles
at the four diagnostic checkpoints. Occupancy axes share 0-100% scales.

Caveats: only four occupancy checkpoints, one seed and fixed budget. Surviving
activity and changing counters do not identify where causal computational work
is redistributed. No inference benchmark, tile pressure, scaling result or
manuscript update is included.

Figure: `../figures/02-learning-and-tile-structure.pdf`, rendered with Poppler and
visually inspected. Layerwise counts are in `../results/per-layer.csv`.
