# Element sparsity versus empty tiles at a/m/h/z

Question: what fraction of elements and aligned 8x16 tiles are exactly zero in
each site under the two complete gate operators?

Sources/generator: verified Run058 Ph and Run060 activation-statistics files,
hashed in `../data/source-identity.json`, reduced by `../01_analyze.py`.
The full wide table is `../results/comparison.md`, with unrounded JSON/CSV and
layerwise CSV beside it. No pressure/gate is applied at a/m in either cohort.

Coverage/reduction: all 500 validation documents and 338 complete 2048-token blocks,
692224 inputs, excluded tail 1444. Pool integer zero/element and empty/total-tile
counts across batches and all six layers. A/M/Z each have 531628032 elements and
4153344 tiles; H has 2126512128 elements and 16613376 tiles. Equal tile sizes make
the empty-tile percentage also the proportion of elements in fully empty tiles.
It is a separate statistic from total exact-zero mass.

Result: a/m have 0 exact zeros and0 empty tiles in all four endpoints. In h/z,
tile gates yield element-zero and empty-tile percentages equal to rounding;
their gaps are less than .000011 percentage points, from incidental zeros inside
admitted tiles. Each post-gate corner correctly certifies tile emptiness; all
corner/position/occupancy counters reconcile with zero certificate mismatches.

At0/.1, the tile operator reduces total h zeros by 35.6171/13.4746pp and z zeros
by 15.5309/21.4174pp, while increasing empty h tiles by 54.4614/40.6120pp and empty
z tiles by 44.9108/50.2527pp. Thus increased tile structure comes with fewer
individual zeros, rather than increased overall element sparsity. Logical
R_model also falls from 4.6301% to 2.9404% and 5.1708% to 4.3653%; those are logical
scalar-product opportunities and do not settle inference performance.

Caption: Sitewise element zeros and empty 8x16 tiles after 712 updates in matched
Pythia-14M T2/Ph cohorts. Columns are a,m,h,z and rows are kappa 0/.1. Solid blue
bars are scalar gates; hatched orange bars are corner gates. Within each panel,
the paired categories show exact-element zeros and fully empty tiles. All panels
share 0-100% axes. Explicit zero labels at a/m are measured values, not N/A sites.

Caveats: the noncorner sign/equality rule changes together with grouping. These
counts do not isolate a pure structure-only intervention and imply no speedup,
weight-load savings, decoding implementation or transfer across model sizes.
One seed; no uncertainty over training seeds. No manuscript claim changed.

Figure: `../figures/03-site-sparsity-and-tiles.pdf`, rendered with Poppler and
visually inspected. Additional 1x16/16x16/32x16 counters remain in raw diagnostics.
