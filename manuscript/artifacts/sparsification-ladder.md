# Sparsification ladder

## Question

How does the proposed manuscript method progress from the stock Pythia block to
architecture-wide mixed thresholding with orthogonal L1 pressure?

## Method and coverage

The table is a design map rather than an empirical result. Each row adds one
non-linearity placement or pressure intervention while retaining the preceding
row's configuration. It covers the seven manuscript-map sites `h`, `m`, `a`,
`z`, `q`, `k`, and `v`. As in `pythia-architecture-map.pdf`, the compact `q`
and `k` labels mean the post-RoPE operands `q_post` and `k_post`.

The ladder IDs deliberately separate a row's complete intervention identity
from the underlying topology ID:

| Step | ID | Non-linearity configuration | Pressure |
| ---: | --- | --- | --- |
| 1 | `A0` | stock GeLU at `h` | none |
| 2 | `A1-H` | ReLU at `h` | none |
| 3 | `A1-H-L1` | ReLU at `h` | naive L1 (`L1N`) at `h` |
| 4 | `A1-H-OL1` | ReLU at `h` | OL1 at `h` |
| 5 | `A4` | one-sided threshold at `h,m,a,z` | none |
| 6 | `A4-OL1` | one-sided threshold at `h,m,a,z` | OL1 at all four active sites |
| 7 | `A7` | one-sided threshold at `h,m,a,z`; symmetric threshold at `q,k,v` | none |
| 8 | `A7-OL1` | the Step 7 mixed non-linearities | OL1 at all seven active sites |

`A4` and `A7` are the ladder's compact paper-facing names for the operational
registry topologies `A4-Z` and `A7-Z-POST`. The pressure suffix is not a
topology: non-linearity sites, pressure method, and pressure sites remain
independent fields in an executed condition.

The visual uses the operational gate convention from `research/METHODS.md`:
threshold equality survives. Thus `G_{+,kappa}` rejects `x < kappa` and
`G_{+/- ,kappa}` rejects `abs(x) < kappa`. The `<=` notation in the initial
plain-language sketch is treated as an informal description of the rejected
region, not as a change to the repository's exact boundary convention.

## Architecture ceilings

The displayed `R_model^max` values use the count-first architecture definition
in `research/METRICS.md`, a full uncached sequence of `T=2,048`, Pythia
`d_f=4d`, and a dense language-model head. Pressure does not affect this
topology-conditioned ceiling, so pressure-only row pairs repeat the same value.

The user-requested column labels and the exact architecture tuples used are:

| Display label | `(L,d,V)` |
| --- | --- |
| Pythia-14M | `(6,128,50,304)` |
| Pythia-70M | `(6,512,50,304)` |
| Pythia-410M | `(24,1,024,50,304)` |

For A0, A1-H, A4, and A7, the per-block reachable numerators are respectively
`0`, `4d^2`, `12d^2`, and the complete block
`12d^2 + d(T+1)`.

## Visual encoding and caption

Small filled circles use color alone to distinguish stock GeLU, ReLU,
one-sided thresholding, symmetric thresholding, and no non-linearity. Pressure
is encoded as plain `none`, `L1N`, or `OL1` text. The rounded light panel,
sans-serif typography, navy ink, cool-gray rules, and compact header key reuse
the visual system of `pythia-architecture-map.pdf`. The final three columns
report the analytic model-wide ceilings to one decimal place. Detailed
reproducibility inputs remain recorded in the section above rather than in the
PDF.

**Figure caption.** Sparsification ladder from the stock Pythia block (A0) to
mixed architecture-wide thresholding with orthogonal L1 pressure. Rows separate
non-linearity placement from pressure: ReLU first replaces GeLU at the
FFN-hidden site; naive L1 and OL1 are then compared at that topology; one-sided
thresholding expands to `h,m,a,z`; and symmetric post-RoPE thresholding finally
adds `q,k,v`. Circle color encodes the non-linearity at each site, while plain
text encodes the pressure method. `R_model^max` is the topology-conditioned
logical-product ceiling at `T=2,048`, not an observed zero rate or measured
speedup.

## Result and caveats

This artifact defines proposed terminology and intervention structure; it does
not establish that any row improves task quality, logical opportunity, or
runtime. The ceiling values are analytic and do not depend on a checkpoint.
The three model labels use the architecture tuples above. Any future use of
different layers, width, vocabulary, sequence length, attention workload, or LM
head convention requires recomputing the columns.

## Source

- Editable source: `sparsification-ladder.tex`
- Generated artifact: `sparsification-ladder.pdf`
- Companion architecture map: `pythia-architecture-map.tex`
- Scientific contracts: `../../research/DEFINITIONS.md`,
  `../../research/METHODS.md`, `../../research/METRICS.md`, and
  `../methodology.tex`
