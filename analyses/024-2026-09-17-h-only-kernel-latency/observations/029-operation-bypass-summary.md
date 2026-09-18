# High-threshold instruction bypass and manuscript explanation

Subsequent integration: this artwork is retained unchanged as manuscript
Figure 6, following the logical-operation decomposition in the merged
Section 4.3. The shorter text links additional QK/PV sparsity to the tested
attention implementation's lack of a timing benefit. See the
[integration record](../../../manuscript/draft/reviews/2026-09-18-operation-sparsity-speedup/README.md).
The details below document the original figure generation and adoption.

Figure: [12-operation-bypass-summary.pdf](../figures/12-operation-bypass-summary.pdf).
Counts and provenance: [operation-bypass-summary.json](../data/operation-bypass-summary.json).
Source script: [21_plot_operation_bypass_summary.py](../21_plot_operation_bypass_summary.py).
The complete [Figure 10](../figures/10-operation-bypass.pdf), its 102-setting
reduction and [Observation 027](027-operation-bypass.md) remain unchanged.

## Question and approved scope

The user requested a lean, first-principles rewrite of the manuscript's
"When model-wide sparsity translates to speedup" subsection and a simpler
replacement for its quality/latency figure. The explicitly selected comparison
is T4/Pall versus T7/Pall at kappa=0.5, with (a) 14M and (b) 70M. Each panel
has grouped bars for the six operation families. This is a presentation of
retained counters, not a new experiment or a new definition of S_model.

## Method and coverage

The script selects exactly four trained settings from `operation-bypass.json`:
14M c20/c30 and 70M c06/c11. Each has one checkpoint at step 712 from training
seed 1234. It recomputes every bar as
`100 * bypassed_mmas / (issued_mmas + bypassed_mmas)` from pooled integers,
checks conservation against potential instructions, and verifies the four
underlying diagnostic hashes. Counts cover six layers and all 338 complete
2,048-token validation blocks from all 500 MiniPile documents: 692,224 input
tokens, excluding the 1,444-token tail. The measured operands are BF16.

Both recipes use their declared all-site OL1 pressure and the highest retained
threshold. The kernel is K050 at 14M and the qualified k050-70m-v2 shape port at
70M. The JSON retains checkpoint/weight identities, issued and bypassed counts,
scalar product counts, baseline PV bypass, and source/script/PDF hashes.
There are no interpolated thresholds, unmeasured recipes or error bars across
independent seeds. Figure 08's blue/orange recipe colors and typography are
retained. The manuscript receives a byte-identical copy and the data export.

## Caption and legend

**Skippable work is not the same as saved time.** Matrix instructions bypassed
for T4/Pall (blue) and T7/Pall (orange) at kappa=0.5: (a) 14M and (b) 70M.
Each bar pools bypassed instructions divided by issued plus bypassed
instructions over all layers and complete validation. Projection labels name
their input sites: a feeds QKV, m feeds FFN-up, h feeds FFN-down and z feeds
attention-output. QK and PV label the attention products. Bypass at h/z
includes replacement by a few scalar operations; attention counts include
causal masking and padding. The bars do not measure latency or the fraction
of total model work saved.

## Result and interpretation

| Size / recipe | QKV (%) | FFN-up (%) | FFN-down (%) | Attention-output (%) | QK (%) | PV (%) |
|---|---:|---:|---:|---:|---:|---:|
| 14M T4/Pall | 67.1224 | 54.4910 | 97.4710 | 99.9811 | 0.0000 | 10.4167 |
| 14M T7/Pall | 8.4385 | 0.0000 | 94.4975 | 99.6449 | 57.9952 | 67.2084 |
| 70M T4/Pall | 27.6732 | 3.2573 | 98.4082 | 99.0370 | 0.0000 | 5.1471 |
| 70M T7/Pall | 0.0000 | 0.0000 | 91.4259 | 96.7840 | 8.7119 | 6.8954 |

Both sizes have extensive h/z instruction bypass at these endpoints. The
14M T7 endpoint also has substantial QK/PV bypass, despite the negative
attention-skipping timing control. This is enough to reject the blanket
interpretation that its attention zeros simply lack skippable structure.
It does not establish profitable attention execution. The bars use separate
per-operation denominators and cannot be summed or averaged into model savings.
The T4 PV values equal the same-size base-model bypass: masked/padded work is
included, so these are not entirely intervention-induced skips.

The manuscript's explanation distinguishes three quantities: mathematically
unnecessary products, actually bypassed matrix instructions, and time saved.
Both paths inspect current values. The h/z implementation can decide before
reading the corresponding weights, whereas QK/PV checks already loaded
operands and retains attention normalization and other work. The matched 14M
controls demonstrate that attention skipping is slower with projection
skipping held fixed. The code identifies remaining costs, but the controls
do not isolate how much time each cost contributes. Likewise, the evidence
points to h/z together without separating their individual time savings.
Native-relative speedup also contains fusion benefits. The 70M counters do
not extend the historical 14M timing attribution.

The kernel is not an optimal latency configuration: its qualified
attention-dense control is faster. The text treats that result as an
implementation limitation and leaves earlier avoidance of data movement
unresolved. It does not infer a hardware limit or claim a successful new
attention algorithm.

## Implementation and search provenance

- [K049 h/z code](../../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k049/joint.cu)
  checks activation rows and traverses active reduction groups before loading
  their weights; eligible short rows use the surviving weighted contributions.
- [K035 attention code](../../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k035/sparse_gemm.h)
  performs operand copies before its zero checks. The fixed kernel does not
  enable the zero-query prefix shortcut.
- [Run028 development record](../../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/README.md)
  records K029-K035 zero-query/prefix attempts, K041 non-overlapping-support
  detection and K046 cheaper conditional execution. Qualified variants did
  not establish a reliable attention benefit. Earlier aggressive candidates
  had numerical failures. K044's confounded composition is not used as an
  attention-only negative result; K046 is the corrected comparison.
- [Run029 audit](../../../runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/observations/06-implementation-and-claim-audit.md)
  and Observation 027 preserve the qualified matched timing evidence.

## Manuscript adoption and verification

The user-approved rewrite is in `manuscript/draft/kernel-autoresearch.tex`.
It removes numerical endpoint/regression narration from the main subsection,
replaces the old figure under `fig:kernel-autoresearch`, and explains the
implementation limitation. Appendix references are updated to the new figure
scope; historical numerical tables and association results are retained.
The previous figure PDF remains available as historical artwork but is no
longer included in the manuscript. No experiment or cloud work ran.

All 24 bars match the original detailed plot exactly and retain the four
specified checkpoint identities. Four existing operation-bypass tests pass.
The standalone figure and manuscript layout were rendered and visually checked.
Build, source-preservation and adoption checks are recorded in the
[manuscript revision record](../../../manuscript/draft/reviews/2026-09-18-kernel-explanation/README.md).
