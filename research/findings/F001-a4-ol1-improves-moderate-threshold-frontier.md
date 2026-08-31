# F001 - Discarded four-site A4-OL1 claim

Status: discarded

Approved: 2026-08-30

Discarded: 2026-08-31

## Correction

This finding is withdrawn. Run 012 declared OL1 pressure at `a,m,h,z`, but its
training loop reused Run 004's hard-coded `ActivationCapture(model, ["h"])`.
The realized objective therefore contained only the six `h.layer_*` tensors;
the config, manifest, and original verifier did not prove otherwise. The
numbers below remain a valid description of matched Run 011 A4-Z versus the
historical **Run 012 A4-Z + OL1@h** intervention, but they do not test the
four-site statement that was approved.

Run 015 is the separately numbered correction. This finding stays discarded
unless corrected evidence is completed, reconciled in a new analysis, and
explicitly approved as a new or restored finding.

## Statement

The original, now unsupported statement was that for one matched seed and one
full MiniPile pretraining pass at Pythia-14M, adding four-site OL1 pressure
(`lambda=1`, `step_budget=1`) to the operational
`A4-Z = {a,m,h,z}` one-sided threshold topology lowers complete-validation loss
and increases measured `R_model` at each matched `kappa` in
`{0, 0.01, 0.05, 0.1}`. The advantage contracts as `kappa` increases and does
not extend to `kappa=0.5`: there, validation loss is 0.062986 higher than A4
while `R_model` increases by only 0.0119 percentage points.

Because the pressure-set premise was false, this is not evidence that
four-site conflict-aware pressure improves the A4 frontier.

## Evidence

The historical table covers five matched A4 versus A4-Z + OL1@h pairs. Each condition trained from
the same random initialization and realized data order for 712 optimizer
boundaries. Every reported endpoint uses all 338 complete 2,048-token blocks
from all 500 MiniPile validation documents (692,224 input tokens); the
1,444-token incomplete tail is excluded and reported.

| `kappa` | A4 loss | A4 `R_model` | historical A4-Z + OL1@h loss | historical A4-Z + OL1@h `R_model` | pressure loss delta | pressure `R_model` delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.470497 | 7.2120% | 5.215749 | 8.4094% | -0.254749 | +1.1974 pp |
| 0.01 | 5.466500 | 7.4137% | 5.204504 | 8.5858% | -0.261996 | +1.1721 pp |
| 0.05 | 5.434110 | 8.2059% | 5.195590 | 9.0356% | -0.238520 | +0.8297 pp |
| 0.1 | 5.419642 | 8.9530% | 5.228687 | 9.3435% | -0.190955 | +0.3905 pp |
| 0.5 | 5.659680 | 10.2155% | 5.722666 | 10.2274% | +0.062986 | +0.0119 pp |

Within historical Run 012, the lowest validation loss is 5.195590 at `kappa=0.05` and
`R_model=9.0356%`. The `kappa=0.1` endpoint trades a slightly higher loss of
5.228687 for `R_model=9.3435%`. Exact-zero mass rises monotonically at all four
selected sites; the full count-pooled site table is retained by Analysis 007.

## Sources and provenance

- Primary analysis: `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/`
- Primary observation:
  `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/observations/O001-full-pass-frontier-with-a4-ol1.md`
- Generating script:
  `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/01_build.py`
- Machine-readable reduction:
  `analyses/007-2026-08-30-full-pass-frontier-a4-ol1/figure_data.json`
- A4 source: `runs/011-2026-08-30-pythia14m-full-pass-a4z/artifacts/verification.json`
- Historical A4-Z + OL1@h source:
  `runs/012-2026-08-30-pythia14m-full-pass-a4-ol1/artifacts/verification.json`

Analysis 007 recomputes every Run 012 `R_model` from integer logical-product
counts and every selected-site exact-zero fraction from pooled integer
activation counts before division. Run 012's verifier records the matched A4
deltas above and locks the source attempts by hash.

## Definitions

- `A4-Z` activates one-sided threshold gates at `a`, `m`, `h`, and `z`, where
  `z` is the concatenated attention context immediately before `W_o`.
- Run 012 actually adds orthogonal L1 pressure only at post-gate `h`; Run 015
  is designed to add it at all four post-gate sites without changing the other
  matched identities.
- `R_model` is the count-first fraction of model-wide logical scalar products
  with at least one exactly-zero operand, including the dense LM head in its
  denominator. It is an opportunity measure, not removed FLOPs or speedup.

## Caveats

- One seed, one Pythia-14M scale, and one MiniPile pass do not establish
  replication or scaling behavior.
- Run 012 combines four-site gates with `h`-only pressure, so it cannot isolate
  either a four-site pressure effect or a single-site gate effect.
- Separately scheduled GPUs are matched by scientific identities but are not
  bitwise replicas.
- Connected frontier lines describe dose order; they are not fitted response
  curves.
- No sparse kernel or wall-clock acceleration was evaluated.

## What would change the conclusion

The discarded claim can only be reconsidered after an implementation that
proves realized pressure at all four sites completes the matched grid, a new
analysis reconciles its integer counts and coverage, and the user explicitly
approves the resulting statement. Additional seeds or scales cannot repair the
missing intervention in Run 012 itself.

## Manuscript effect

This discarded finding supports no manuscript claim. The existing F001-backed
text in `manuscript/introduction.tex` is stale pending an explicit user-approved
TeX correction; the workflow crosswalk records that state without silently
rewriting result-bearing TeX.
