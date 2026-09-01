# Pythia-70M selected-ladder result

## Question

Does the validation-loss versus measured-`R_model` tradeoff from the selected
Pythia-14M controls and OL1 ladders persist when promoted to randomly
initialized Pythia-70M?

## Method and coverage

Twelve independent Secure H200 runs used the same hash-pinned random
initialization and one-pass schedule: A0, A1-H, and A4-OL1/A7-OL1 at kappa 0,
0.01, 0.05, 0.1, and 0.5. Each canonical attempt completed 712 optimizer
boundaries and 1,493,172,224 training tokens with no FP16 overflow or skipped
update. Final evaluation reloaded the retained checkpoint and covered all 338
complete 2,048-token validation blocks (692,224 input tokens); the declared
1,444-token tail was excluded. Logical-product and per-site activation
fractions were pooled from integer counts over the complete workload.

| Condition | Validation loss | `R_model` (%) | Conflicts | Projections |
|---|---:|---:|---:|---:|
| A0 GELU | 4.099767 | 0.0005 | n/a | n/a |
| A1-H ReLU | 4.222745 | 10.0658 | n/a | n/a |
| A4-OL1 kappa 0 | 4.805289 | 25.6725 | 351 | 383 |
| A4-OL1 kappa 0.01 | 4.822732 | 26.2242 | 355 | 332 |
| A4-OL1 kappa 0.05 | 4.873932 | 28.4900 | 388 | 710 |
| A4-OL1 kappa 0.1 | 4.943729 | 30.5743 | 391 | 710 |
| A4-OL1 kappa 0.5 | 5.389543 | 35.5962 | 475 | 709 |
| A7-OL1 kappa 0 | 4.941204 | 23.5624 | 375 | 711 |
| A7-OL1 kappa 0.01 | 4.936382 | 24.2263 | 373 | 711 |
| A7-OL1 kappa 0.05 | 4.963642 | 26.5239 | 390 | 711 |
| A7-OL1 kappa 0.1 | 4.950103 | 28.6938 | 397 | 710 |
| A7-OL1 kappa 0.5 | 5.215976 | 40.6019 | 375 | 710 |

The complete per-site exact/near-zero, RMS/L2, layer-norm,
logical-opportunity, OL1-boundary, and checkpoint records are retained in the
canonical attempts. A0 and A1-H also have complete ten-point post-hoc TEAL
frontiers over target sparsities 0.0 through 0.9.

## Result

The 70M trained ladder shows a clear loss/opportunity tradeoff rather than a
single intervention ordering. A4 dominates A7 at kappa 0. A7 overtakes A4 at
kappa 0.5, reaching 40.602% `R_model` at loss 5.2160 versus A4's 35.596% at
loss 5.3895. A7 kappa 0.1 is a favorable local knee: relative to A7 kappa 0,
it gains 5.131 percentage points of `R_model` for 0.00890 additional loss.

The corresponding 14M crossover and the control TEAL frontiers are compared in
Analysis 010. That comparison supports descriptive persistence of the selected
tradeoff from 14M to 70M, while showing that absolute `R_model` changes with
scale.

## Infrastructure and verification notes

One original A7 kappa-0.1 Pod failed before scientific execution because SCP
was unusable. A replacement in EUR-IS-4 was retired after a topology-matched
unchanged worker showed severe regional contention; its partial records are
explicitly marked non-evidence. A non-EUR replacement completed the canonical
attempt. Remainder smokes used topology-equivalent kappa-0 A4/A7 sentinels
because `02_smoke.py` accepts only the sentinel identifiers; exact thresholds
were covered by config tests, construction checks, and the completed runs.

The first four and last eight attempts recorded two run-code inventory hashes.
Verification established that their inventories were byte-identical except for
LF versus CRLF encoding of `prelaunch/initialization/metadata.json`; parsed JSON
and every code file were identical. The cohort verifier accepts only those two
known hashes, normalizes to the LF identity, and rejects any other divergence.

## Caveats

There is one seed, one 70M scale, and one MiniPile pass. `R_model` and
`R_model_max` are logical-product opportunities, not measured runtime gains.
TEAL is post-hoc clipping rather than gated training. These data do not yet
establish persistence at 410M or a scaling law.

## Sources

- Canonical attempts: `artifacts/attempts/001-*` through `012-*`.
- Cohort verification: `artifacts/verification.json`.
- TEAL frontiers: `artifacts/teal/teal_frontiers.json`.
- Execution/retrieval evidence: `launch-control/sentinel-20260901/` and
  `launch-control/remainder-20260901/`.
- Cross-scale reduction and figure: Analysis 010.
