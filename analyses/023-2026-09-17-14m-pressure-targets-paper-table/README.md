# Analysis 023 — 14M pressure-target table for the paper

The user approved these data for paper use on 17 September 2026 and requested
a new analysis folder so the results can be found and reproduced later.

**Start with [TABLE.md](TABLE.md): the exact 30 endpoints delivered in chat.**
It contains `(S_model %, validation loss)` for all five thresholds and all six
recipes: A4, A4+OL1(all), A4+OL1(h), A7, A7+OL1(all), and A7+OL1(h).
[results.json](results.json) preserves full precision, integer counts, both
validation measurements, realized pressure targets, checkpoint identities,
source paths, and SHA-256 hashes.

## Loss-provenance correction

The earlier response and Run 032 figure metadata described every loss as
ordinary final validation. The source audit found a narrower distinction:
the four older overview curves use loss from the final checkpoint's eager
logical diagnostic pass; the two h-only curves use ordinary final validation.
Both cover the same complete validation data and final checkpoint, but they
use different attention/batch evaluation paths. Across all 30 endpoints the
largest absolute difference is **0.000105806356 nats**. Sparsity values are
unchanged. This is a reporting correction, not a training or data-recovery error.

`TABLE.md` intentionally preserves the exact approved numbers. Two clearly
labeled alternatives avoid mixing evaluation passes when preparing the paper:

- [TABLE-ordinary-final.md](TABLE-ordinary-final.md): ordinary final validation
  for every column, consistent with the Run 032 cohort verifier.
- [TABLE-logical-pass.md](TABLE-logical-pass.md): loss and sparsity from the
  same logical diagnostic pass for every column, consistent with the older
  paper overview's convention.

No alternative is silently substituted for the approved table. The historical
figure and its source snapshot remain unchanged; the correction is also linked
from Run 032. Manuscript insertion remains a separate editorial step.

## Sources and intervention identity

| Table column | Source run | Realized pressure sites |
|---|---|---|
| A4 | 011 | none |
| A4+OL1(all) | 015 | a,m,h,z |
| A4+OL1(h) | 012 | h; original all-site declaration was incorrect |
| A7 | 013 | none |
| A7+OL1(all) | 014 | a,m,h,q_post,k_post,v,z |
| A7+OL1(h) | 032 | h |

A4 gates a,m,h,z with the one-sided threshold; A7 adds symmetric post-RoPE
q_post,k_post and v thresholds. Pressure is OL1 at lambda = 1 and b = 1.
All five kappa values are retained: 0, 0.01, 0.05, 0.1, 0.5.
Analysis 009's audited source-code realization establishes Run 012 as h-only;
its old verification label alone is not evidence of four-site pressure.

All 30 conditions share seed 1234, initialization SHA-256
`ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57`,
training-order SHA-256
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`,
712 updates, and 1,493,172,224 training input tokens. Evaluation covers
500 MiniPile validation documents, 338 complete 2,048-token blocks and
692,224 input tokens, with the 1,444-token excluded tail recorded.
The script pools integer zero-product and model-product counts before division.
`S_model (%) = 100 * R_model` is a logical opportunity, not measured speedup.

## Result and paper connection

See [observation 001](observations/001-pressure-target-table.md) for the
descriptive result and limits. This evidence extends the paper's
`manuscript/draft/training-results.tex` sections on quality–sparsity and
the paired effect of adding pressure by making the h-only target set explicit.
The existing overview is retained at
[`runs/032-2026-09-16-pythia14m-a7-h-only-ol1/figures/01-14m-quality-sparsity-h-only.pdf`](../../runs/032-2026-09-16-pythia14m-a7-h-only-ol1/figures/01-14m-quality-sparsity-h-only.pdf).
Analysis 022 separately retains the complete A7 three-way comparison.

## Reproduction and verification

From the repository root:

```powershell
.venv/Scripts/python.exe analyses/023-2026-09-17-14m-pressure-targets-paper-table/01_build_tables.py
.venv/Scripts/python.exe analyses/023-2026-09-17-14m-pressure-targets-paper-table/02_estimate_70m.py
```

The table builder checks all 30 source metrics/manifests/logical diagnostics
against their cohort verifiers, the exact previous figure snapshot, pooled
counts, the five-threshold grid, complete validation, initialization and
order identities, pressure captures, and the Run 012 realization audit.
It retains source hashes without copying model weights or datasets.
Verification on 17 September passed: all 30 reported cells exactly reproduce
the approved figure snapshot; all three tables contain the complete grid;
all recorded source hashes and README links resolve; rerunning both scripts
reproduces all six generated files byte-for-byte.
The analysis number is 023 because folder 022 already exists; the old
`research/INDEX.md` next-analysis counter still said 022 and is corrected to 024.

## Requested 70M ETC

**Frozen/deferred by the user on 17 September 2026.**
See [the freeze record](70M-FROZEN.md); resume only on a new user instruction.

[70M-ETC.md](70M-ETC.md) and [70m-etc.json](70m-etc.json) retain a separate
provisional estimate for A4-OL1(h) and A7-OL1(h), each at kappa 0.05 and 0.5.
It uses actual Run 018 H200 wall times and its five-boundary timing preflight.
Four independent H200s imply **115–160 minutes elapsed**, including a
20–45 minute setup/preflight/retrieval allowance under normal conditions.
Contention was material in the old cohort and can extend this range.
This is a historical timing estimate, not a new run, live price quote, or
authorization to create resources.

The subsequent [70M cost estimate](70M-COST.md) uses RunPod's public H200
price checked on 17 September: USD4.59/GPU-hour. Four parallel conditions
are approximately USD35–50 in incremental usage under the normal ETC,
with a USD65–70 planning allowance for slower workers and retrieval.

## Retained kernel measurements

The requested final-K050 benchmark extension is complete in
[Analysis024](../024-2026-09-17-h-only-kernel-latency/README.md). It retains raw
provenance, a complete 40-checkpoint latency/speedup table and clean PDF figures,
including both h-only families. Only the five new A7 h-only models were timed;
the historical A4 h-only measurements were reused. This does not resume 70M.
