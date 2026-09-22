# Lean appendix results with native-Base deltas

The user requested removal of the 14M post-hoc clipping section, the standalone
14M-only results table, the 410M stress-test figure, and the later-70M session
table/discussion. The shared 14M/70M results and twelve-condition 410M table
remain. Source artifacts are retained; this change removes their manuscript
inclusions, not experimental evidence.

`03_appendix_table.py` reads Analysis030 `data/full-trained-results.json`
(SHA256 eaf1d4d8bdf8c86fa4231c81cec7cc498f20d21f1b2c948ba95352c587925752).
The paired table contains the same 27 shared conditions at each size: Base,
ReLU, and five thresholds for T2/Ph, T4/Ph, T4/Pall, T7/Ph, T7/Pall.
Every non-Base latency uses K050 at 14M or opt073 at 70M. The original-port
latency column is removed. Loss and sparsity values are unchanged before
rounding; validation covers all 338 complete blocks from 500 documents, with
the 1,444-token tail excluded. Timing retains BF16, RTX5090, batch one,
2,048 tokens, full output logits, and the original three-process measurements.

For each model, cells report sparsity, loss (delta loss), and latency
(delta latency in ms). Deltas are computed from unrounded values as value
minus Base. Base itself is shown with dense PyTorch/SDPA latency and zero
deltas. Fixed references come from `data/combined-figure.json`:

| Size | Base loss | Native Base latency (ms) |
| --- | ---: | ---: |
| 14M | 5.208582500028893 | 0.6554423980056099 |
| 70M | 4.099767337889361 | 1.6110477085782202 |

The machine-readable `data/appendix-paired-results.json` records all 54
checkpoint identities, original session identifiers, metrics, deltas, and
references. The user requested removing session-specific prose and the later
reference table from the paper. Provenance remains here: the 70M T2/Ph
kappa=0.5 timing is Run047 while the fixed 70M Base is Run045. Its displayed
delta is therefore -0.396679 ms against 1.611048 ms, not a matched-session
causal effect. No original timing is rescaled or pooled across sessions.

The 410M table and its interpretation remain; the figure reference is replaced
with the table reference. The kernel appendix now links to the paired table
for fixed Base references instead of the removed later-session table.

Verification: all 27 pairs have unique source records; every 70M non-Base
latency equals its qualified candidate measurement. All 54 deltas are
calculated before rounding. A full temporary build has 19 pages, with no
undefined references, duplicate labels, or box warnings. The paired table
and 410M table were visually checked on pages 16--17. Main.pdf was not replaced.
