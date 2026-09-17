# Interim comparison: four completely recovered thresholds

Question: compare A7 gating with no OL1, h-only OL1, and seven-site OL1 under
the approved matched protocol. The high-threshold kappa=0.5 row is unavailable;
this observation does not answer the primary high-threshold hypothesis.

Method and coverage: `02_compare_recovered_four.py` reads the retained Run
013/014 verifications and the four full local Run 032 condition verifications.
It checks model topology, gates, pressure sites, update/token budgets, and
matched initialization/order hashes. Every included h-only checkpoint inventory
is completely local and verified. Each final validation covers 500 documents,
338 complete 2,048-token blocks and a 1,444-token excluded tail.

Legend: tables in `artifacts/recovered-four-comparison.md` and `.json` order
arms as no pressure / h-only / all-seven pressure. Loss is cross-entropy in
nats; R_model is a logical-product opportunity percentage, not measured speedup.
The JSON retains exact source paths and SHA-256 values.

Result: at the four recovered thresholds, h-only OL1 has lower validation loss
than both matched arms. Its R_model is higher than no pressure at each of the
four thresholds. It exceeds seven-site pressure at kappa=0, 0.01 and 0.05,
but is lower at 0.1. The withheld 0.5 comparison may behave differently.

Caveats: one seed and one scale; no uncertainty estimate or runtime claim.
The scalar averages six h tensors versus 42 tensors in the all-site arm, so
the target change also changes pressure composition and normalization. Failure
of h-only at the unavailable endpoint would not isolate Q/K/V necessity because
the all-site arm also adds a, m and z. No manuscript update or finding promotion.

Source script: `02_compare_recovered_four.py`. The complete five-threshold
reduction remains `01_compare.py`, pending recovery of the last condition.
