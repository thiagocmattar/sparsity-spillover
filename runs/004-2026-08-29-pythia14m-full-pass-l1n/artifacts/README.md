# Run 004 scientific artifacts

Completed remote attempt trees are copied into `attempts/` and accepted only
after every file matches the attempt's remote `transfer_inventory.json` and the
run-level verifier passes. A copied tree is not by itself evidence of verified
retrieval.

Run 004 now contains all six terminal attempts. Each has 60 declared files and
passed its remote transfer inventory. `verification.json` is the successful
six-condition scientific verification record. `event-history-audit.json`
documents the one raw ReLU-control JSONL prefix anomaly and the exact
non-mutating parse normalization that recovers all 712 events. The raw attempt
trees and their inventories are preserved unchanged.

The post-hoc spillover figures use only those terminal activation diagnostics.
`figure_data_sitewise.json` and `figure_data_attention_output.json` retain the
six-condition count-first reductions at `epsilon = 1e-3`, including integer
hits and denominators. `spillover_figure_completion.json` records the plotting
script and PDF hashes. No checkpoint replay or mutation of an attempt tree was
needed.

`closeout.json` is the machine-readable run closure. It records the verified
scientific totals, the 2026-08-30 zero-Pod audit, the intentionally retained
network volume, posted Pod and volume billing, evidence status, and the one
deferred checkpoint-validation trajectory. It does not promote the observations
to a finding or manuscript claim.
