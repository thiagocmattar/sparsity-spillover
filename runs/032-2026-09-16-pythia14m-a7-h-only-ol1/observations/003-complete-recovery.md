# All five conditions recovered and verified

Question: recover the complete approved A7-gating/h-only-OL1 experiment,
including the missing kappa=0.5 endpoint and retained checkpoints.

Method and coverage: the unchanged `03_verify.py` checks each original
transfer inventory and scientific invariants, then verifies the five-condition
cohort. All five conditions completed 712 optimizer updates and 1,493,172,224
input tokens. Each final validation covers 500 documents, 338 complete
2,048-token blocks and the reported excluded tail of 1,444 tokens. Initialization
and realized data-order hashes match the approved comparison runs.

Result: all five original per-file inventories pass locally. Each condition
retains twelve model checkpoints (steps 0,1,2,4,8,16,32,64,128,256,512,712)
and three optimizer/scaler/RNG training-state files (steps 256,512,712), plus
training events, activation statistics, weight norms, logical-product counters
and OL1 interaction diagnostics. The remaining condition finished training
on 16 September at 22:23:51 UTC; no retraining was needed. Its recovery passed
on 17 September at approximately 04:50:33 UTC, and its Pod was deleted after
verification. The complete verifier was also rerun successfully at closeout.

Legend/caption: `artifacts/verification.json` is the complete cohort record;
Analysis 022's `artifacts/comparison.json` contains the matched final metrics
and source hashes. R_model is a logical-product opportunity, not a speedup.

Caveats: one seed/scale. Original budget enforcement failed during a controller
interruption; recovery does not make that budget compliant. Auxiliary process
logs were retrieved over authenticated SSH and do not have an independent
original per-file hash inventory. The scientific files and checkpoints do.

Source scripts: `03_verify.py`, `19_retrieve_missing.py`,
`21_stream_staged_archive.py`, `22_retry_kappa05.py`, and `23_wait_for_kappa05.ps1`.
The full account audit records zero Pods/endpoints and the unchanged pre-existing
network volume. No manuscript edit or finding promotion was performed.
