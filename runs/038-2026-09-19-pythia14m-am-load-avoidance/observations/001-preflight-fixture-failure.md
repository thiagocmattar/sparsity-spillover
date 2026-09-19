# Synthetic fixture mismatch before qualification

Question: can the final h/z projection strategy reduce full-model latency when
ported to a/m in the retained 14M T7/Pall checkpoint at kappa 0.5?

Method: the approved direct checks compare the port with native linear output
and the frozen K042 a/m implementation before any full-model experiment.
The initial synthetic tensor used 16 rows; K042 explicitly requires the row
count to be divisible by 32 (`Pythia a/m shapes`). Its call failed before the
new candidate was called in the first case.

Coverage: zero completed direct cases, zero model-level smoke runs, zero
scientific validation/timing processes. No validation loss or latency result
is claimed. There is no figure or caption for this operational observation.

Result: a test-fixture error, not evidence about port correctness or speed.
[Run039](../../039-2026-09-19-pythia14m-am-port-fixture/README.md) changes the
synthetic row count to 32 and preserves the candidate CUDA and full-model
protocol. The same Pod and original cost/deadline envelope are reused.

Evidence: [direct-check record](../artifacts/cuda-controls.json),
[final worker log](../artifacts/closeout/execution-final.log), and
[retrieval verification](../artifacts/verification.json). All 38 archived
files and the final worker/guard records were copied and SHA256 verified
locally before the corrected execution began. Source: [06_cuda_checks.py](../06_cuda_checks.py).

Infrastructure notes: the input transfer verified; the initial background
shell held the SSH channel after starting the stop guard. Setup was started
with a detached subprocess and completed successfully. This is recorded in
`artifacts/infrastructure/002-setup-detach/launch.json`.
