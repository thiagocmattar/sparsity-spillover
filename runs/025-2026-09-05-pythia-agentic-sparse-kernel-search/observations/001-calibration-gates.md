# Calibration gates: a primitive pass is insufficient

Status: pilot evidence finalized; both Pods deleted. No optimized candidate.

## Question and method

Can the minimally adapted exact Sakana TwELL baseline P0 qualify for a
bounded Pythia specialization search, and what does setup/evaluation cost?
Use retained checkpoint weights, pinned caches and runtime, BF16 B=1,T=2048
full-logit inference. Timing includes dynamic packing and synchronized
completion. Native modules are restored for the reference; QK/PV and LM head
remain dense. Native SDPA is not yet the strongest compiled/graph reference.

Sources: unchanged `afad780` evaluator `01_calibrate.py`, `measurement.py`,
`p0.py` and `kernels/twell_pythia.cu`. Invocation-only retries are retained in
`launch-control/`. The two-block rounding diagnostic is
`launch-control/diagnose_rounding.py`; it does not change the evaluator gates.

## Coverage and result

RTX 5090 (32,607 MiB, driver 575.57.08): all 48 adversarial primitive cases
passed, with zero compute-sanitizer errors. Compile time was 28.745 seconds.
The full-model worker stopped after one of eight conditions completed.

| Checkpoint | Native median ms | P0 median ms | Paired geometric speedup | Validation coverage | Status |
| --- | ---: | ---: | ---: | --- | --- |
| A0-14M | 1.7043 | 2.7391 | 0.6231x | 338/338 blocks | Pass |
| A1-H-14M | 2.0239 | 2.2099 | 0.9156x | 2/338 blocks | **Fail: logit tolerance** |
| Remaining six 70M/410M cases | — | — | — | Not run | Blocked by fail-fast gate |

Table caption: 48 paired measurements per mode across 16 rotating training
inputs and three passes, one process/GPU session. These are calibration
measurements, not confidence-bounded final performance claims. Failed A1-H
timing is retained for diagnosis, not accepted as a valid speedup result.

A0 validation losses: native **5.209999416**, P0 **5.209966001**, difference
**-0.000033415 nat/token**. Coverage includes all 500 documents represented
by 338 complete blocks: 692,224 input tokens, 691,886 shifted predictions,
and the declared 1,444-token excluded tail. Paired validation took about
5.56 seconds (~124,737 input tokens/s); this includes numerical checks and
both implementations, and is distinct from the forward-only timing above.
Timing-phase peak allocation was 621,871,104 bytes (0.579 GiB), reserved
1,371,537,408 bytes (1.277 GiB). This is not a 410M memory-fit result.

A1-H failed on zero-indexed block 1: relative L2 **0.001758925** passed its
0.02 bound, but 26 logits exceeded `0.25 + 0.02*abs(reference)`; largest
absolute error was 0.375 and largest tolerance excess was 0.0396094.
Partial loss is not a complete-validation result and is not promoted.

The independent two-block diagnostic found bit-identical adapter-dense
outputs and reproduced P0's failure. A fused FP32-linear oracle passed those
two blocks; a separately rounded bias oracle failed more substantially.
This supports investigating accumulation order/rounding, but does not prove
the full root cause or validate an oracle-based replacement on all blocks.
No kernel, weight, gate or tolerance was changed after observing the failure.

## Evidence and limits

RTX evidence: `retrieved/rtx5090-001/inventory.json`, 57 files verified after
transfer. Archive SHA256:
`92ad68f79cfe7caf70ecba599be217fb9cdad44e3520af8c4dc24d63c52bb75f`.
Includes raw timings, every A0 block gate, partial A1-H gates, profile,
activation/weight diagnostics, errors, environment and source identities.
The RTX Pod was deleted at approximately 15:29 UTC, after local verification.

This pilot does not validate the full argument chain, attention sparsity,
agentic search gains, architecture scaling, or a speedup/R_model relation.
The failed baseline must not be sealed as a correct starting point.
Full-study ETC across all sizes remains uncalibrated until P0 qualifies.

## Next decision

Resolve the numerical mismatch with fixed tolerances, qualify all eight
calibration checkpoints, then finalize the strongest dense reference and
seal the evaluator before K001. Do not consume the remaining study budget
on an optimization trajectory whose baseline has not passed correctness.

## H100 portability and upstream control

H100 80 GB HBM3, driver 580.126.09: all 48 P0 primitive cases passed with
zero compute-sanitizer errors. Compilation took 40.546 seconds; primitive
worker duration was 44.501 seconds. This is primitive portability, not a
Pythia composed-model pass on H100.

The untouched upstream `benchmark_inference.py` at commit
`661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5` completed with SparseLM0.5B revision
`7c2a0473ec982facd5b21af81fe39c1783f3407e`, B=64,T=2048,BF16, 5 warmups and
50 repeats per implementation. Mean Torch latency was **335.8571 ms**;
TwELL **288.1809 ms**, ratio **1.16544x** (+16.54% throughput).

Important scope clarification: FFN modules are the optimized operations, but
the timer covers the **transformer backbone**, not isolated FFN calls and
not the final LM head. Upstream calls `get_core_model(model)`, which returns
`model.model`; see the pinned [upstream helper](https://github.com/SakanaAI/sparser-faster-llms/blob/661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5/benchmark_base.py#L26)
and [benchmark](https://github.com/SakanaAI/sparser-faster-llms/blob/661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5/benchmark_inference.py).
Inputs are fixed seed-0 random token IDs. Its source producer/cap and numerical
contract differ from exact P0; this run did not add a validation-loss or
Pythia-style numerical gate to the untouched benchmark. It is a successful
upstream execution/timing control, **not a new correctness certification**.

Do not compare 1.1654x directly with RTX's 0.6231x to attribute a hardware
or architecture effect: GPU, model, batch, input distribution, timer boundary,
kernel and numerical contracts are all different. The matched frozen-kernel
comparison remains future work.

H100 setup/primitive/upstream worker ran from 15:39:32 to 15:46:05 UTC
(6 minutes 33 seconds). Evidence was retrieved and all **32 file hashes**
verified before deleting the Pod at approximately 15:51:38 UTC. Archive SHA256:
`15983d95b4e9da648c51c6b1bb981d7c0288918999bbdb64f1b6a13e103572d2`.
See `retrieved/h100-001/inventory.json` and its upstream CSV/source/model manifests.

No publication figure is warranted by this partial, unmatched calibration.
See [cost and ETC closeout](../CALIBRATION_CLOSEOUT.md) for the provisional
ledger and the boundary between measured durations and future budget caps.
