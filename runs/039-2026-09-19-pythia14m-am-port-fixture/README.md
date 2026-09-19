# Run039: a/m load-avoidance port with corrected qualification fixture

Status: **complete, recovered, verified, and Pod deleted** on 19 September 2026.
The direct h/z strategy port is correct but slower at a/m for this checkpoint.
All 224 primitive checks, five smokes and 15 scientific processes passed.

| Implementation | Full-model latency (ms) | Change from frozen |
|---|---:|---:|
| Frozen K050 | 0.494995 | reference |
| Port at a | 0.580284 | +17.23% |
| Port at m | 0.630549 | +27.38% |
| Port at a and m | 0.715227 | +44.49% |
| Port at a and m, sparse logic disabled | 0.708533 | +43.14% |

These are geometric means of synchronized full-model host latencies, measured
afresh on one physical RTX 5090 with three randomized fresh processes per mode.
All port process ranges are slower than the frozen range. Maximum measured
logit and pooled-loss differences are zero; BF16 validation loss is
5.83130066301001 in every execution path. Canonical paper evaluation is not
replaced by this BF16 kernel qualification.

At a/m, empty 8x16 tiles occupy 11.12%/0% of the grid, and only 15.37%/0% of
rows have at most two nonzeros. The corresponding short-row fractions at h/z
are 92.75%/99.73%. The port executes 1.895/2.000 times the frozen a/m matrix
instructions because its eight real rows occupy padded sixteen-row matrix
instructions. Most latency regression persists in the dense-layout control;
sparse logic adds a further 6.694 microseconds in the both-port comparison.
This rejects the benefit of this specific direct port, not all possible
load-avoiding a/m implementations.

See the [full observation](observations/001-am-port-verdict.md),
[latency results](results/verdict.md), [pooled mechanism evidence](results/mechanism.md),
and [local raw-data verification](artifacts/local-reduction-verification.json).

## Why this new record exists

Run038 stopped before its first primitive output check: its synthetic input
had 16 rows, while the frozen K042 comparison kernel requires a multiple of
32 rows. Run039 uses 32 synthetic rows. The candidate CUDA, checkpoint,
validation coverage, five comparison modes, numerical tolerances, timing
inputs, repetitions, diagnostics, software and device remain unchanged.
Run038's failure artifacts were recovered and all 38 files hash verified.
This is a fixture correction, not evidence against the port.

The [approved design](../../analyses/024-2026-09-17-h-only-kernel-latency/AM-LOAD-AVOIDANCE-DESIGN.md)
and [Run038 protocol](../038-2026-09-19-pythia14m-am-load-avoidance/README.md)
specify the question, fixed inputs, bounds, interpretation and retention.
The user confirmed both design and launch, and explicitly requested continued
execution without another permission request.

## Execution and evidence

- Same retained 14M T7/Pall checkpoint at kappa 0.5; no training.
- Untouched K050, a-only port, m-only port, both ports, and the both-port dense
  layout control. Only the selected linear implementations change.
- 32 synthetic and 192 captured training-operand checks, five small smoke
  runs, then three fresh full-validation processes per mode.
- Each scientific process checks all 338 complete validation blocks from
  500 documents; 1,444 tail tokens are excluded as in the original protocol.
- Timing uses the same 64 inputs, seven passes and randomized process order.
  Report raw paired measurements and descriptive process ranges.
- Activation zeros, tile occupancy, short-row eligibility, issued/bypassed
  instructions and source-level weight requests are retained. Weight requests
  are not measured memory-bus traffic.

Sources and transferred artifact inventories retain SHA256 identities. The
original Pod deadline was 2026-09-19 17:00:30 UTC, with ten minutes reserved
for retrieval. Both independent stop guards remained active until closeout.

## Verification and closeout

Focused tests: 14 passed in 0.96 s; full bootstrap: 242 passed in 7.68 s.
All 1,383 frozen source/input identities verified on the Pod. The 224 primitive
cases matched native and frozen projection output with zero measured error.
Fifteen scientific processes each checked all 338 blocks; 13,440 raw candidate
and native timing samples were retained. Instrumented counters matched their
independent operand-based oracles over all five complete diagnostic passes.

The closed output archive contains 190 files, all SHA256 verified locally;
its SHA256 is `70229567feb6094b4430451496a2c7b061eb1308487c6fc4da1194392e0bffa2`.
The separate final worker/guard records also verified. `13_verify_local_reduction.py`
recomputed all latency contrasts from raw samples, checked all 50 result-source
identities, and agreed within 1e-12 ms. `12_mechanism.py` pools integer counts
and cross-checks row histograms against operation-product counts.

The corrected input transfer needed an infrastructure retry after an SSH reset;
the declared original-provenance files were restored before verification and
execution. Neither event changed the scientific candidate or benchmark inputs.
Run038 retains the earlier synthetic-fixture failure; its 38 output files were
also recovered and verified. The local reduction verifier initially confused
timing positions with validation-block IDs; the verified selection maps positions
0--63 to the same 64 declared blocks. The corrected verifier preserves all
original samples and the Pod's result files.

One Pod served both records, from 15:30:30 to approximately 16:07:51 UTC
(37.35 minutes), at USD 0.99/hour: approximately USD 0.616 GPU cost plus
temporary storage, below the USD 2 envelope. It was deleted only after all
agreed evidence was recovered and verified. The provider confirmed no Pods
remain; the pre-existing 100 GB shared volume is unchanged. The local stop
guard was stopped after deletion. See [teardown](artifacts/closeout/teardown.json).
