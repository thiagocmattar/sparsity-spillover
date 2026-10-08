# Run065: retained31M/70M forward decomposition

The user selected profiling existing larger models and pre-approved RunPod execution.
Question: how is complete inference latency distributed in existing T2/Ph31M/70M
checkpoints, and where does each frozen sparse implementation save time?

Four final712-update checkpoints: Run05431M and Run04370M, kappa0/.1.
All are random-init seed1234 pretraining with h-only OL1 and scalar one-sided
h/z thresholds. Weights and gate operators are unchanged. No training, optimizer
updates, threshold changes, kernel search or manuscript edits occur here.
Historical training:712 updates,1024 sequences/update,2048 tokens/sequence,
1,493,172,224 input tokens; h OL1 trust budget1 and weight1; initialization,
checkpoint metadata and source gates are retained with exact input identities.

BF16 inference, batch1,2048 tokens, uncached full50304-vocabulary logits.
One physical RTX5090 for all four conditions. Three fresh processes per condition,
64 fixed random validation inputs,7 randomized paired passes. Modes: native,
frozen sparse and same-implementation skip-disabled. Full own-native numerical
qualification: all338 complete blocks from500 MiniPile validation documents,
1444 tail tokens excluded; unchanged logit atol.25/rtol.02,relativeL2.02,
full validation loss atol.001. Smoke covers8blocks before full cohort.

31M uses Run054 opt073-31m-v1 (short-row prepass and fused MMA fallback).
70M uses Run051 sparse_c with h feature compaction in layers1..5,
h0 dense gate+linear, all z dense gate+dot. No-skip retains conversion and gates,
while disabling sparse gather/short-row/tile skipping as implemented originally.
This is not an architecture-only scaling study:14M Run064 has a corner-tile
operator and a different consumer; these larger models have scalar gates.
The decomposition addresses runtime only, not a new quality-training claim.

Collect baseline host/CUDA latency, additive event partitions with their overhead,
raw CUPTI graph/eager traces and exact ordered kernel-to-stage mappings,
per-layer h/z exact and near-zero counts, row support/empty8x16tiles and RMS/L2,
weight norms, GPU/runtime identities, checkpoints/cache hashes and logs.
Existing final checkpoints remain locally retained. Gradient interaction was
training-time-only and is not reconstructed in this inference run. Internal
fused CUDA work is reported at actual kernel boundaries; no wall-time fractions
are inferred from logical opportunity or invented for overlapping phases.

Support: qualified sparse gains with attributable measured savings; refute:
no sparse gain, gain lost at larger sizes, or numerical qualification failure.
One seed and checkpoint per size/threshold; no claim of broad model generality.

Status: implementation/prelaunch; no larger-model decomposition result yet.

## Execution record

Owned community RTX5090 Pod qon53j3aqgr9gf launched2026-10-07T23:04:54Z,
USD.69/hour; flexible3-hour ceiling andUSD3total envelope. Foreign stopped
Run052 Pod and shared100GBvolume were discovered and left untouched.
Source commit f670d4786f9536fae4bb3c361ed6c550aba758f0 plus sealed Run065
instrumentation. All242bootstrap tests and8instrumentation tests passed;
four checkpoint headers matched30,494,720/70,426,624parameter architectures
and step712. Actual remote numerical smoke remains authoritative.

Infrastructure ingress attempts001..004 preserve the identical sealed bundle:
initial single SSH stream was0.42MB/s; four-stream resume raised aggregate
throughput, but individual slow connections were retried with verified prefixes.
Final sequential remaining ranges assembled the original757,424,421-byte archive;
its SHA256 and all2,987member identities were verified before execution.
Attempts are documented underprelaunch/ingress-attempts; no scientific input
or worker source changed. Frozen environment wheel prefetch ran separately
while upload proceeded; authoritative setup used the same58package pins.

The persistent checkout, inputs, logs, artifacts and uv cache are in/workspace.
Compilation/venv cache is in/opt; only caches are ephemeral. Detached controller
starts four8-block smokes, then12fresh full profiling processes in a randomized
order. The scoped hidden local guard stops only the owned Pod at the3-hour
backstop, retaining the disk if retrieval is outstanding. Normal teardown will
verify all artifacts locally and terminate the Pod sooner.

## Current status: complete and verified

This supersedes the historical prelaunch status above. All 4 GPU smokes and
12 fresh full processes passed. The 24,336 complete-output comparisons cover
all 338 validation blocks in every process; all 48 trace mappings succeeded.
Unchanged bounds: atol .25 + rtol .02 per logit, relative L2 .02, loss atol .001.
Maximum observed absolute logit difference is .75, relative L2 .001005;
maximum full loss difference is .00003298. These results satisfy the original
bounds; the larger-model optimized graphs are not claimed to be bit-identical.

Full-forward synchronized host geometric means, milliseconds:

| Model | kappa | Native | Sparse | No-skip | Reduction vs native |
| --- | --- | --- | --- | --- | --- |
| 31M | 0 | 1.0558 | 1.0069 | 1.0827 | 4.632% |
| 31M | .1 | 1.0556 | .8908 | 1.0826 | 15.612% |
| 70M | 0 | 1.7363 | 1.2463 | 1.8154 | 28.223% |
| 70M | .1 | 1.7318 | 1.1944 | 1.8090 | 31.032% |

The distribution changes. The 31M joint gate/down/residual stage costs
431.555/315.837 microseconds (44.43%/36.89% of sparse GPU-active time),
versus native gates/down/residual at 172.881/172.851 microseconds. Other
component and dispatch savings compensate for this inefficient joint consumer.
The kappa0 prepass completes no token row; kappa .1 helps mainly layers1-3.
The 70M vocabulary projection is its largest sparse component, at
36.55%/38.01%; attention takes 18.17%/18.92%. Actual five-layer h feature
unions retain 7.815%/3.052% of their no-skip padded tensorcore operands.
Those are logical product counters, not runtime speedups; z stays dense.

The no-skip control measures the net skipping benefit within each frozen
implementation. It retains conversion and small-group padding and is not the
best dense implementation. These are existing scalar-gated larger models;
Run064's 14M corner gate and specialized consumer differ. The comparison is
descriptive, not architecture-only scaling of the new corner consumer.

[Complete map and diagnostic](observations/01-larger-forward-decomposition.md),
[PDF](figures/01-larger-forward-decomposition.pdf),
[every stage](results/forward-stages.csv),
[every ordered CUDA activity](results/cuda-kernel-nodes.csv), and
[verified reduction](results/decomposition.json) retain the evidence.
Events perturb execution by 8.46-16.81%; no instrumented measurement was
rescaled into the baseline. Raw graph traces contain no omitted GPU memcpy
or memset activities; graph gate-copy kernels are included explicitly.

Post-cohort diagnostics read actual 70M union counts and 31M capped prepass
states over all 338 blocks after latency collection finished. The original
frozen forward calls remain unchanged. Original checkpoints/cache remain
local. All 272 returned artifact/runtime files passed SHA256 checks.

The owned Pod was terminated at 2026-10-07T23:51:04Z after verified retrieval;
re-listing found no new billable resources. The scoped stop guard was canceled
only after confirmed termination. Foreign stopped Run052 and shared100GB
volume remain untouched. Provisional run cost is USD .531 GPU + .007 storage,
approximately .54 total. A provider bucket received after teardown reports
USD .07235; final billing reconciliation is pending because that is below the
elapsed-time estimate. Neither value is represented as a finalized invoice.
[Closeout](prelaunch/closeout.json) and [transfer receipt](transfer/receipt-001.json).
Storage rates were checked against
[RunPod pricing](https://docs.runpod.io/pods/pricing).

All five PDF pages were rendered and visually inspected. No manuscript or
finding-promotion edit was requested or made. Source scripts are run-owned.
