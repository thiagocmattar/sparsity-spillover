# Run 026: Pythia-14M fused sparse kernel search

Status: implementation verified locally; first GPU calibration being prepared.
This continues the user-authorized
autoresearch objective (>1.6x full-model speedup, $25 total RunPod budget).
Run 025 is retained as the immutable source/evidence predecessor.

## Fixed experiment

Target: Run 025 `14m/a7-0p5`, the retained Run 014 step-712 checkpoint.
Pythia-14M has six layers, hidden 128, FFN 512, four heads, vocabulary 50304.
Inference is BF16, B=1, T=2048, full logits, causal SDPA, no KV cache.
All weights, trained gates, thresholds, topology, seeds and pressure provenance
are imported by hash from Run 025's input manifest. No new training occurs.
The source used seed-1234 random initialization and orthogonal L1 at A7-Z-POST
sites; one-sided kappa=.5 at a/m/h/z and symmetric .5 at q_post/k_post/v.

Hypothesis: exact gate/projection fusion and shape-specialized sparse kernels
can reduce end-to-end latency while satisfying the sealed Run 025 numerical
gates. Search varies only implementation, tiling, site selection, and legal
fusion. A correct full-model paired speedup >1.6x supports the objective;
failure to reach it is a bounded negative result, not a universal limit.

Primary comparator is the fastest correct dense implementation with equal input
staging/capture treatment. Eager speedup is also recorded. Dense configurations
are selected on development inputs before candidate promotion. Canonical
R_model is inherited with integer counts and provenance for every variant;
actual BF16 diagnostics are collected separately. Product counts do not imply
a wall-clock ceiling or speedup.

Development: 64 training blocks, seed 2500; timing initially uses their first
16 with seed 2504. Finalists freeze before validation timing (64 validation
blocks selected by seed 2504), seven paired passes and three fresh processes.
Quality covers all 500 documents / 338 complete blocks / 692224 input tokens;
the 1444-token tail is excluded. Logit bounds are atol=.25, rtol=.02,
relative L2=.02; absolute pooled loss difference <=.001 nat/token.

Every attempted variant has a row in `autoresearch/artifacts/*/result.json`,
raw paired samples, code/config hashes, an idea and candidate ID. Compilation
or numerical failures retain their available latency and counts, with explicit
qualification. `autoresearch/progress.csv` tracks every iteration and the best
qualified speedup so far. No failed variant is promoted or omitted.

## Budget and evidence

The new $25 cap includes all new Pod attempts, startup, compilation, transfer,
idle development time and storage. Prefer community RTX 5090; use a measured
live-price fallback when capacity is absent. Protect final validation/retrieval
funds; arm an independent stop guard for each lease. Durable work lives under
/workspace. Retrieve and verify hashes before deleting a Pod.

Collect/retain: exact and near-zero counts at 0/.001/.01, activation RMS/L2,
weight norms, logical products and architecture-ceiling counts, occupancy,
kernel/operator timings, workspace/memory, branch coverage, all errors, and
raw paired timings. The original final checkpoint and full cache identity stay
local. Gradient interaction is training-only; this inference run links the
existing training record. No extra clipping or gradient experiment is added.

This run addresses the runtime qualification of manuscript methodology
`eq:r-model-measured` and `eq:r-model-max`; no TeX change is authorized here.
Matched dense controls distinguish sparse-kernel contributions from capture
and other implementation improvements. Monitor at phase completion or every
10 minutes during long work, reporting loss, throughput, progress and ETC.

## Authorization

The user reiterated the active objective with full access and approval for all
RunPod runs, deployments and transfers on 2026-09-06 after the detailed design
proposal. Proceed under that standing authorization, preserving the inherited
strongest-matched-dense comparator rather than adding another approval wait.

## Iterations

- K017: fuse the existing one-sided gate into exact warp-compacted projection;
  specialize output elements per lane and row warps for narrow Pythia shapes.
  The gate consumes each new input inside the timer; equality survives.

## Current execution

Local: five focused tests and all 242 bootstrap tests pass. CUDA tests remain
remote-only: 120 primitive configuration/shape/input/replay cases, followed by
dense eager/graph/compile screening and complete validation. Local GPU has
about 9 GB free; the rented CUDA development environment is used under the
standing cloud authorization.

First lease: `gc38kxs85gwxra`, RTX 5090 community, confirmed $0.69/hour,
40 GB container + 40 GB Pod volume, pinned RunPod PyTorch image digest.
Independent stop guard PID 28976 is armed for 2026-09-06 15:52:47 UTC.
Maximum first lease: four hours (~$2.80 including Pod disks); full study cap
remains $25. Input archive is 53.4 MB. Exact lease, deadline and guard logs
live in `launch-control/rtx5090-001/`. Verify/retrieve before deletion.
