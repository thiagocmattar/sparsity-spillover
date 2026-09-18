# Run035: K050-derived shape port for 22 retained Pythia-70M checkpoints

The user approved the Analysis024 compatibility proposal on 18 September 2026
and explicitly authorized RunPod execution, requesting scientific rigor and
comparability for eventual paper use. This authorizes the port and measurement,
not an open-ended optimization search. The detailed approved design is in
[Analysis024](../../analyses/024-2026-09-17-h-only-kernel-latency/70M-KERNEL-COMPATIBILITY.md).

Status: numerical qualification in progress. The initial `k050-70m-v1`
failed the unchanged full-model logit bounds. Its source snapshot and failed
qualification are retained. The corrected `k050-70m-v2` matches the observed
70M native attention schedule. No scientific cohort result is accepted yet.
A shape port is not an equal optimization-budget claim.

## Numerical compatibility record

The first shape port passed all 28 operator checks but failed elementwise
logit bounds in 14 of 338 baseline blocks. Its loss delta was only -0.0000457;
passing loss alone was insufficient. Untimed component substitutions on three
failing blocks isolated attention: substituting native attention made complete
model logits bit-exact while retaining the ported normalization and sparse
projections. Native profiling then identified D64/M128/N128/four warps and no
KV splitting, versus the 14M-derived two-split M64/N256 schedule.

Version2 keeps the exact-zero MMA bypass and adopts that observed native
schedule, preserving its softmax and reduction order. This is a correctness
correction, not a performance search or relaxed tolerance. Its operator checks
match native attention bit-exactly for dense, zero-query and sparse cases.
Independent issued+skipped attention MMA totals are 557056 per operation per
layer/block. The change from v1's 589824 reflects tile padding; logical-product
denominators do not change. All 248 bootstrap and run-contract tests passed.

`00_port_sources.py` creates the initial shape port; `22_native_attention_port.py`
applies the recorded v2 correction. The original v1 source bundle and hashes
are under `bundles/port-v1-preflight.tar` and `provenance/port-v1-preflight.json`.
Raw qualification, native dispatch and substitution evidence are retained in
`artifacts/`, including failed attempt `qualify-c00-r1-001`.

## Contract

All 22 local final step712 endpoints from Runs018/034: A0 GeLU, A1-H ReLU,
A4/A7 with OL1(all)/OL1(h) at kappa 0,.01,.05,.1,.5. No training or weight
updates. Canonical initialization/data order/seed1234 and training budgets stay
those in their original source manifests. No pressure-free A4/A7 grids exist.

One RTX5090 session, BF16 B1/T2048 full50304 logits, same pinned runtime as
Run033, runtime seed2801 and timing seed2504. Native SDPA and candidate graphs
are paired in each process, with an unmodified eager native correctness anchor.
64 validation inputs x7 paired passes x3 fresh processes =1344 timing pairs per
checkpoint,66 final processes. Recurring preprocessing and logits are timed;
equal input staging and setup/compilation are separate. Absolute latency is
the geometric mean of raw times; speedup is the geometric mean of paired ratios.

All338 complete blocks/500 documents/692224 input tokens qualify each process;
1444 tail tokens excluded. Logit atol=.25,rtol=.02,relative-L2<=.02 and loss
delta<=.001 remain fixed. Failures remain visible and cannot qualify as gains.
Canonical FP16 pooled integer S_model is reused separately from BF16 operand
and executed-work diagnostics. Raw timings, full validation, failure examples,
exact/near-zero counts, RMS/L2, weight norms, logical/skip/occupancy counts,
runtime/source identities and original final checkpoints are retained.

## Execution envelope

User authorization: "Approved. Go on. You can use runpod to it.. we have
balances there. Make sure you follow scientific rigor and comparability, so
we can move this result to our paper."

One Secure RTX5090, live quote USD0.99/hour (18 September), no Community stock.
Cap USD10 including temporary storage and retries, maximum eight cumulative
GPU-hours. Use the prior pinned CUDA12.8 image and dependency lock. Local GPU
is a 12GB RTX5070Ti Laptop, not the matched RTX5090. Prepare locally; use the
cloud for CUDA compilation, operator checks, representative full validation,
calibration and final measurement. Refresh ETC after smoke/calibration. No
kernel-search budget is included. Persistent /workspace holds all logs/results.
Stop guard preserves data at the deadline. Copy and SHA256-verify every output
before deletion, then confirm no task Pods remain. Existing network volume
is not deleted. Monitor at60s, warning on process failure, nonfinite values,
numerical rejection, missing progress, or projected budget overrun.

The launch record will capture the exact image, storage, source hash, price,
deadline and checks. The publication figure will use the four topology labels,
separate dashed pressure curves, subtle annotations and fitted data limits.
No manuscript text is changed by this run.
