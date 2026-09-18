# Run035: K050-derived shape port for 22 retained Pythia-70M checkpoints

The user approved the Analysis024 compatibility proposal on 18 September 2026
and explicitly authorized RunPod execution, requesting scientific rigor and
comparability for eventual paper use. This authorizes the port and measurement,
not an open-ended optimization search. The detailed approved design is in
[Analysis024](../../analyses/024-2026-09-17-h-only-kernel-latency/70M-KERNEL-COMPATIBILITY.md).

Status: complete, retrieved and verified. All 66 final processes (22 checkpoints
times three) qualified. All 705 returned files passed size/SHA256 checks before
Pod deletion; no Pods or endpoints remain. The initial `k050-70m-v1`
failed the unchanged full-model logit bounds. Its source snapshot and failed
qualification are retained. The corrected `k050-70m-v2` matches the observed
70M native attention schedule and is the single frozen final implementation.
A shape port is not an equal optimization-budget claim.

## Results and recovery

[Full table](TABLE.md), [full-precision reduction](results/70m-final-kernel.json),
[qualification audit](results/qualification-audit.json), and the
[Analysis024 figure and caption](../../analyses/024-2026-09-17-h-only-kernel-latency/observations/004-70m-final-latency-topology.md)
retain the completed evidence. The figure plots all 22 available checkpoints.
At kappa .5, A4 all/h-only latency is 1.626444/1.619968ms and A7 all/h-only
latency is 1.748420/1.618370ms. The highest paired speedup is 1.188014x for
A7+OL1(h). Only the four kappa .5 endpoints exceed their native graph reference;
the other 18 endpoints are slower. These are results of the bounded shape port,
not a 70M optimization search.

Each endpoint retains 1344 matched timing pairs; the cohort has 29568 pairs.
All 22308 candidate validation blocks pass the original combined absolute/
relative tolerance; 22293 are bit-exact. Maximum absolute logit error is .5
(allowed by the combined relative term), maximum relative L2 is 5.17917e-5,
and maximum absolute loss delta is 3.95206e-8. All native graphs are bit-exact
to their eager anchor. Full 338-block diagnostics are retained for all 22 models.

The immutable returned archive is `retrieval/output-001.tar.gz`, SHA256
`3ff8033c63cea78f4b6a820978330f9974e0aaaafebded86d64a038c217ec772`.
`transfer/inventory-001.json` lists every member. Original checkpoints remain
in Runs018/034 and the verified run-local input archive. Reproduce locally with
`09_verify_retrieval.py`, `16_reduce.py`, `28_audit_results.py`, then Analysis024's
`04_plot_70m_final_latency.py`. Raw attempts include the failed v1 preflight;
only `scientific-*-001` v2 attempts enter the final reduction.

The Pod existed from 12:18:08 to 13:19:38 UTC on 18 September. At USD0.99/hour,
estimated compute expense is USD1.0147, plus less than USD0.02 estimated temporary
storage, within the USD10 cap. Billing had no posted records at closeout; that
does not mean zero cost. The existing 100GB network volume was preserved and
the local deadline guard stopped. See `artifacts/cloud-closeout.json`.

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

All six v2 preflight processes qualified across 338 blocks: A0, A4+OL1(all)
at kappa .01/.5, A7+OL1(all) at .5, and both h-only families at .5. The largest
absolute loss delta among these was below 4e-8. The final cohort uses one
unchanged source identity, frozen in `artifacts/frozen-final-port.json`.
After the interrupted input transfer resumed, the complete 6236231680-byte
archive and all 1529 source/input identities were verified before the final
cohort started. GPU measurements run sequentially with no concurrent GPU job.

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

One RTX5090 session, BF16 B1/T2048 full 50304 logits, same pinned runtime as
Run033, runtime seed 2801 and timing seed 2504. Native SDPA and candidate graphs
are paired in each process, with an unmodified eager native correctness anchor.
64 validation inputs x7 paired passes x3 fresh processes =1344 timing pairs per
checkpoint,66 final processes. Recurring preprocessing and logits are timed;
equal input staging and setup/compilation are separate. Absolute latency is
the geometric mean of raw times; speedup is the geometric mean of paired ratios.

All 338 complete blocks/500 documents/692224 input tokens qualify each process;
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
