# Run067 launch packet â€” approval pending

## Implemented comparison and qualification

Frozen Run066 endpoints:both kappa0/.1 and Ph/Ph+head histories. Each uses native
FP32 parameters, FP16 autocast and the original FP32 final head gate. Dense
cuBLAS, custom no-skip and custom sparse head paths are paired. Transformer
blocks remain native dense execution, with unchanged h/z tile gates. All50304
logits at all2048 positions remain required. No training or optimizer executes.

`head.cu` supplies native-precision support/cast preparation and the projection.
Empty M16/K16 consumer groups bypass weight reads and MMA instructions. They
combine two independently gated M8/K16 tiles; both occupancy levels are counted.
Static FP16 weight preparation is matched; custom weight-fragment permutation
is additionally reported. `forward.py` preserves eager causal SDPA during graph
capture. The original eager model is the numerical anchor. `02_benchmark.py`
qualifies and measures the three paths; verification, sealing, receipt, scoped
stop guard and report/figure scripts are run-local.

Evidence:254 CPU tests pass (242 bootstrap+12 focused,8.24s);48 final GPU component
checks pass. All four models pass full local qualification:three full graphs
per model Ã—338 blocks=4056 comparisons, all exact logits and loss agreement.
Six head graphs per model Ã—64 real operands=1536 comparisons also match exactly.
Final-code tag006 repeats the component checks and four eight-block smokes after
metadata/setup additions; arithmetic/CUDA/forward/bounds remain unchanged.
All Python, shell and PowerShell sources parse. Failed checks are retained.

The local RTX5070Ti laptop has12GiB VRAM. The exact B1/T2048 representative fits
at1.593GiB peak allocation. The four full qualification processes took
86.36â€“89.60s each, including numerical checks, diagnostics and development
timings. Local timing does not establish the requested RTX5090 result.

## Execution definition and ETC

One physical RTX5090/32GB, twelve sequential fresh processes:three per model.
Each qualifies all500 validation documents/338 complete2048-token blocks,
692224 input tokens/691886 predictions, excluding1444 tail tokens. Each times
64 fixed inputs Ã—7 randomized paired passes, with host and CUDA-event replays
separate. Full-forward and gate/cast/projection include recurring preparation;
projection-only excludes it and is diagnostic. Equal staging/setup are excluded.
No softmax, loss or token selection is inside forward timing.

Run the pinned component checks and four eight-block smokes first. Failures stop
before the twelve full processes. The representative local cohort projects
17.6min for twelve qualification/measurement processes; allow15â€“25min on the
5090, plus provisioning/setup/upload/cold compilation/retrieval. Proposed total
ETC25â€“50min; availability and package-transfer uncertainty remain. Keep the
flexible two-hour backstop rather than tightening it to the estimate.

## Live cloud envelope

Catalog checked2026-10-08 19:01:22UTC:one RTX5090 has LOW stock in Secure and
Community. Secure quoteUSD1.19/h,32GB, compatible CUDA12.8 stock in EU-CZ-1 and
EUR-IS-1. Community quoteUSD.69/h, compatible host CUDA13.0/13.1/13.2 stock.
Earlier reads returned NONE; these current reads supersede them. See
[catalog](prelaunch/catalog-002.json). Recheck stock/price immediately before
creation; approval can cover either cloud at an hourly ceilingUSD1.19, with
Secure preferred while available. Use one GPU throughout all twelve processes.

Pod name:`run067-sparse-vocabulary-001`; pinned image:
`runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
40GB container disk+25GB persistent Pod workspace mounted at`/workspace`.
No network volume is required. The existing stopped Pod`qa00sm05lmrs7w` and
100GB network volume`9luykg5yc3` are neither reused nor modified.
Pin PyTorch2.11.0+cu128,Transformers5.12.1,NumPy2.5.0,safetensors.8.0,
ninja1.11.1.4; retain actual package freeze, GPU identity and CUDA details.

Maximum billable GPU duration:two hours from creation. GPU cost at the quoted
hourly ceiling:USD2.38. Running65GB storage atUSD.10/GB/month is approximately
USD.018 for two hours using730h/month; expected total approximatelyUSD2.40.
Proposed total budget:USD5, providing operational margin. Storage remains billed
if the deadline guard stops the Pod for retrieval:25GB atUSD.20/GB/month is
approximatelyUSD.00685/h until deletion. Runpod documents no ingress/egress fees.
[Official storage/transfer pricing](https://docs.runpod.io/pods/pricing).

## Transfer, monitoring and teardown

The sealed upload has69 members/228,325,386 uncompressed bytes; archive
209,386,776 bytes. All members have byte/SHA256 identities. It contains only
four final model checkpoint files/configs/provenance, the full validation cache,
exact gate/runtime sources and this run's executable packet. Training cache,
optimizer checkpoints and other source-run artifacts remain locally retained.
No dataset/model download is needed on the Pod. Estimated scientific return
inventory<20MB plus logs/environment evidence; actual sizes/hashes are sealed.

Return all raw timings, full numerical and pooled loss checks, head-graph checks,
input identities, native head activation and8x16/16x16 counters, independent MMA
counts, logical activation-zero opportunities, runtime/setup/memory/source
identities, logs, failures and verification. Original checkpoints, weight/LN
statistics and training-only gradient/OL1 evidence remain retained in Run066.
The user confirmed this post-hoc inventory; no clipping sweep or new gradients
apply to frozen inference.

Upload under`/workspace/run067`; detach the authoritative setup/pipeline process
with `RUN067_SOURCE_COMMIT` set to the locally committed, blob-verified source.
The minimal remote checkout verifies every uploaded byte and publishes that
source Git identity in each manifest; it does not need repository credentials.
Detach the authoritative process
with PID, durable logs and terminal exit file. Monitor read-only every60s,
reporting stage/model/replicate, completed validation blocks, current finite
losses, qualification throughput, refreshed ETC, memory/utilization and disk.
Check at a nearer projected completion when ETC<60s. Warning conditions:
nonfinite values, numerical/loss criterion failure, OOM, disk exhaustion or
stale progress>300s; inspect active compiler/download separately during setup.

Arm the workstation stop guard with the created Pod ID, expected name and
creation+two-hour deadline; launch it hidden. Stop is a retrieval-preserving
backstop. At normal completion verify the complete twelve-process cohort,
seal/copy the result archive and logs, and verify every local SHA256/byte count
before deleting the owned Pod. Disarm its guard and re-list resources. Never
delete the Pod before agreed artifacts are verified. Generate the comparison
table and visually checked publication PDF locally, retaining observations.

Separate explicit launch approval is required by AGENTS.md; this packet is
implemented and reviewable, but no billable resource has been created.
