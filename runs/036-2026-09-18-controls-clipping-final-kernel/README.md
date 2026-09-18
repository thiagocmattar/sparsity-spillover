# Run036: final-kernel latency of post-hoc control clipping

Status: complete. All 40 settings, 120 full-validation processes and 12 smoke
settings qualify. All 1,032 returned files are verified locally. The Pod was
deleted at 19:20:58 UTC on 18 September 2026; no Pods or endpoints remain.
Estimated GPU expense is USD2.161 plus temporary storage, below the USD5 cap.
The existing network volume is unchanged; the local deadline guard is stopped.

Results: [complete latency table](TABLE.md), [source JSON](results/clipping-final-kernel.json),
and [Analysis024 figure and observation](../../analyses/024-2026-09-17-h-only-kernel-latency/observations/024-controls-posthoc-final-latency.md).
Every positive-p 14M point is slower than its own p=0 final-kernel control.
At 70M, p=0.9 gives 1.2315x/1.1815x over the dense/ReLU final-kernel controls,
with large loss increases; the matched clipped native graph remains faster.

The user requested the missing Analysis024 dense/ReLU clipping latencies at
14M and 70M, then explicitly said to do it and authorized RunPod. The user
also accepted the diagnostic inventory below. This is inference only.

## Design and interpretation

Measure whether the retained post-hoc clipping sweeps produce useful observed
quality/latency operating points under the final kernel at each size. Vary only
the retained target p=0,.1,...,.9 within each fixed checkpoint. The four inputs
are the A0 GeLU and A1-H ReLU step712 checkpoints from Runs004/018, from the
original random pretraining with seed1234. No initialization, training, optimizer,
pressure, learned gate, weight, calibration sample or threshold is changed.
There is no new OL1 objective or backward pass.

Use all four clipping sites a,m,h,z in every layer, after the trained activation,
zeroing abs(x)<=t with the original per-site/layer thresholds calibrated on the
first ten training blocks. A1-H means the trained ReLU topology, not h-only
post-hoc clipping. Natural zero ties mean p is not achieved sparsity.

Freeze original K050 at 14M and k050-70m-v2 at 70M, with their original projection
and attention skip policy. A separate PyTorch clipping adapter runs inside each
timed graph. This is final-kernel execution plus the cost of post-hoc clipping,
not a newly fused clipping kernel or an optimization search. At all-zero
thresholds, the value-identity clipping is elided. Candidate h clipping on ReLU
combines the existing deferred ReLU and threshold mask; the frozen joint gate
then sees nonnegative operands. Source kernels remain byte-identical.

Match RTX5090, BF16, batch 1, T=2048, all50304 logits, uncached causal attention,
runtime seed2801 and timing seed2504. Each of 40 settings uses three fresh
processes, 64 fixed validation inputs and seven paired passes (1344 pairs).
Each process qualifies against the matching clipped native eager reference on
all 338 complete blocks from 500 MiniPile documents: 692224 input tokens,
691886 prediction tokens,1444 excluded tail tokens. Keep the existing numerical
bounds (logit atol .25/rtol .02, relative L2 .02, loss delta .001); never qualify
from loss alone. Failed configurations remain visible and are excluded from
claims of runtime gain. Native graph has the same clipping rule and is the
paired denominator. Compilation/setup and equal input staging are untimed.

Retain the original FP16 loss and count-pooled R_model separately from new BF16
native/candidate losses and actual-operand diagnostics. Do not pretend precision
changes preserve clipping masks or use evaluation_seconds as inference latency.
The result addresses Analysis024's quality/latency comparison and the manuscript
kernel-realization question. No manuscript TeX or promoted finding is authorized.
Support means qualified lower latency at an acceptable measured loss; null or
negative latency gains refute acceleration for the tested implementation. This
is one training seed, one device/session, no decode/KV-cache measurement, and
not a causal isolation of sparsity or an equal optimization-budget size comparison.

## Agreed diagnostics and retention

Retain raw timing samples/identities, full-block numerical gates and losses,
exact/near-zero counts (0,.001,.01), activation sums/squares/RMS/L2, per-layer
weight norms, BF16 operand logical opportunity, issued/bypassed MMA and SIMT
counters, per-row occupancy, thresholds and original FP16 logical counters.
Gradient conflict is inapplicable to this inference-only study. The unchanged
final checkpoints and exact cache identities are already retained locally.
Hash-check source/input bundles and all returned artifacts before Pod deletion.

## Execution envelope

One sequential RTX5090 Pod, pinned Run035 image/environment. Live secure GPU
quote on 18 September 2026:USD0.99/hour. Maximum four cumulative GPU-hours and
USD5 total task spend including temporary storage; expected1â€“2 hours, to be
refreshed after preflight. Use20GB container plus30GB Pod storage at/workspace,
no new network volume, preserve the existing100GB volume. The local5070Ti
laptop is not the RTX5090 comparator and cannot provide the matched hardware.

Monitor every60 seconds with read-only checks; report setting/process progress,
loss, validation/diagnostic throughput and ETC. Warn on a stale event for5min,
nonfinite values, failed correctness, process exit, disk<5GB, or budget risk.
Stop scheduling with sufficient transfer reserve before the four-hour deadline.
Detached execution, persistent logs, a local cost guard, verified retrieval and
explicit Pod deletion are required. The transfer includes sources, manifests,
raw results, logs and diagnostics; weights need not be returned unchanged.

## Verification

Four focused tests passed (cutoff equality, signed/deferred-ReLU behavior, exact
hook placement, all 40 retained condition identities). All242 bootstrap tests
passed. Prepared1417 hash-recorded source/input files. Remote preflight covers
12 settings (both sizes, both activations, p=0,.5,.9), eight validation blocks
and four timing inputs each. All 120 final processes then use full qualification.
All 12 GPU smoke settings passed the unchanged numerical qualification bounds.
All 120 full-validation processes also passed. The first returned process was
checked directly against the raw 448-pair timing contract; final reduction checks
all 1,344 pairs per setting and all 40 full diagnostic records.

## Attempt 001 infrastructure record

Pod `283sr45ea620r2` started at 2026-09-18 17:09:58 UTC on an RTX5090 in
EU-RO-1, at USD0.99/hour. The four-hour stop guard is armed for 21:09:58 UTC;
it preserves storage if the deadline is reached. The existing network volume
was not attached or changed. Source preparation is committed as `9197b61`.

Initial direct SFTP was too slow, so the same 714,229,760-byte input archive
was sent through the encrypted relay. The relay left incomplete chunks; bounded
parallel SFTP repaired only mismatched 8 MiB chunks. Two intermediate repair
attempts are retained: the first hit a local SSH-cache write race, and the
second left two stalled workers. Serializing connection setup and bounding
channel waits let the final repair finish. The complete archive SHA-256 is
`53a9cc985cf8827f0969c52e1af9998b46c44d5dc670eaacf8d3ed8369751291`.
All 1,417 frozen file identities passed before any GPU benchmark began.

The detached preflight began at 17:31:27 UTC. Compilation of the first 14M
kernel exceeded the five-minute status warning; read-only process/log checks
confirmed successive CUDA extensions were still building. Compilation is
outside timed inference. Transfer receipts, repair records, and runtime logs
retain the chronology; no scientific input or kernel source changed.

After the first warm smoke process spent about 90 seconds on imports/model
loading/extension setup but less than one second on timing, validation and
diagnostics, the pinned virtual environment was copied to `/opt/run036-venv`
on the same Pod's local disk. `19_local_runtime.sh` completed `diff -qr` with
no differences before activating a launcher wrapper for subsequent processes.
The original Python symlink is retained as `python.network-original`.
Scientific scripts, checkpoints, cache, package bytes and GPU are unchanged;
outputs remain on persistent storage. The copy and verification ran during
preflight only, before the scientific sweep. `runtime/local-runtime.json`
records activation, and runtime logs retain any consequent extension rebuilds.

## Verified closeout

Scientific execution completed at 19:20:02 UTC. Retrieval verified the archive
SHA-256 `477bb68e22d63bec19e335948e87fa08c42b684981b30f5e10c9d57c639be24d`
and all 1,032 inventoried files before deletion. See
[verification](artifacts/verification.json), [inventory](transfer/inventory-001.json),
and [provider closeout](prelaunch/closeout.json). The immediate provider billing
snapshot is delayed (USD1.182 including disk at query time); it is not a final
invoice. The elapsed-time GPU estimate uses the live USD0.99/hour rate through
deletion. Inputs, final checkpoints and source/cache identities remain local.

The reduction retains original FP16 quality/sparsity and new BF16 losses,
same-session p=0-relative ratios, matched native ratios, three-process timing
ranges, and source hashes. No manuscript edit or finding promotion is included.
