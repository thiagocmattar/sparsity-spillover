# Run036: final-kernel latency of post-hoc control clipping

Status: implementation and local verification; no compute launched yet.

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

Match RTX5090, BF16, batch1, T2048, all50304 logits, uncached causal attention,
runtime seed2801 and timing seed2504. Each of 40 settings uses three fresh
processes, 64 fixed validation inputs and seven paired passes (1344 pairs).
Each process qualifies against the matching clipped native eager reference on
all338 complete blocks from500 MiniPile documents:692224 input tokens,
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
quote on18 September2026:USD0.99/hour. Maximum four cumulative GPU-hours and
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
hook placement, all40 retained condition identities). All242 bootstrap tests
passed. Prepared1417 hash-recorded source/input files. Remote preflight covers
12 settings (both sizes, both activations, p=0,.5,.9), eight validation blocks
and four timing inputs each. All120 final processes then use full qualification.
GPU smoke is still pending; CPU tests are insufficient for numerical qualification.
