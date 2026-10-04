# Run059: direct P2 versus Ph inference latency

Status: completed and verified. P2 has 0.677% higher latency at kappa=0 and
1.660% lower latency at kappa=.1 with frozen K050. All three process ratios
have the same direction within each threshold. All numerical checks passed,
with zero observed logit error against each checkpoint's own BF16 eager anchor.
All 99 returned files are hash-verified locally; the owned Pod is deleted and
its local guard is disarmed. Estimated lifecycle cost USD1.45, below USD2.
The provider ledger is still incomplete. See [tables](results/tables.md),
[the observation](observations/01-p2-ph-latency.md), and
[compute closeout](results/compute-closeout.json).

The current request
and standing authorization to run responsible, bounded RunPod experiments and
complete their verified analysis authorize this matched inference follow-up.
The previously executed Run058 remains unchanged.

Question: is T2/P2 faster than T2/Ph using the existing frozen 14M K050 kernel,
at each previously trained threshold 0 and .1? This is a systems benchmark of
four retained seed1234, step712 checkpoints, not training or kernel search.
The models retain their matched random initialization, data order, AdamW/OL1
training history, unilateral h/z gates and absent a/m gates. Ph pressure was h;
P2 pressure was h/z. Original final checkpoints and recovery remain local.

Use the frozen Run033 K050 sources and replay, round_p=False, shortcut=False,
BF16 parameters/activations, FP32 accumulation, B1/T2048 uncached causal
inference with all50304 logits. Preserve exact existing gates and thresholds;
register the existing HZ site alias before checkpoint loading. Native graph
and K050 execution for both checkpoints run in the same fresh process. Equal
input staging is excluded; recurring zero detection, packing, gates and full
logits remain timed. The kernel does not use a static measured tile mask.

Runtime Python3.12, Torch2.11.0+cu128, Transformers5.12.1, NumPy2.5.0,
runtime seed2801, timing seed2504. Disable TF32 and reduced-precision BF16
reductions. Each threshold gets three fresh processes, with the same64 fixed
validation inputs and seven randomized, interleaved timing passes. Primary
estimator is the geometric mean of paired Ph/P2 synchronized-host latencies;
report all process values, absolute medians, CUDA-event timings and throughput.
Do not count individual input repetitions as independent training seeds.

Every process checks each implementation against its own checkpoint's native
eager anchor on all500 validation documents:338 complete2048-token blocks,
692224 inputs,691886 predicted tokens, excluded1444-token tail. Preserve
logit atol.25, rtol.02, relativeL2<=.02 and pooled loss delta<=.001. Failed
qualification is retained and prevents an accepted speedup claim.

Once per checkpoint retain the frozen full-validation exact/near-zero counts,
RMS/L2, weight norms, row occupancy, logical and instrumented kernel work
counters. Retain raw timing pairs, inputs, numerical checks, environment/GPU
identity, compilation/setup times, peak memory and failure examples. Prior
training-only gradient geometry remains in Run058. No clipping sweep or
additional diagnostic selection is needed for this inference question.

P2/Ph speedup>1 supports lower latency in this exact workload; <=1 refutes
that advantage. Compare process variability before labeling a small difference.
This does not establish general GPU, decoding/KV-cache, quality-matched or
training-speed results. BF16 kernel diagnostics differ from Run058's canonical
FP16 logical opportunity; preserve both identities. The manuscript systems
question is addressed using direct measured runtime, without a TeX edit or
automatic finding promotion. Tile pressure remains deferred.

Execution: one Secure RTX5090 currently quoted USD.99/hour. Same-device direct
pairs are preferred over concurrent processes that would contend for the GPU.
The verified14M reference used3.03GiB; this two-checkpoint process is expected
to fit32GB with ample headroom. A representative two-threshold remote smoke
confirms real memory fit, correctness and ETC before complete measurements.
Initial ETC20--45minutes including setup/transfer/cold compile, expected cost
belowUSD1, hard maximum90GPUminutes/USD2 including disk and retrieval.
Use pinned image and dependency lock,20GB container disk and25GB workspace
Pod volume; do not create or attach a network volume. Upload only four final
inference checkpoint sets, frozen sources and validation cache. Return all
results/logs/environment/inventories and verify every hash before Pod deletion.

Both a hidden local identity-scoped provider-stop guard and a credential-free
remote workload deadline enforce the creation-time cap. All execution survives
terminal disconnect and logs under/workspace. Monitor every60seconds, sooner
only for declared warnings or projected completion: OOM, nonfinite/reference
failure, process timeout, stale progress or deadline risk. Report progress,
loss/error, throughput and refreshed ETC. Explicit scoped teardown follows
local scientific/hash verification; retain foreign Pods and the existing volume.

## Compatibility retry before timing

Smoke001 stopped at adapter installation: the archived Run027 characterization
allowlist excluded HZ. No timing or numerical result was produced. The complete
failure and pipeline log remain immutable. Retry002 appends the already used
Run044 HZ adapter and replay: the adapter differs from the original by only
adding HZ to its label allowlist. Every CUDA kernel, checkpoint, gate, timing
and qualification input is unchanged. Alternate02b/03b/05b entry points retain
the original executed sources. All four real checkpoints pass native CPU
forward and BF16 K050 installation checks with the existing adapter. Retry uses
the same owned Pod, environment and original creation-time deadline.

Smoke002 reached the external 600-second limit during cold compilation, before
timing. Attempt003 separates unchanged extension precompilation from the bounded
scientific leaves. Slow imports and vendor metadata on the network-backed Pod
volume remained material. Attempt003 was cancelled before timing and retained.
Infrastructure attempt004 caches the original verified source upload and exact
runtime files on local disk, preserving both persistent originals. Every source
and copied runtime file is hash-checked; symlink targets are checked. This follows
the earlier Run048 local-cache approach. It changes storage placement only, with
the same Pod, runtime, scientific inputs and original deadline. The source path
change can trigger compilation; precompilation remains outside all timed samples.

The resolved local dependency paths triggered duplicate compilation inside the
first smoke. A bounded cold-capture guard briefly paused its supervising
controller, with a maximum900-second cold allowance and no change to the child
measurement code or Pod deadline. The child completed in516.75seconds, within
the original600 seconds. The second smoke completed in83.86seconds; both passed.

Attempt004 completed four scientific processes, then stopped before the next
job because its660-second reservation exceeded the remaining lease. Attempt005
resumed only the two unfinished fixed jobs, in the original order and with the
same inputs, seeds, measurements and numerical bounds. Its stricter300-second
cached-process limit was based on the observed99--136-second full processes;
it retained a60-second retrieval reserve. All six processes completed, taking
99--168seconds each. Original and resumed summaries remain unchanged beside
the explicit combined cohort. This is an infrastructure continuation, with no
retraining, scientific-input correction or repeated/selected successful trial.

Primary host geomeans are0.659690/0.664153ms (Ph/P2) at0 and
0.625165/0.614790ms at.1. P2/Ph paired speedup ranges are
0.988689--0.996753 and1.015749--1.018378. The native graph control is effectively
tied (P2/Ph0.998415/0.998565). BF16 full-validation loss is
5.128739/5.285315 at0 and5.153716/5.331877 at.1; P2's loss penalties are
.156577/.178161. Run058's canonical FP16 figures retain their own precision
identity. The measured small gain at.1 does not establish a general P2 latency
advantage or a quality-matched improvement.

For future reuse, put immutable vendor sources and the pinned runtime on local
disk before compilation, preserve runtime header timestamps, and precompile
dependencies using the same resolved paths as their actual installation chain.
This run retained all original files, failed attempts and setup logs.
