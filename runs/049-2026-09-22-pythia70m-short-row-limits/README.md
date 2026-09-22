# Run049: 70M h/z short-row limits at lower loss

The user explicitly approved both design and RunPod launch on 22 September 2026.
Primary question: can limits 16/32/64 improve the retained T2/Ph kappa=.1
checkpoint over frozen opt073 and native PyTorch Base? Base and T2/Ph .05 are
matched controls. No training, weight update, clipping or threshold change.

Use the step712 checkpoints c00/c24/c25 from Run045. These inherit random
initialization and data-order seed1234, the same 1.493B-token/712-step training
recipe and AdamW history; T2 is HZ=(h,z), one-sided gates, h-only OL1 lambda=b=1.
The manuscript question is lower-loss runtime improvement at 70M; existing
operational gates and numerical bounds execute. No manuscript edit is included.

All new limits retain opt073's dense components and M8/N256/K16 layout. A dense
ungated layer uses the existing native-h/z replacement, identified by gate
configuration rather than checkpoint identity. Gated layers use the selected
short-row limit and inherited matrix fallback. Native h/z throughout is also
evaluated as a practical fallback/comparator. This first bounded comparison
does not add an activation-dependent native-GEMM dispatcher.

Six implementations: native PyTorch, frozen opt073, opt073 with native h/z,
and limits16/32/64 with the same efficient ungated-layer fallback. All are
measured in randomized paired order on one physical RTX5090. BF16, batch1,
T2048, full50304 logits, no decode cache. All recurring scans, gates, layouts,
scalar work and fallback execution are timed; compilation, static weight
layout, graph capture and equal input staging are excluded. Precision flags
and runtime versions match Run045.

Development uses the first16 fixed training blocks, five timing passes. Freeze
the fastest qualified c25 policy among native_hz and the three new limits,
requiring qualification at all three checkpoints. All three limits are
predeclared and retain final results, including failures; no retuning from
validation. Final measurements use64 fixed validation inputs x7 passes x3
fresh processes per checkpoint. Every implementation is checked against the
same-checkpoint unmodified eager output over all338 complete blocks from500
MiniPile documents (692224 inputs,691886 prediction tokens,1444 excluded tail).
Bounds remain logit atol.25+rtol.02, relativeL2<=.02 and pooled loss delta<=.001.

Support requires numerically qualified latency reduction at c25, reported
against the current kernel, same-checkpoint native/efficient h/z and a common
native Base. A1.2x native-Base ratio is an illustrative practical target, not
an assumed result. Slower new limits or failures refute their usefulness for
this implementation. Scalar and matrix reduction ordering may differ.

Retain all checkpoints locally, cache/source hashes, raw host/CUDA timings,
all per-block errors/losses, exact/near-zero/RMS/L2 and weight statistics,
per-layer h/z occupancy and joint group counts, independently verified MMA and
scalar work counters, profiles, compilation logs, selection and failures.
Gradient interaction remains in original training records and cannot be
reconstructed here. The user confirmed this diagnostic inventory on22 September.

Execution envelope: one Secure32GB RTX5090; live22 September quote USD.99/hour,
LOW availability EUR-NO-1. Maximum3 cumulative GPU-hours andUSD5 including
temporary storage; no new network volume. Use the previously pinned image
digest,40GB persistent /workspace and30GB container storage, SSH only.
Expected45--100min including setup, transfer, compilation and recovery, with
uncertainty from first compilation and host transfer speed. The local12GB
RTX5070Ti cannot provide the approved same-GPU RTX5090 comparison.

Run detached with persistent logs, a workstation provider-stop guard and a
remote workload deadline. Reserve20min for collection and verification. Monitor
every60s, reporting progress/loss/throughput/ETC; investigate numerical failures,
source/runtime drift, less than8GiB free GPU memory, less than5GB free disk,
ten-minute stale progress or deadline risk. Copy and SHA256-verify artifacts
before deleting the owned Pod. Existing shared100GB volume stays untouched.

Local prelaunch verification: seven focused selection/count/coverage tests and
all242 bootstrap tests pass. Three real CPU checkpoint smokes pass for Base,
T2/Ph .05 and .1, with unchanged gate identities, finite native logits and all
five custom-policy installations. These checks do not execute CUDA. The remote
preflight will check112 synthetic cases (four capacities including the exact
limit8 reference, seven occupancy/gate cases, two gate settings, skips on/off),
including independent work counters and count-on/off equality, then a four-block
six-backend model smoke. Full338-block qualification remains mandatory.

The input preparation verifies1465 retained source/input files. The selected
checkpoints and validation/training-development cache are copied unchanged.
The pinned CLI was checked against its update endpoint, which still offers
2.14.0; that version remains fixed for this run. The exact lease, bundle hash,
resource fit and numerical results will be retained under prelaunch/artifacts.

## Execution and verified outcome

Prepared sources were committed as `0f46d2de`. Pod `kym4fmsrbsg1s6`
(`run049-70m-short-rows-001`) was created in EU-RO-1 at17:30:12UTC on22 September,
atUSD.99/hour. The first local launcher used a Python environment without
Paramiko; restarting it with the repository environment required no new Pod or
scientific attempt. The788384619-byte input bundle transferred in99.4seconds;
all1465 retained input/source files verified on the Pod.

All112 GPU operator checks and the four-input six-backend model smoke passed.
Development selected `native_hz` on the original16 training inputs. A diagnostics
import collision interrupted the first .05 final replicate after validation.
The frozen scientific files were preserved; explicit bindings to the intended
run-local diagnostics/profile modules resumed execution through
`provenance/recovery-001/`. A complete one-input diagnostic/profile smoke passed
before the unchanged final schedule resumed. The failed attempt remains intact;
only its new complete attempt002 enters the final reduction.

All nine final processes completed by18:07:30UTC. All298 output files and the
archive SHA256 verify locally; the recovery reducer reproduces the remote
summary exactly. A local optional-control metadata collision was repaired from
the verified archive and the retriever now preserves inventoried provenance.
Its repair record is `prelaunch/retrieval-repair-001.json`.

**Result:** efficient dense h/z reaches1.324743ms at T2/Ph .1, versus1.642892ms
for native Base:1.2402x throughput,19.4% lower latency. Efficient Base itself is
1.254255ms. Larger sparse limits16/32/64 do not establish an additional benefit:
all fail full-model elementwise numerical qualification at .05 and .1, and
limits32/64 are slower. The complete comparison, work/profile explanation and
limits are in [observation001](observations/001-short-row-limits.md).

Reproduce the returned reduction with:
`python provenance/recovery-001/reduce.py --verify-only`.
Rebuild the post-hoc mechanism summary with `python 11_interpret.py`.

**Retention overrides the original teardown plan:** the user explicitly asked
to keep the Pod available after completion for further instructions. No delete
or stop was issued at completion. The original provider guard remains armed for
20:30:06UTC (17:30:06 America/Sao_Paulo); it stops compute while preserving Pod
files. The existing100GB shared network volume remains untouched. The ongoing
lease, budget, retention instruction and handoff state are recorded in
`prelaunch/`; expense continues while the Pod remains running.

## Subsequent compute closeout, 22 September 2026

The original guard was superseded by Run050's approved extension to22:42:28UTC.
After Run051 and the user's instruction to end the GPU session, that guard
stopped compute and the Pod was terminated. At22:43:56UTC, provider lookup
returned404 and the complete Pod list was empty. All298 Run049 output files
had already been verified locally; the pre-existing shared100GB network volume
remains untouched. See the
[terminal compute evidence](../051-2026-09-22-pythia70m-frozen-kernel-grid/results/compute-closeout.json)
and the subsequent [F003 finding](../../research/findings/F003-70m-sparse-h-gain-does-not-establish-broad-hz-exploitation.md).
