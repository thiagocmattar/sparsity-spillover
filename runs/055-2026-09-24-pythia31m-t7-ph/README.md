# Run 055: 31M T7/Ph

Status: completed and retrieved on 24 September 2026. All five training
conditions and all 15 final timing processes are verified; all task Pods are
terminated. On 24 September the user confirmed the
five-condition design and explicitly requested immediate, fastest parallel
RunPod execution. This supplies design and launch authorization. The previously
approved complete retention package carries forward. No manuscript edit.

## Question and matched design

Does T7/Ph extend the full-model loss-latency frontier relative to T2/Ph across
14M, 31M and 70M? Train only the missing five 31M T7/Ph endpoints at kappa
0,0.01,0.05,0.1,0.5. Reuse existing matched Base/T2 endpoints and the retained
14M/70M T7/Ph endpoints. New nondominated points support an improved frontier;
domination weakens it. One seed, fixed budget and distinct sessions/size-specific
kernels support a descriptive scale comparison, not a general scaling law.

Pythia-31M has 30,494,720 parameters: six layers, D256, FFN1024, eight heads
of dimension32, vocabulary50304. Architecture revision, canonical random
seed1234 initialization/RNG and MiniPile/tokenizer/cache identities are copied
unchanged from Run054. Initial parameter SHA256 is
731554f7a0e11b2056650f5928fc8dc79db5b0738f49f360ae33a91d631b767e.
No released weights. Model and data seeds1234, same flattened training order.

T7/Ph exactly matches Runs032/034 and the manuscript T7/Ph assignment:
A7-Z-POST; one-sided a,m,h,z; symmetric q_post,k_post,v; equality retained;
Q/K gates follow RoPE; h replaces GELU. OL1 pressure targets only post-gate h,
the equal mean of six layer-tensor means, lambda1 and trust budget1. Task-only
AdamW moments, conflict removal in adaptive geometry, correction after AdamW.

712 updates x1024 sequences x2048 tokens =1,493,172,224 tokens per condition;
five conditions total7,465,861,120. MB4/GAS256, fused AdamW betas(.9,.95), eps1e-8,
weight decay.1 excluding bias/LayerNorm, task clip1, pre-step cosine LR.001 to
.0001 with1% warmup. Dynamic FP16, FP32 parameters/moments, flash SDPA, no dropout
or activation checkpointing. Python3.12/Torch2.11.0+cu128/Transformers5.12.1.
The existing Transformers mapping is retained, not a bitwise NeoX reproduction.

Full validation after step1 and final checkpoint reload covers500 documents,
338 complete2048-token blocks,692224 input tokens,691886 predictions and the
excluded1444-token tail. Complete final activation and eager logical passes
use the same coverage. Pool integer counts before division.

## Retention

Retain model steps0,1,2,4,8,16,32,64,128,256,512,712; optimizer/scaler and
Python/NumPy/Torch CPU/all-CUDA RNG recovery at256,512,712. This gives60 models
and15 recoveries, approximately11GB. Retain all712 gradient/OL1/capture/clip/
overflow boundaries, all named parameter norms, exact/near-zero counts at
0/.001/.01, RMS/L2 and moments at all eight declared sites, six-operation
logical-product counts, and the analytic architecture ceiling. T7 reach is
16109273088/42483056640 products per full sequence (37.919289%). This is analytic
reach, distinct from measured zero products and runtime. No clipping sweep.

## Kernel and timing

Copy the Run054 D256 specialization of Run045 opt073, preserving its fixed
component policy. Pass a/m gates into paired LayerNorm before replacing the
standalone gates. Keep native dense a/m projections, symmetric post-RoPE Q/K/V
gates, dense Flash attention, short-row h/z scalar work with MMA fallback and
the full-vocabulary CUTLASS head. No new optimization search.

RTX5090, BF16, B1/T2048/full logits, CUDA graphs,64 fixed validation inputs x7
paired passes x3 fresh processes per checkpoint. Measure both native and kernel;
plot T7 kernel and the existing Base kernel/PyTorch references. Every process
qualifies all338 blocks: atol.25,rtol.02,relativeL2<=.02,pooled loss delta<=.001.
Replicate1 retains untimed runtime activation moments, row occupancies and
independently checked issued/bypassed MMA and scalar products. Full-model
qualification and explicit fused-gate component tests precede final timing.

## Execution and output

Use five independently preflighted training GPUs concurrently, preferring fast
available H200/H100 capacity, plus a separate RTX5090 timing GPU. Jobs/logs and
checkpoints persist under /workspace and survive SSH disconnection. Read-only
training monitoring every5min; setup/timing every1min; sooner for completion or
nonfinite/skipped updates, stale events10min, missing process, source/cache
drift, disk pressure, or budget/deadline danger. Stop instead of deleting if
artifacts remain unretrieved. Verify archive and every file locally before Pod
deletion, then audit resources. Existing stopped Pod/shared volume are untouched.

Analysis038 will contain the scale figure: Base(kernel/PyTorch), T2/Ph, T7/Ph
at14M/31M/70M,36 execution points from33 checkpoints. Historical coordinates
remain unchanged. PDF only, exact source data, observation and index. No claim
promotion or manuscript edit is authorized. Live allocation, test/ETC evidence,
deadline/cost envelope and transfer receipts belong under prelaunch/.

## Execution notes

Initial training allocation was five H200 GPUs across three Pods. Two GPUs on
the US-CA host measured about18s/update; no scientific run started there. Their
conditions moved to independently preflighted single-GPU replacements. The
US-NC replacement measured5.65s/update. The Iceland replacement measured10.46s,
so its three-update attempt was retained separately and restarted canonically
on an already freed US-NC GPU at5.62s/update. No scientific input was changed.
Interrupted attempt files retain their original state under
`artifacts/retired-attempts/`; they are excluded from the completed cohort.

RTX5090 allocations001/003/004/006 never exposed a runtime or SSH endpoint and
received no task files. All were deleted and absence verified. Several other
requests failed for lack of capacity without creating a Pod. Allocation011
booted on a distinct CUDA12.8 Community host at17:35UTC, at$0.69/hour. It is the
dedicated timing host. The common hard stop is21:43:56UTC; the total billable
envelope is$160, with artifact-preserving stop guards.

An extra idle-H200 check passed34 component cases (24 paired-normalization and
10 joint arithmetic/counter cases). The earlier58-case label double-counted
normalization branches. A full338-block H200 check of the trained kappa0 model
failed the elementwise tolerance despite a pooled loss difference of2.61e-5.
Substitution on failed block15 isolated the difference to custom Flash attention:
native attention, or fused RoPE with native SDPA, restored bitwise-equal logits;
native normalization, RoPE alone, joint projections or head did not. These are
non-evidence diagnostics. Final qualification remains on the prescribed RTX5090,
with the original tolerances and unchanged fixed kernel policy.

On RTX5090 the34 component cases and four initial checkpoints passed. Initial
kappa0.5 failed only block323 (max logit difference0.569824, relativeL2.004596,
pooled loss difference1.02e-7). Native-joint or MMA-only substitution removed
the difference; it arose in the short-row path. The actual trained kappa0.5
checkpoint then passed all338 blocks with bitwise-equal logits. The additional
all-initial-checkpoint launch gate was replaced by full qualification of the
endpoint that is actually measured; all failed preflight evidence is retained.
`prelaunch/kernel-preflight-acceptance.json` records the scope explicitly.
Kernel code, numerical bounds, workload and final15-process qualification were
unchanged. Qualification is checkpoint-specific, not a universal claim of exact
short-row equivalence for arbitrary weights and inputs.

## Verified results and closeout

All five conditions completed 712 updates without skipped optimizer steps.
The complete local `03_verify.py` cohort check passed, including source identity,
initialization, data order, validation coverage, every gradient boundary,
checkpoint cadence, integer-count aggregation and retained artifact hashes.
The launch checks were 10 focused tests, 242 bootstrap tests and actual cloud
lifecycle smokes at kappa 0 and 0.5; see `prelaunch/tests.json` and retained
preflight artifacts. Median training throughput ranged from 366,310 to 387,390
input tokens/s. The final condition completed by 18:25:45 UTC.

| Kappa | Validation loss | Kernel latency (ms) | Native latency (ms) | R_model |
| --- | ---: | ---: | ---: | ---: |
| 0 | 4.894259 | 0.986853 | 1.076050 | 0.153365 |
| 0.01 | 4.874077 | 0.985217 | 1.191179 | 0.157420 |
| 0.05 | 4.893721 | 0.895587 | 1.190956 | 0.167749 |
| 0.1 | 4.942717 | 0.800644 | 1.189999 | 0.175327 |
| 0.5 | 5.520965 | 0.606869 | 1.190163 | 0.226573 |

Loss is canonical final-checkpoint FP16-autocast validation. Latencies are
BF16 CUDA-graph geometric means over three fresh processes, each with 448
paired samples per backend. All 15 processes passed the unchanged full
338-block qualification on one physical RTX 5090 and one kernel source identity.
All five replicate-1 runtime diagnostic and independent work-counter checks
passed. `R_model` is a logical-product opportunity, not a measured speedup.

The retained completed cohort contains 60 model checkpoints and 15 recovery
states, totaling 10,979,972,348 checkpoint bytes, plus the agreed per-update,
activation, weight, logical-product and kernel-work evidence. Calibration,
interrupted-attempt and failed preflight evidence is retained separately.
The final seed-Pod download hit Windows' path-length limit during extraction;
the helper retried with extended filesystem paths and verified the same sealed
archive and every file before teardown. No scientific input or result changed.

The original 21:43:56 UTC deadline was sufficient; the user's extension
authorization was not needed. All ten allocated Pods, including empty capacity
retries, are confirmed absent. The pre-existing stopped Run052 Pod and shared
100 GB volume were left untouched. The conservative creation-to-confirmed-
absence compute rental estimate is $55.44, below the $160 envelope. This is
not an invoice and does not infer storage charges or account credits. See
`prelaunch/compute-closeout.json`, the individual verified transfer/termination
receipts and `prelaunch/final-resource-audit.json`.

The [Analysis038 scale comparison](../../analyses/038-2026-09-24-base-t2-t7-scale-frontier/README.md)
contains the PDF, exact 36-point data and observation. At this measured resolution,
the new kappa 0.05 and 0.1 points extend the cross-scale Pareto frontier.
Historical endpoints are unchanged; no scaling law is fitted.
