# Run053: local 70M calibration for high-sparsity kernel development

Status: implementing the local calibration authorized by the user's "Go on"
after WSL setup. The high-kappa-first direction was already approved. This
stage checks resource fit and frozen reference implementations; it does not
declare a new kernel winner or change the manuscript.

## Question and fixed inputs

Can the RTX 5070 Ti Laptop execute the existing full-sequence 70M workload
with headroom, and does the adapted kernel retain its strong high-kappa
response on this development device? Compare Base c00 and existing T2/Ph
checkpoints c24/c25/c26 at kappa .05/.1/.5. Prior random initialization,
seed1234, step712 weights, 1,493,172,224-token training histories, AdamW and
h-only orthogonal-L1 histories are unchanged. No training or new threshold.
T2 is operational HZ: one-sided gates at h and z, equality survives, h replaces
GELU, z is immediately before W_o. Preserve BF16 branches and residual rounding.

BF16, FP32 accumulation, B1/T2048, all50304 logits, TF32 and reduced-precision
BF16 reductions disabled. Runtime seed2801 and timing seed2504. Frozen modes:
native PyTorch, original K050-derived 70M port (`legacy`), optimized adaptation
`opt073`, its same-layout h/z skip-disabled control, and `native_hz` (the
opt073 scaffold with native dense h/z projections). Skip disabling retains
gates and weights. These are calibration references, not a complete search
over the strongest dense kernels.

The immediate smoke uses the first four blocks of the retained training
development cache, two randomized paired timing passes and one fresh process
per checkpoint. It establishes neither validation quality nor significance.
The retained cache contains64 blocks despite its older description saying
"first16"; four explicit block indices are recorded here.

The separate full-reference mode uses all500 validation documents:338 complete
blocks,692224 input tokens,691886 predicted tokens,1444 excluded tail tokens.
It retains64 timing inputs and seven passes, but only one process per cell;
it is still local calibration, not the final replicated manuscript comparison.
Neither stage selects a kernel on validation. Bounds remain per-logit
atol.25+rtol.02, per-block relativeL2<=.02 and pooled loss delta<=.001.

## Attribution and next decisions

The manuscript goal remains both moderate endpoints faster than native Base,
a larger positive .05-to-.1 latency drop than14M, and sparse contribution
beyond optimized dense execution. This run alone cannot establish that claim:
it is a separate device cohort and initially contains no matched14M matrix.
High-kappa acceleration plus a controlled skip benefit is a positive control;
failure at moderate kappa diagnoses this family/device, not impossibility.

Prioritize the high-kappa bottleneck, then reduce thresholds while keeping
.05/.1 in the evaluation loop. Any next candidate is appended with immutable
source/configuration identity before execution. Preserve original references;
do not slow .05 merely to enlarge the latency drop. No TeX changes authorized.

## Diagnostic inventory and local execution

The previously approved inventory persists: all checkpoint/cache identities,
full-validation exact/near-zero counts, RMS/L2, weight norms, row/group/tile
occupancy, h/z logical and executed-work counters, separate profiles, raw
timings, compiler evidence and numerical failures. Source checkpoints and
their existing full-validation diagnostics remain retained. The initial smoke
collects timing, correctness, memory and profiles only; it does not relabel
four blocks as full diagnostics. Training gradient interactions cannot be
reconstructed from these checkpoints.

`01_stage.py` copies an explicit hash-checked inventory into the Linux filesystem
through stdin tar, without mounting Windows drives. No prior run is modified.
`02_calibrate.py` owns each attempt, logs progress and checks GPU/runtime identity.
`03_local.py` launches one hidden persistent Windows WSL client, executes
checkpoints sequentially with a bounded timeout, and retrieves/hash-verifies
artifacts. Existing cloud resources are not touched.

Initial smoke budget: maximum20minutes, estimated5--12minutes including cold
CUDA compilation. Only smoke is launched initially. Monitor every60seconds;
warn/stop on numerical failure, less than1.5GiB free after capture, OOM,
nonfinite loss or timeout. Report measured full-reference ETC after the smoke.
No model, compiler or benchmark process remains running after this stage.

## Verification

Implementation checks and actual local smoke results will be appended after
execution. Environment readiness is recorded separately in
[tools/local_gpu](../../tools/local_gpu/README.md).
