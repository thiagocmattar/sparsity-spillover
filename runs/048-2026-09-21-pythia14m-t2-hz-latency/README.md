# Run048: causal h/z latency control at the 14M T2/Ph headline point

Status: implementation and prelaunch verification. The user explicitly authorized
both design and execution on 21 September 2026. No training or manuscript edit.

## Question and matched comparison

How much latency does exploiting h/z zeros save at Run041's existing final
Pythia-14M T2/Ph kappa=0.1 checkpoint? A enables both paths; B disables h;
C disables z; D disables both. All weights, thresholds, biases, residuals,
attention, input projections, output head, fusion and tensor shapes remain fixed.
One-sided h/z thresholding remains inside every forward. h replaces GELU;
z is immediately before W_o. Equality survives. Ph is the checkpoint's h-only
orthogonal L1 training history; no pressure optimization happens at inference.

Reuse Run037's independently controlled joint kernel, byte for byte. Disabling
a site's zero exploitation disables both empty-tile skipping and the short-row
scalar substitution, retaining its dense matrix path within the same fused
kernel. Thus this measures the benefit of the implemented sparse h/z paths,
including their inspection overhead, rather than a hardware cost per zero.
The untouched Run041 K050 is an additional qualification control.

Three fresh processes each time all four modes, untouched K050, and same-checkpoint
native PyTorch/SDPA in randomized paired order. Native timing is a supporting
engineering reference, not the ablation denominator. Use B-A and C-A for
conditional savings, D-A for the joint saving, and D-B-C+A for interaction.
Retain signed effects. These are not additive allocations. Lower qualified A
latency than D supports a sparse-path contribution; a zero/negative difference
or changed zero patterns weakens or invalidates that claim.

## Identity and coverage

The unchanged Run041 c03 step712 checkpoint came from random initialization,
model/data seeds1234, 1,493,172,224 training tokens, one-sided HZ gates and h-only
OL1 with lambda=1 and trust budget=1. Six layers, width128, FFN512, four heads,
vocabulary50304. Source configuration, checkpoint hashes, complete canonical
validation loss/logical counts and training provenance are retained under
provenance/. Earlier checkpoints, optimizer states and gradient/OL1 histories
remain in Run041. This run performs zero optimizer steps.

Pinned MiniPile/tokenizer cache: 500 validation documents,338 complete2048-token
blocks,692224 input tokens,691886 predicted tokens,1444 excluded tail tokens.
Cache SHA256:51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451.
All modes use this same checkpoint and cache, BF16 inference, batch1,length2048,
all50304 logits, no KV cache. RTX5090; Python3.12,Torch2.11.0/CUDA12.8,
Transformers5.12.1,NumPy2.5.0; disable TF32 and reduced-precision BF16 reduction.
Runtime seed2801,timing seed2504. No post-hoc clipping.

Each process uses64 fixed validation inputs x7 paired passes. Timings include
the full CUDA-graph forward, including all gates and input-dependent checks;
exclude compilation, weight preparation, graph capture and equal input staging.
Retain raw host/CUDA timings, pooled geometric means and process ranges.
All338 blocks qualify against unmodified eager logits in every process using
the original bounds:atol.25,rtol.02,relative L2<=.02,pooled loss delta<=.001.
Also check exact cross-mode agreement and untouched K050 agreement explicitly.

Untimed diagnostics on all338 blocks in replicate1 retain h/z post-threshold
value and zero-mask hashes for every layer, exact/near-zero counts at0/.001/.01,
RMS/L2, weight norms, row occupancy, logical opportunities and instrumented
kernel counters. Require identical gate values/masks across A-D and verify
zero bypass/scalar counts at disabled sites. These diagnostics are the package
the user previously confirmed sufficient; the checkpoint stays local throughout.

## Execution and retention

One Secure RTX5090 at the live quote0.99 USD/hour. No active Pods existed at
discovery. Use the pinned official image recorded in the launch lease,30GB
container and40GB persistent /workspace; leave the existing100GB shared volume
untouched. Maximum90 billable minutes and USD2 including temporary storage,
with20 minutes reserved for collection. Expected cloud ETC30-60 minutes.
Local CPU tests cannot qualify CUDA performance; local hardware differs from
the published RTX5090 workload.

Run GPU control tests and an8-block smoke before the three full processes.
Persist detached logs/manifests under /workspace, with a remote workload
deadline and independent workstation provider-stop guard. Monitor every5min,
sooner near projected completion or upon numerical/source failures, less than
8GiB VRAM headroom, less than5GB free disk, stale progress for10min, or a
projected deadline overrun. A deadline stop preserves unretrieved artifacts.
Retrieve every raw result/log/diagnostic and verify archive and member hashes
before deleting the run-owned Pod. Report any remaining storage and cost.
