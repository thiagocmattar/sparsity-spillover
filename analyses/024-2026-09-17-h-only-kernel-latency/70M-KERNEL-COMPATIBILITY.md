# 70M final-kernel figure: compatibility and evidence audit

Audited 18 September 2026 in response to the request for a 70M version of
Figure03, using only retained checkpoints. This is a source/artifact audit and
a proposed measurement design, not an executed kernel port or benchmark.

The user subsequently approved this design and explicitly authorized RunPod.
[Run035](../../runs/035-2026-09-18-pythia70m-k050-port/README.md) owns the port,
qualification and new measurement. Run035 is now complete: all22 checkpoints
qualify, with a [new figure and caption](observations/004-70m-final-latency-topology.md).
The availability statements below describe the pre-port audit.

## What is available

All 22 trained final endpoints are retained in Runs018/034 and reduced in
[Analysis025](../025-2026-09-17-70m-quality-sparsity/README.md):

| Family | Kappas | Checkpoints | Final K050 latency measurements |
|---|---|---:|---:|
| Baseline (A0, GeLU) | N/A | 1 | 0 |
| 1-site (A1-H, ReLU) | N/A | 1 | 0 |
| A4 + OL1(all) | 0, .01, .05, .1, .5 | 5 | 0 |
| A4 + OL1(h) | 0, .01, .05, .1, .5 | 5 | 0 |
| A7 + OL1(all) | 0, .01, .05, .1, .5 | 5 | 0 |
| A7 + OL1(h) | 0, .01, .05, .1, .5 | 5 | 0 |

There are no completed pressure-free A4/A7 grids or A1-H+OL1 grid at 70M.
No retraining is necessary for the proposed 22-point measurement.

Older 70M timings exist in
[Analysis015](../015-2026-09-06-pythia-agentic-kernel-search/README.md), using
K016 on RTX PRO4500 Blackwell, with H100 NVL sentinel measurements. The primary
replication reduction has 12 70M checkpoints, of which nine qualify. A4+OL1(all)
at kappa .01, .05 and .1 fail the numerical gates. None of the ten new h-only
checkpoints has a latency measurement there. That campaign used eager timing,
16 training-cache inputs and five passes, unlike Figure03's CUDA-graph timing
with 64 validation inputs and seven passes. Its kernel, hardware and timing
estimand differ; those measurements cannot be presented as 70M K050 results.

## Why unchanged K050 cannot execute 70M

The checkpoint architectures have the following dimensions:

| Dimension | 14M | 70M |
|---|---:|---:|
| Hidden width | 128 | 512 |
| FFN intermediate width | 512 | 2048 |
| Attention heads | 4 | 8 |
| Per-head width | 32 | 64 |
| Layers | 6 | 6 |
| Vocabulary | 50,304 | 50,304 |

These are implementation restrictions, not just tuning choices:

- [K050 normalization](../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k050/candidate.py)
  rejects LayerNorm shapes other than `(128,)` before installing the kernel.
  Its [CUDA source](../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k050/norm.cu)
  also fixes indexing, normalization and affine shapes to width128.
- [K049 output projections](../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k049/candidate.py)
  reshape h to width512 and z/residual to width128.
- [K042 input projections](../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k042/candidate.py)
  reshape input activations to width128.
- [K035 attention](../../runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/candidates/k035/candidate.py),
  inherited through K042, accepts only `(B,H,T,D)=(1,4,2048,32)` and fixes its
  intermediate buffer shapes accordingly. The 70M shape is `(1,8,2048,64)`.

All five linked source files were byte-compared with the frozen Run033 archive;
all match. Removing the Python guards alone would not port the CUDA kernels.
The inherited RoPE and other installation paths also need shape review before
execution; this audit is sufficient to prove incompatibility, not a complete
implementation specification.

The gate/pressure topology does not require a new sparsity definition. A4/A7
and their OL1(h)/OL1(all) variants can use one inference implementation with
their own retained weights and gates. A shape port is required; an open-ended
kernel search is not a prerequisite. Performance after a correct port is
unknown, and any later optimization search should be a separate decision.

## Approved measurement design (original proposal)

Question: how does final full-model latency vary with model-wide sparsity across
the 22 available 70M endpoints when using a correctly qualified K050-derived
implementation? A derived implementation must receive a new identity; it is
not byte-identical K050 and does not imply equal optimization maturity to 14M.

Port the required shape-specific code and preserve the existing gate semantics,
BF16 precision, numerical bounds and final policy (`round_p=False`,
`shortcut=False`). Qualify representative control/A4/A7 endpoints before the
full grid. Freeze one implementation for the grid; retain failures rather than
selecting a different kernel for each checkpoint. No kernel search is proposed.

Reuse all final step712 checkpoints, preserving the matched seed1234 random
initialization, realized data order and 1,493,172,224-token training budget.
Training optimizer and weights stay unchanged; this is inference only. A4 gates
are one-sided at a,m,h,z; A7 additionally gates q_post,k_post,v symmetrically.
OL1 was trained at h or at all gated sites, with lambda=1 and b=1. Baseline and
ReLU retain their original unpressured definitions.

Use the Run033 measurement contract: one RTX5090, BF16, batch1, T=2048, uncached
causal inference, full 50,304 logits, pinned runtime, runtime seed2801 and timing
seed2504. Pair native SDPA and candidate CUDA graphs in each fresh process.
Use 64 fixed validation inputs, seven paired passes and three fresh processes
per checkpoint: 66 processes and 1,344 timing pairs per checkpoint.

Every process must qualify all 338 complete blocks from all 500 MiniPile
validation documents; retain the 1,444-token excluded tail. Preserve logit
atol=.25, rtol=.02, relative-L2<=.02 and absolute loss difference<=.001.
Qualification failures preclude reporting a valid candidate speedup. Report
absolute latency as the geometric mean of raw host times, and speedup as the
geometric mean of paired native/candidate ratios, matching Figure03's source.

Retain raw timings, validation/logit checks, failures, timing-input indices,
source/checkpoint/runtime identities, setup/compilation times and peak memory.
Carry over the existing exact/near-zero counts, activation RMS/L2, weight norms,
logical/skip counters and occupancy diagnostics after checking their 70M shape
support. Reuse canonical FP16 integer-pooled S_model for the x axis; distinguish
it from BF16 execution diagnostics. Retain all original checkpoints. Training
gradient interactions cannot be reconstructed by this inference experiment.
Confirm additional post-hoc measurements before launch; no new clipping sweep
is currently proposed.

The figure would retain four topology legend entries, data-fitting y limits,
subtle pressure annotations and four separate dashed five-point OL1 curves.
A0/ReLU would be isolated markers. No unavailable no-pressure curves would be
invented. The manuscript connection is the measured-runtime scale discussion
and kernel case study; no manuscript changes are proposed. Qualified lower
latencies support a runtime benefit for this workload; flat trends, slowdowns
or numerical failures limit it. Sparsity alone does not establish causation or
guarantee speedup, and the 14M/70M runs use different weights and kernel shapes.

No GPU was provisioned and no benchmark ran during this initial audit. Following
the user's approval, Run035 implemented, qualified and executed this design
within its stated USD10/eight-hour envelope. The final figure and full evidence
are linked above; this original audit is retained as the pre-port record.
