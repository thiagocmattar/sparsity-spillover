# Proposed 14M kernel benchmark extension — awaiting design confirmation

On 17 September 2026 the user deferred the 70M work and requested kernel
latency/speedup benchmarks for the new runs plus updated model-wide sparsity,
full-model speedup, and latency figures. This document makes the measurement
scope reviewable before the required design confirmation. It is not a new
numbered run, experiment implementation, or launch authorization.

## Cohort decision

The five Run 032 A7+OL1(h) final checkpoints are new to the kernel benchmark.
The five historical Run 012 A4+OL1(h) checkpoints already have Run 029 timing
records (c31–c35), but the paper's current 30-point figure excludes them.

Recommended scope: refresh all 40 checkpoints together on one physical
RTX 5090: the 30 paper checkpoints, five A4+OL1(h), and five A7+OL1(h).
This makes absolute latency and common-A0 ratios comparable within the new
measurement session. Reusing the old measurements alongside five new ones
would require labeling distinct physical-GPU sessions and retaining separate
fits/reference times. That reduced scope is offered as an explicit alternative.

## Question and fixed protocol

Do the h-only pressure endpoints translate their canonical logical sparsity
into qualified full-model runtime improvements under the existing K050 policy?
Only the evaluated trained checkpoint and the existing skip-control mode vary.
The kernel algorithm and settings are frozen; this is not another kernel search.

- Reuse the final step-712 weights, gates, thresholds, and pressure identities
  of the seed-1234 Pythia-14M random-pretraining cohorts. No training updates,
  optimizer state changes, initialization draws, or data-order changes occur.
  The six A4/A7 groups use kappa 0, 0.01, 0.05, 0.1, and 0.5 as already trained.
- One physical RTX 5090; BF16 parameters/logits; batch one, T=2048,
  uncached causal inference with all 50,304 vocabulary logits.
- Reuse Run 029's pinned Python 3.12, Torch 2.11.0+cu128, Transformers 5.12.1,
  NumPy 2.5.0 environment and frozen source archive. CUDA graphs use the same
  Run 028 forward scaffold; unmodified stock eager inference anchors correctness.
- Three unchanged K050 modes: full K050, all sparse skips disabled, and
  projection skipping enabled with attention skipping disabled. Each is paired
  with its own native SDPA CUDA-graph measurement. Three fresh processes per
  mode/checkpoint yield 360 processes for the recommended cohort.
- Timing seed 2504, runtime seed 2801, 64 fixed validation inputs, seven
  paired passes per process: 1,344 timing pairs per checkpoint/mode.
  Input-dependent packing/detection and full logits are timed; graph setup,
  compilation, and equal input staging remain separately recorded.
- Numerical checks cover all 500 validation documents through 338 complete
  blocks (692,224 input tokens, 691,886 prediction tokens), excluding the
  1,444-token tail. Preserve existing elementwise logit atol 0.25, rtol 0.02,
  relative-L2 bound 0.02, and absolute validation-loss tolerance 0.001.

Qualified improvements support a runtime benefit for these checkpoints and
this workload; slowdowns, no association, or numerical failures weaken that
claim and remain reported. A cross-checkpoint association does not isolate a
causal effect of S_model, pressure at Q/K/V, or a universally faster kernel.

## Diagnostics and outputs

Retain all raw latency pairs, process IDs/seeds, complete-validation errors
and losses, source/config/environment/checkpoint hashes, and compact failure
examples. Reuse the full Run 029 diagnostic pass once per checkpoint: exact
and near-zero activation counts (0, 0.001, 0.01), RMS/L2, weight norms,
BF16 executed-skip counters and row/fragment occupancy distributions.
Canonical S_model is independently recomputed from the retained pooled FP16
integer counters, with explicit precision and coverage. Original training
gradient/OL1-boundary records and checkpoints remain retained; inference
does not reconstruct gradient interaction. No new post-hoc clipping is proposed.

The updated figures and tables will expose both same-checkpoint native-relative
speedup and speed relative to the newly measured common native A0 reference,
absolute native/K050 latencies, and matched projection-skipping benefit. Their
reference definitions stay explicit. Analysis 021's manuscript-adopted v2 and
separate common-A0 v3 are different estimands; neither old ratio nor old A0
latency is silently substituted into the new session. The corresponding paper
section is `manuscript/draft/kernel-autoresearch.tex`, Section 4.6. Historical
figures/results remain preserved. New publication figures need their own
source reduction and observation records.

## Read-only resource check and provisional ETC

The local machine is an RTX 5070 Ti Laptop (12,227 MiB), a different GPU from
the reference study. Prior exact RTX 5090 calibration peaked near 3.07 GiB
allocated. A new RTX 5090 is proposed to retain the hardware model; its
physical identity and complete numerical qualification must be recorded.

The authenticated runpodctl 2.12.0 read on 17 September reports zero Pods and
RTX 5090 availability with low stock. Published catalog rates returned by that
read are USD0.69/hour Community and USD0.99/hour Secure. The pre-existing
network volume is not a new task resource.

Run 029's six-process calibration took 95.25 seconds including process
lifecycle. Scaling that observed reference to 360 processes gives about
95 minutes, before the different diagnostic frequency, setup/compilation,
input transfer, and retrieval. Initial overall ETC is **2–3 hours**, about
**USD1.50–3.10** including temporary disk at those quoted GPU rates. These
are planning estimates; a short exact calibration must refresh them.

After design approval, prepare the new run, source/input inventory, focused
tests and full bootstrap checks, then present a concrete launch packet with
fresh availability, exact price, cost/duration cap, persistent output paths,
independent deadline protection, monitoring, and verified retrieval/teardown.
No old Run 028/029 or Run 032 spend authorization is reused.
