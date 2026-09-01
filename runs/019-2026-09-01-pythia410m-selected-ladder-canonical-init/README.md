# Run 019 - Pythia-410M selected ladder with canonical initialization

## Status

Implemented and locally verified on 2026-09-01. The canonical initialization
has been generated and independently strict-loaded twice. No calibration Pod,
scientific attempt, or other Run 019 cloud resource has been created. GPU type,
cloud tier, calibration deletion guard, scientific deletion guard, and maximum
cost remain intentionally unset pending a separate calibration-launch approval.

## Question and matched design

Does the relationship between complete-validation loss and measured logical
product opportunity (`R_model`) observed at Pythia-14M and Pythia-70M persist
for the selected ladder at randomly initialized Pythia-410M?

The twelve independent seed-1234 conditions are:

- A0/GELU and A1-H/ReLU;
- A4-OL1 at `kappa` 0, 0.01, 0.05, 0.1, and 0.5;
- A7-OL1 at the same five `kappa` values.

All conditions use the checked-in Pythia-410M architecture: 24 layers, hidden
width 1,024, FFN width 4,096, 16 heads, vocabulary 50,304, 405,334,016 total
parameters, and 302,311,424 non-embedding parameters. They share one canonical
FP32 initialization artifact, its post-initialization CPU RNG state, the same
712-boundary order over 1,493,172,224 MiniPile input tokens, and global batch
1,024 as microbatch 4 times 256 accumulation. AdamW uses betas `(0.9,0.95)`,
epsilon `1e-8`, weight decay 0.1, task-gradient clipping at 1.0, dynamic FP16,
3e-4 peak LR, 3e-5 minimum LR, one-percent warmup, cosine decay, and final
recovery state. Dropout is zero.

The official 410M GPT-NeoX recipe enables activation checkpointing. Run 019
keeps it disabled because the approved comparison matches the executed 70M
Transformers mapping and its exact OL1/capture path. The calibration must prove
that this exact workload fits; an OOM does not authorize changing microbatch,
checkpointing, or the intervention.

This is the approved ladder promotion, not a broad larger-model ablation. The
implementation directly reuses only the verified architecture-independent Run
004 lifecycle and shared scientific primitives. It does not import Runs 017 or
018, and it locally defines the 410M architecture, initialization, capture,
diagnostic, verification, and calibration adapters.

## Validation and diagnostics

Step 1 and the reloaded final checkpoint evaluate all 500 MiniPile validation
documents: 338 complete 2,048-token blocks and 692,224 input tokens, with the
1,444-token tail excluded and reported. Counts are pooled as integers.

The canonical endpoint pairs validation loss and `R_model` from the same eager
logical-diagnostic forward configuration. The ordinary SDPA/flash final loss is
retained as an execution diagnostic but is not the plotted endpoint loss.
Activation statistics also run under eager attention and cover `a`, `m`, `h`,
`q_post`, `k_post`, `v`, `z`, and attention output at every layer, including
exact/near-zero counts and RMS/L2 accumulators. Weight statistics, full logical
integers, analytic topology ceilings, and all boundary-level OL1 interaction
and trust-budget fields are retained. Observed `R_model` is validated against
`[0,1]`; it is not incorrectly bounded by selected-topology `R_model_max`.

A0 and A1-H each receive the ten-target post-hoc TEAL sweep over clipping sites
`a,m,h,z`, calibrated from the first ten source-order complete training blocks.
Each point evaluates all 338 validation blocks and records passive downstream
statistics at all eight diagnostic sites, including Q/K/V and attention output.

Final checkpoints retain model, optimizer, loss scaler, schedule identity, and
Python/NumPy/Torch CPU/CUDA RNG states. Gradient-interaction measurements are
captured during training and are not deferred.

## Initialization identity

- model artifact: 1,621,370,392 bytes, SHA-256
  `edba25fa7535bef6469228afe7d60db741a1a4ecd6c0c4362989e0b45a8cac51`;
- RNG artifact: 14,830 bytes, SHA-256
  `38c9f36a2e259018c183039771c5732d8aa5a5f63b4363d9e6403e6683a9ef43`;
- metadata artifact: 2,320 bytes, SHA-256
  `b909c5c578ee264127097b7c2f226cabf982de45b9033dc2d9f8e413ae174523`;
- strict-load parameter SHA-256:
  `76217bf2ef13de2377c7750515a559cb71f574c274476e08408a5f7749ea9cff`;
- 292 tensors and 1,621,336,064 tensor bytes.

`08_verify_initialization.py` constructs two fresh CPU models, verifies the
artifact bytes before each load, strict-loads every tensor, recomputes the
parameter hash, and proves identical restored RNG probes. Remote setup runs the
same check before calibration. CUDA seeding happens before artifact load; the
pinned CPU RNG state is then restored. Each worker hashes the strict-loaded CPU
parameters, moves the model to CUDA, hashes the CUDA round trip again, and only
then creates the optimizer.

## Analytic ceilings at sequence length 2,048

The model denominator is 827,099,971,584 products per complete sequence.

| Topology | Reachable numerator | `R_model_max` |
| --- | ---: | ---: |
| A0 | 0 | 0.000000% |
| A1-H | 206,158,430,208 | 24.925455% |
| A4-Z | 618,475,290,624 | 74.776365% |
| A7-Z-POST | 721,604,837,376 | 87.245177% |

These are topology-conditioned analytic reach ceilings, not observed sparsity
or runtime speedup.

## Calibration gate

Before scientific launch, at least two candidate GPUs will run the same
non-evidence calibration. The preferred cost--speed comparison is RTX A6000
against A100 SXM 80 GB, with A40, L40S, or A100 PCIe available if live stock
changes. Each candidate
measures exact A0, A4 `kappa=0.5`, and A7 `kappa=0.5` training boundaries. The
first of five boundaries is excluded as warm-up; the remaining four include
batch construction, CPU-to-GPU staging, forward/backward work, gradient
processing, and optimizer update. Complete validation, both eager diagnostics,
all-parameter weight statistics, checkpoint write/hash/reload, TEAL
calibration, and a complete TEAL point are timed independently. The ETC
projection includes the repeated checkpoint inventories performed by training
closeout and TEAL source verification.

The comparison reports projected GPU-hours, total GPU cost, per-condition ETC,
twelve-way makespan, hourly fleet burn, peak reserved VRAM, and remaining
headroom. Provisioning, package installation, 7,591,009,171 bytes (7.070 GiB)
of inputs, and result download are measured separately and added to the launch
envelope. A candidate fails if an exact boundary overflows/skips, a required
diagnostic is incomplete, or peak reserved VRAM exceeds 90% of visible memory.
The comparison script reports the cost/time Pareto set but never selects a GPU.

## Interpretation

Support means the promoted endpoints and TEAL frontiers retain an informative
loss--`R_model` tradeoff and the A4/A7 ordering remains meaningfully comparable
across 14M, 70M, and 410M. Loss collapse, negligible measured opportunity, or a
qualitatively reversed frontier would refute that expectation. This remains
one-seed descriptive evidence, not a scaling law, causal mediation result, or
runtime-speedup claim.

The operational contracts in `research/METHODS.md` and `research/METRICS.md`
execute. The result may affect the manuscript's proposed scale-persistence
figure/table only after evidence is complete and the user approves
consolidation.

## Launch boundary

See `DEPLOYMENT_PLAYBOOK.md`. Calibration is billable and still requires an
explicit launch approval containing refreshed live price/stock, exact Pod
definitions, maximum duration/cost, transfer plan, ten-minute monitoring, and
teardown. Scientific execution requires another decision after calibration,
GPU selection, config amendment, and refreshed launch approval.
