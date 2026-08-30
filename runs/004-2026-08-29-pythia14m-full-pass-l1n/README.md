# Run 004 - Pythia-14M full MiniPile pass with ReLU L1N

## Status

### Closed state - authoritative

Run 004 is **closed** as of 2026-08-30 with evidence label `valid`. All six
conditions completed, all remote transfer inventories and the local
cross-condition verifier passed, all retained checkpoints are local, and the
two requested post-hoc figures and their observation records are complete. The
results remain descriptive observations: no consolidated finding was approved
and no manuscript TeX was changed.

A live RunPod audit at `2026-08-30T10:17:58Z` returned zero Pods and therefore
zero active GPU cost. It returned exactly one intentionally retained resource:
100 GB Standard network volume `sparsity-spillover-shared` (`9luykg5yc3`) in
`EUR-IS-1`, continuing at `$7/month`. The now-complete Pod billing history has
24 rows totaling `$21.5213452158`, including the two RTX 4090 preflight Pods,
the five A100 production Pods, and Pod disk. This posted total supersedes the
earlier reconstructed `$17.67--$17.80` estimate. Network-volume billing posted
through the closeout audit totals `$0.1361111160` and continues accruing.

The latest Run 004 focused module passes 14/14, including the count-first figure
reductions. At implementation time, the then-current 18 focused checks and
90-test bootstrap suite passed; after retrieval, the then-current repository
suite recorded 107 passes and one unrelated stale Run 007 launch-state failure.
Those historical test counts are retained below with their timestamps rather
than presented as one current suite size. The machine-readable closure is
`artifacts/closeout.json`.

### Chronological deployment record

The first stage-one creation request at 16:09 -03:00 was rejected because no
Community RTX 4090 instance matching the CUDA 12.8 specification remained
available. A post-request inventory confirmed zero Pods and zero network
volumes, so that request incurred no billable cost. The user subsequently
authorized continued capacity retries and GPU substitution. Community capacity
returned, and Pod `ottlrd7awawf8h` began the exact preflight preparation at
16:12 -03:00 for `$0.34/GPU-hour` with a seven-hour termination guard.

Two local-to-Pod uploads of the 5.97 GB token-cache archive became unusably slow
after reaching 74% and 91%; both incomplete copies were removed. The approved
replacement downloaded MiniPile directly from Hugging Face at the pinned
dataset and tokenizer revisions. It reproduced both caches exactly. Validation
has 500 documents, 693,668 tokens, and SHA-256
`51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451`;
train has 1,000,000 documents, 1,491,711,416 tokens, and SHA-256
`da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c`.

At the user's request, a reusable 100 GB Standard network volume now exists as
`sparsity-spillover-shared` (`9luykg5yc3`) in `EUR-IS-1`. It costs `$7/month`,
survives Pod deletion, and is intentionally retained after Run 004. RunPod
network volumes constrain Pods to their data center and require Secure Cloud.
Pod `qq6u7wif86vdyk` therefore stages the shared source/environment and durable
cache at `$0.74/GPU-hour`, also with a seven-hour guard. The immutable master
cache will live on the volume, but each worker will copy it to disposable local
container storage before training to avoid network-volume random-read latency.

A read-only audit at `2026-08-29T20:07:49Z` found no training or transfer
process running. The Community Pod's CUDA preflight failed before an optimizer
boundary because PyTorch could not initialize a GPU that `nvidia-smi` could
see. The Secure Pod sees CUDA and flash SDPA and is the next authoritative
preflight candidate.

Approved execution Step 1 completed at `2026-08-29T20:32:59Z`. The train cache
was atomically published under its final filename with the exact transferred
metadata. The normal Run 004 loader verified both full files, and the full-pass
schedule reproduced shape `712 x 32 x 32`, 729,088 scheduled blocks, 714 wrapped
blocks, and SHA-256
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`.
The Community cache-build, transfer, setup, and failed-preflight evidence was
copied to persistent storage and into `prelaunch/attempt-community-r2/`; both
SHA-256 inventories pass. No scientific attempt, optimizer update, validation,
checkpoint, or four-Pod scale-out has started. The complete status, incident
timeline, cost reconstruction, lessons, and remaining gates are recorded in
`RUNPOD_STATUS_2026-08-29.md`.

Approved execution Step 2 completed at `2026-08-29T20:38:42Z`. The temporary
pod-to-pod authorization was removed from the Secure Pod, the corresponding
private/public key files were removed from the Community Pod, the revoked key
was rejected, and the normal account key remained usable. Community Pod
`ottlrd7awawf8h` was then terminated: deletion returned 204 and a follow-up get
returned 404. Exactly one Pod remains, Secure RTX 4090
`qq6u7wif86vdyk` with volume `9luykg5yc3`, at `$0.74/hour`. The next approved
action required a separate Step 3 confirmation and was the exact GPU preflight.

Approved execution Step 3 completed at `2026-08-29T20:45:59Z`. Copying the
5.97 GB immutable cache from the shared volume to disposable container storage
took 14 seconds. The normal loader verified both local SHA-256 identities and
reproduced schedule shape `712 x 32 x 32` and schedule SHA-256
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`.
The exact MB32/GAS32 preflight then failed CUDA memory fit on the healthy Secure
RTX 4090 for both ReLU control and ReLU lambda 1. The control already had 20.00
GiB allocated when an operation requested another 12.28 GiB with only 3.03 GiB
free. No complete optimizer boundary or timing was produced. The evidence is
retained in `prelaunch/attempt-secure-preflight-r1/` and on the persistent
volume. The hard preflight gate therefore remains closed: no additional Pod and
no scientific training has been launched. A GPU-capacity or approved batch-
decomposition decision is now required.

Approved execution Step 4 completed at `2026-08-29T20:54:24Z`. Secure Pod
`qq6u7wif86vdyk` was terminated after its persistent preflight evidence had been
verified locally and on the shared volume. Deletion returned 204, a follow-up
read returned 404, and the account Pod inventory is now empty. The separate
100 GB Standard volume `9luykg5yc3` remains present in `EUR-IS-1` with the exact
cache, environment, source, and infrastructure evidence. Active GPU cost is
therefore zero; the intentional volume continues at `$7/month`. Scientific
training remains unstarted.

User-authorized Steps 5-6 removed the `EUR-IS-1` placement restriction and
allowed GPU substitution without changing the scientific configuration. Secure
A100 PCIe 80 GB Pod `zqet1vl0hzsxa0` was created in `CA-MTL-3` at
`2026-08-29T21:02:05Z` for `$1.39/hour`, with the pinned image, 30 GB
container disk, 25 GB persistent Pod volume, direct SSH, and a three-hour
termination guard. The pinned Hugging Face path rebuilt the cache once on that
Pod and reproduced the exact train and validation identities. Train
tokenization and publication took 1,034.9 seconds; later Pods are to receive
the immutable 5.97 GB cache and verify its SHA-256, not retokenize MiniPile.

The unchanged MB32/GAS32 exact preflight passed on that A100 at
`2026-08-29T21:36:20Z`. ReLU control boundaries took 6.384 and 4.806 seconds;
ReLU lambda 1 boundaries took 6.879 and 5.728 seconds. Neither path overflowed
or skipped an optimizer update. Peak reserved memory was 56.941 and 56.953 GiB
against the actual A100 90% limit of 71.325 GiB. The locally retrieved evidence
is in `prelaunch/attempt-a100pcie80-r1/` and its principal hashes match the
remote inventory. The preflight originally carried a 24 GiB 4090 reporting
ceiling; its infrastructure-only target was changed to the actual CUDA device
name and memory so an A100 would not be falsely judged against 4090 capacity.
The run configuration, data, model, optimizer, MB32/GAS32 decomposition, and
interventions were not changed. Scientific training still has not started.

Approved execution Step 8 completed at `2026-08-29T21:45:03Z`. Four Secure
A100 SXM 80 GB workers were requested without a data-center restriction. Three
initial requests succeeded; the `relu-l1n-0p05` request failed with no-resource
capacity and created nothing, then succeeded on retry. The live five-GPU fleet
is A100 PCIe Pod `zqet1vl0hzsxa0` at `$1.39/hour` plus A100 SXM Pods
`7b41lfe965txbh`, `6pi00lbzvp6kpx`, `15vi5yw3cw1u1j`, and
`58ukutbtquykle` at `$1.59/hour` each. All report `RUNNING`, the pinned image,
30 GB container disk, 25 GB `/workspace` volume, and direct public SSH. The
four new workers have automatic termination guards at
`2026-08-30T01:45:00Z`; active GPU rate is `$7.75/hour`. Source/runtime/cache
staging and scientific training have not started on the new workers.

Approved execution Step 9 completed at `2026-08-29T22:06:02Z`. The pinned
source/environment was installed on all four A100 SXM workers. Three are in
`US-MD-1`; `6pi00lbzvp6kpx` is in `US-MO-1`, whose MFS-backed installation took
about 15 minutes versus roughly 5-6 minutes for the others. The seed transferred
the exact 5.97 GB cache concurrently to all four workers in 78-84 seconds. A
temporary SSH key was restricted by a forced write-only `rrsync` command to the
Run 004 cache directory; after all destination hashes passed, its authorization
was removed, rejection was verified, and the seed private/public files were
deleted. Every worker then passed the normal cache loader, pinned runtime,
CUDA/flash-attention, exact schedule SHA-256
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`,
and run-code SHA-256
`501744d66aec47c469c04a1885c97372b39315bd3bc2a67297d8353e4efe5e2d`.

The exact A100 SXM preflight also passed. ReLU control boundaries took 5.463
and 4.717 seconds; ReLU lambda 1 took 6.832 and 5.626 seconds. Both paths were
finite, non-overflowing, and non-skipped with the same 56.941/56.953 GiB peak
reserved memory and 71.325 GiB safety limit. The complete Step 9 evidence is
under `prelaunch/step9-a100-fleet-readiness/`. All five GPUs were idle at the
closing audit; no Run 004 scientific attempt has started.

Approved execution Step 10 launched the fixed five-worker scientific run on
2026-08-29 at approximately `22:10Z`. Pod/worker/PID assignments are:
`zqet1vl0hzsxa0` / `relu-l1n-1` / 2775; `7b41lfe965txbh` / `controls` / 1082;
`58ukutbtquykle` / `relu-l1n-0p05` / 893; `6pi00lbzvp6kpx` /
`relu-l1n-0p1` / 949; and `15vi5yw3cw1u1j` / `relu-l1n-0p5` / 893. The first
read-only snapshot found all five manifests running and recorded steps 16, 9,
6, 10, and 4 respectively. All task/augmented losses were finite, loss scale
was 4096, no gradient overflow or optimizer-step skip occurred, and reserved
memory remained consistent with the passed preflight. Five-minute monitoring
began immediately and changed to ten-minute polling at the user's explicit
direction; this changed detection latency only, not the detached training jobs.

All scientific work was terminal by `21:19:45` BRT. Each attempt completed 712
optimizer updates, final-checkpoint reload, all 338 complete validation blocks,
the declared 1,444-token excluded tail, activation/weight/logical-product
diagnostics, and the approved checkpoint inventory. The final validation losses
from the reloaded final checkpoints are:

| Condition | Final validation loss |
| --- | ---: |
| GeLU control | 5.2085825000 |
| ReLU control | 5.2696459956 |
| ReLU L1N, lambda 0.05 | 5.2061491323 |
| ReLU L1N, lambda 0.1 | 5.1654738505 |
| ReLU L1N, lambda 0.5 | 5.1126882931 |
| ReLU L1N, lambda 1.0 | 5.1022763422 |

Every attempt's 60 declared transfer files passed local size and SHA-256
verification. `03_verify.py` then verified all six conditions, 4,272 optimizer
updates, 8,959,033,344 aggregate training input tokens, one shared initial
parameter hash, one full-pass schedule hash, and one run-code hash. Local
artifacts comprise 370 files and 6,084,775,993 bytes. The lambda-1 transfer
required recursive SFTP resume after an SCP connection reset; the completed
hashes prove that no remote artifact was lost or regenerated.

A strict event-history audit found 4,055 leading NUL bytes before ReLU control's
otherwise complete step-1 JSON event. The raw file matches its remote inventory
and remains unchanged. Removing that prefix for parsing yields all 712 sequential
events, including the recoverable clipped step-1 boundary, with zero overflow,
zero skipped update, and loss scale 4096. The exact audit and normalization rule
are in `artifacts/event-history-audit.json`. The other five event logs parse
strictly without normalization.

The focused Run 004 suite passes 12/12 after retrieval. The repository-wide
suite passes 107 tests and has one unrelated failure: Run 007's test expects an
unapproved launch packet while its current plan records `approved_for_launch`.
Run 004 files are not implicated.

Artifacts and verification finished by approximately `21:28` BRT. All five
scientific Pods were deleted after local verification; the closing Pod list was
empty. The 2026-08-30 billing refresh reports the final posted Pod total as
`$21.5213452158`, including the two earlier RTX 4090 preflight Pods, five A100
Pods, and Pod disk. Volume `9luykg5yc3` remains intentionally retained at
`$7/month`; active GPU cost is zero.

These losses and diagnostics are verified artifacts. The two post-hoc
observations and figures are complete, but they are not an approved consolidated
finding or manuscript claim. The optional checkpoint learning-trajectory
evaluation remains explicitly deferred.

## Question and conditions

This run asks whether naive L1 pressure at the ReLU MLP hidden activation `h`
changes unpressured attention-site activation geometry over one full MiniPile
pass, beyond the GeLU-to-ReLU operator change.

The six matched conditions are:

1. stock GeLU control, topology `A0`, no pressure;
2. ReLU control, topology `A1-H`, no pressure;
3. ReLU `A1-H` with `l1_naive` at `h`, lambda `0.05`;
4. ReLU `A1-H` with `l1_naive` at `h`, lambda `0.1`;
5. ReLU `A1-H` with `l1_naive` at `h`, lambda `0.5`;
6. ReLU `A1-H` with `l1_naive` at `h`, lambda `1.0`.

Pressure uses the operational definition in `research/METHODS.md`: mean absolute
activation within each captured layer tensor followed by an unweighted mean
across the six `h` tensors. The objective is task loss plus lambda times that
quantity. Task and unweighted pressure gradients are retained at every optimizer
boundary; the combined gradient is clipped once at global L2 norm `1.0`.

## Pythia recipe mapping and declared differences

Unspecified model and training choices follow the official Pythia-14M recipe:
2,048-token sequences, global batch 1,024, peak/minimum LR `1e-3`/`1e-4`, AdamW
betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`, one-percent warmup,
cosine decay, zero dropout, FP16 dynamic loss scaling, seed `1234`, no activation
checkpointing, `small_init` for ordinary weights, and `wang_init` for attention
and MLP residual-output projections. The production decomposition is the
recipe's microbatch `32`, accumulated for `32` microbatches on one GPU.

The approved execution engine is a recipe mapping, not a bitwise reproduction:

- Transformers 5.12/PyTorch 2.11 replaces GPT-NeoX v1.0/DeepSpeed;
- PyTorch SDPA's CUDA flash kernel replaces the original NeoX flash-attention
  implementation;
- PyTorch's fused AdamW replaces Apex/DeepSpeed FusedAdam. Biases and all
  LayerNorm parameters remain in the recipe's zero-weight-decay group, but the
  fused kernel, state update rounding, and overflow integration are not
  bitwise-equivalent;
- the run-local scaler implements the declared initial scale, growth window,
  hysteresis, and minimum scale, but optimizer/kernel rounding and RNG draw order
  are not NeoX-bitwise;
- parameter distributions follow `small_init` and `wang_init`, but module
  traversal and random draw order follow Transformers;
- the source recipe used 32-way data parallelism with one 32-sequence
  microbatch per rank; each condition here uses one GPU and accumulates 32 such
  microbatches. The global-batch gradient is mathematically matched, but the
  reduction/rounding path and wall time are not;
- MiniPile, one pass, six activation conditions, complete MiniPile validation,
  and the diagnostic bundle are experiment inputs rather than Pythia defaults.

There is also an operational-document discrepancy worth making explicit:
`research/METHODS.md` currently names BF16 as the bootstrap reference precision,
whereas the pinned Pythia recipe uses FP16 with dynamic loss scaling. This run
uses FP16 because the user's instruction to match Pythia governs this design.
That improves recipe fidelity but adds scaler/overflow behavior absent from the
BF16 bootstrap runs.

The LR scheduler follows GPT-NeoX v1 rather than the bootstrap helper: the
warmup duration is the fractional value `0.01 * 712 = 7.12`, and the scheduler
value in force before each optimizer step is used. Consequently update 1 has LR
zero and later values are shifted relative to a `ceil(0.01*N)` implementation.
This preserves the source recipe's early-learning dynamics, but it differs from
the generic schedule described in `research/METHODS.md`.

The operational MiniPile cache remains divided into disjoint 2,048-token blocks
and uses Transformers' within-block causal shift (2,047 loss targets per block).
GPT-NeoX's indexed-data loader ordinarily reads a 2,049-token source window to
form 2,048 targets. Adopting that would consume boundary/tail tokens and change
the already approved full-block coverage definition, so Run 004 keeps the
repository's operational block contract. The implication is that recipe
matching does not extend to cross-block target construction.

These differences permit matched causal comparisons inside this cohort but do
not support calling its checkpoints exact reproductions of released Pythia
training.

## Budget, data, and validation

The immutable training cache has 728,374 complete blocks plus a 1,464-token
tail. There are 712 constant-size updates. The final update consumes the last
310 unseen blocks and wraps to the first 714 blocks of the seeded permutation.
Thus each condition processes 729,088 sequences and 1,493,172,224 input tokens;
the wrap is 0.098 percent beyond one unique complete-block pass. Across six
conditions the run processes 8,959,033,344 input tokens.

Validation after update 1 and from the reloaded final checkpoint covers all 500
MiniPile validation documents: 338 complete blocks, 692,224 input tokens, and an
explicitly excluded 1,444-token tail.

## Diagnostics and retention

Final count-first activation statistics cover `m`, `h`, `q_post`, `k_post`, `v`,
and the attention output immediately after `W_o` and before residual addition.
They include exact zero, near-zero thresholds `1e-3` and `1e-2`, mean absolute
value, RMS, and L2 moments. Weight statistics include all named parameters,
including bias and normalization. A separate eager-attention validation pass
collects the six logical-product counters and reports `R_block`, `R_model`, and
the topology-conditioned integer `R_model_max`; it is not a speed measurement.

Model snapshots follow Pythia's early learning-dynamics cadence at updates 0, 1,
2, 4, 8, 16, 32, 64, 128, 256, and 512, plus final update 712. Optimizer, scaler,
and RNG state are retained at 256, 512, and 712. Every final checkpoint is kept.

## RunPod deployment closeout

The production path was five independent Secure one-GPU A100 workers: one A100
PCIe 80 GB cache seed/pressure worker and four A100 SXM 80 GB workers. The
controls worker ran GeLU and ReLU controls sequentially; the other four workers
each ran one pressure condition. Scientific workers launched at about `19:10`
BRT, the last condition finished at `21:19:45`, and retrieval, hash
verification, cross-condition verification, and final teardown finished at
about `21:28`.

The deployment experience was:

| Stage | Outcome | Implication for later runs |
| --- | --- | --- |
| Community RTX 4090 capacity | Initial request had no capacity; a later Pod exposed a CUDA initialization failure. | Treat allocation and runtime readiness as separate gates; a `RUNNING` Pod is not a passed preflight. |
| Secure RTX 4090 exact preflight | CUDA and flash SDPA worked, but unchanged MB32/GAS32 OOMed. | Size from a production-shaped optimizer boundary, not model parameters or nominal VRAM alone. |
| Secure A100 80 GB preflight | Both ReLU control and lambda 1 passed; peak reserved memory was about 56.95 GiB. | The approved decomposition required an 80 GB class for this implementation; a different decomposition is a new explicit decision. |
| Local-to-Pod cache upload | Two 5.97 GB uploads became unusably slow at 74% and 91%. | Do not assume workstation upload is the fastest seed path. |
| Direct Hugging Face rebuild | The pinned dataset/tokenizer path reproduced both exact cache hashes; full train publication took about 17 minutes on the eventual A100 seed. | Build once from immutable revisions when transfer is poor, then verify before reuse. |
| Network volume | Preserved the cache but forced Secure Cloud and `EUR-IS-1`, where suitable capacity was scarce. | Decide GPU fit and placement before creating a location-bound volume. Retain one only when its placement constraint is acceptable. |
| Worker cache distribution | Cross-region seed-to-worker transfer to four A100 SXM Pods completed concurrently in 78--84 seconds and all hashes passed. | Build or stage once, distribute the immutable cache, and never retokenize independently on every worker. |
| Training timing | Boundary timing omitted batch construction and host-to-device staging, initially understating ETC. | Estimate from end-to-end update elapsed time; label narrower timers explicitly. |
| Retrieval | One recursive SCP reset; resumable SFTP completed all 60 files without restarting. | Use resumable transport for checkpoint trees and accept delivery only after inventory/hash reconciliation. |
| Teardown | Each completed worker was retrieved, verified, and deleted; the final live audit returned zero Pods. | Teardown per worker after verification, then perform an account-level resource and billing audit. |

The infrastructure changes never altered sequence length, global batch,
microbatch/accumulation, precision, optimizer, pressure, activation topology,
data identity, validation coverage, or checkpoint schedule. The complete
incident chronology remains in `RUNPOD_STATUS_2026-08-29.md`; the measured
reusable commands and cautions remain in `DEPLOYMENT_PLAYBOOK.md`.

### Original launch definition and historical estimates

Five independent one-GPU Secure Pods are proposed. The `controls` worker runs
the two controls sequentially from fresh processes/states. Four other workers
run one pressure condition each. Explicit condition-order attempt numbers make
the independently written attempt directories merge without collision.

The exact source packet is based on Git commit
`39e5be7b27a9f0b6df3746b71991e3439526871f` plus a disclosed dirty overlay: the
current Run 004 folder, `src/sparsity_research`, `pyproject.toml`, and tests. The
production run-code content hash is
`501744d66aec47c469c04a1885c97372b39315bd3bc2a67297d8353e4efe5e2d`; the
resolved config hash is
`e5eba31eefe5a721c6908abff7cc5a1db6a8dccda50bdc7e6b7ff9d4a8ecf511`.
Running from an uncommitted overlay is less convenient to reproduce than a clean
commit, but every behavior-bearing file is inventoried and checked before
preflight. Unrelated local edits are not staged to the Pods.

The first preflight Pod uses verified `runpodctl` 2.8.0 with the image digest,
Community 4090, one GPU, 30 GB container disk, 25 GB `/workspace` volume, CUDA
>=12.8, SSH, and an absolute RFC3339 termination time seven hours after
creation. The reusable execution path instead attaches the 100 GB network
volume at `/workspace`, which forces Secure Cloud and `EUR-IS-1`; container
disk remains 30 GB. `DEPLOYMENT_PLAYBOOK.md` records the exact measured staging,
direct-Hugging-Face cache, local-cache copy, preflight, launch, monitoring,
retrieval, and teardown steps. GPU substitution is authorized but requires a
fresh target-GPU preflight and an explicit recorded cost/ETC revision; the
scientific config remains immutable.

The refreshed 2026-08-29 live catalog reports Community/Secure RTX 4090 at
`$0.34`/`$0.74` per GPU-hour with low availability. The shared-volume path caps
five seven-hour Secure workers at `$25.90` GPU compute. The original guarded
Community preflight can add at most `$2.38`; container and pro-rated in-run
network storage add less than `$0.25`. Thus the revised Run 004 worst-case
compute/storage envelope is about `$28.5`, plus the intentionally retained
network volume at `$7/month`. The earlier `$12.17` ceiling applied only to the
superseded Community-only/no-network-volume deployment. There is no RunPod
ingress/egress charge. `prelaunch/launch-plan.json` records the live resource
IDs, attempts, transport decisions, and arithmetic.
The 32-sequence microbatch is a hard preflight condition. If it does not fit a
24 GB RTX 4090, implementation will stop before training; it will not silently
reduce the microbatch or enable activation checkpointing.

The staged launch first runs `05_remote_preflight.py` on one candidate 4090. It
verifies the pinned runtime and both cache hashes, then measures two complete
32-microbatch optimizer boundaries for the ReLU control and lambda-1 pressure
paths. Both must keep peak reserved CUDA memory at or below 90 percent of 24 GiB
with finite, non-overflowing updates. Only then may the other four Pods be
created and the five detached workers launched.

## Post-hoc spillover figures

`07_plot_spillover_figures.py` generates two Run 004 analogues of the Run 002
sitewise figures at the same `|x| <= 1e-3` threshold. It recomputes every point
from integer threshold hits and totals and verifies the six layer rows against
the pooled record before dividing. `figures/01-h-vs-site-near-zero-grid.pdf`
compares `h` with `q_post`, `k_post`, `v`, and `m` in a shared-axis grid;
`figures/02-h-vs-attention-output-near-zero.pdf` compares `h` with the output
after `W_o` and before residual addition. Because Run 004 has only one GeLU
condition, GeLU is shown as a standalone control while the five ReLU points form
the dose-ordered trajectory. The full captions, values, coverage, caveats, and
provenance are in `observations/O001-h-vs-site-near-zero-grid.md` and
`observations/O002-h-vs-attention-output-near-zero.md`.

These are descriptive post-hoc views of already collected terminal statistics.
They do not change a scientific input, require checkpoint replay, or by
themselves authorize a manuscript or consolidated-finding update. The complete
Run 004 focused test module passes 14/14, including count-first reconciliation
and control/dose-topology tests for this reduction.

`near-zero-mass-and-r-model.md` gives the corresponding six-condition table for
all activation sites collected by Run 004 at `epsilon = 1e-3`, together with
the exact-zero logical-product `R_model` percentage. The table keeps these two
estimands distinct and records their artifact provenance.

### TODO - validation-loss learning trajectories

Generate a processed-training-tokens versus complete-validation-loss figure for
all six conditions. The verified local attempt trees retain model checkpoints
at updates `0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 712`, so this requires
post-hoc checkpoint evaluation but no retraining. Evaluate every checkpoint on
all 338 complete validation blocks, report the 1,444-token excluded tail, and
use `update * 2,097,152` as the processed-training-token coordinate. The stored
step-1 and reloaded-final validation losses must serve as reproduction checks.
Do not substitute per-update training loss or interpolate unevaluated
checkpoints. Record the new integer coverage and loss artifacts, add a Run 004
observation, and save the publication figure as PDF beside the existing figures.

## Closed-run handoff

- Do not relaunch or modify a Run 004 scientific condition. A corrected
  scientific input would require a new numbered run.
- The optional validation-loss trajectory can be completed from the retained
  checkpoints without retraining, following the TODO above.
- Promoting O001/O002 into `research/findings/` or changing manuscript TeX
  requires a separate user decision about the scientific statement and caveats.
- The retained network volume is an independent billing decision. Keep it only
  if near-term `EUR-IS-1` Secure workloads benefit from the cache; deletion or
  migration requires explicit approval.
- Repository execution has already moved on to Run 008. New experimental
  direction remains user-selected; the next unused run number is 009 and the
  next unused analysis number is 001.

## Interpretation

A dose-ordered movement in ReLU `h` sparsity together with systematic movement
at unpressured attention sites supports the simple spillover signature. A
successful `h` manipulation with stable attention sites refutes that signature;
failure to change `h` means the manipulation did not succeed. One seed, one
scale, and one pass are descriptive evidence, not a general scaling claim,
causal-route proof, or runtime-speedup result.
