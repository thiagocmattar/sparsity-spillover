# Run 009 - Pythia-14M full MiniPile pass with ReLU OL1

## Status

Design and scientific launch were approved by the user on 2026-08-30. All four
OL1 conditions completed, their archives and internal inventories were
hash-verified locally, and the global verifier passed against Run 004's reused
controls and matched L1 conditions. All six Run 009 Pods are terminated.

The focused Run 009 suite passes 8/8. The repository-wide suite passes 131/132;
the sole failure is the pre-existing Run 007 assertion that expects its launch
packet to be unapproved even though Run 007 now records `approved_for_launch`.
Run 009 and shared scientific modules are not implicated.

## Question and hypothesis

This run asks whether operational orthogonal L1 pressure at the ReLU MLP hidden
activation `h` improves the full-pass quality--sparsity/logical-opportunity
frontier relative to Run 004's matched naive L1 conditions.

At a matched lambda, support means lower complete-validation loss at comparable
`h` sparsity/`R_model`, or higher `h` sparsity/`R_model` at comparable loss. The
hypothesis is limited if all OL1 points are dominated by their Run 004 L1
comparators, if `h` does not move, if quality degrades severely, or if trust
saturation makes nominally different lambdas operationally indistinguishable.

## Conditions and reused evidence

Four new conditions use ReLU topology `A1-H`, operational `orthogonal_l1` only
at `h`, trust budget `1.0`, and lambda in `{0.05, 0.1, 0.5, 1.0}`. Each condition
has one independent RunPod worker and one immutable attempt order.

Run 004's GeLU and ReLU controls are reused rather than rerun. Its four naive-L1
conditions are the pairwise method comparators. Run 009 locks Run 004's valid
verification, initial-parameter hash, training-schedule hash, and run-code hash;
terminal verification will reject any mismatch.

## Matched Pythia recipe

The scientific configuration matches Run 004 except for the pressure method:

- pinned Pythia-14M architecture config, random initialization, seed `1234`;
- `small_init` ordinary weights and `wang_init` residual-output projections;
- 2,048-token sequences, global batch 1,024 as MB32/GAS32;
- 712 boundaries, 729,088 scheduled blocks, and the same 714-block wrap;
- 1,493,172,224 input tokens per condition and 5,972,688,896 total;
- peak/minimum LR `1e-3`/`1e-4` with GPT-NeoX-v1 pre-step semantics;
- fused AdamW, betas `(0.9, 0.95)`, epsilon `1e-8`, weight decay `0.1`, and
  zero decay for biases and LayerNorm parameters;
- FP32 parameters/state, FP16 autocast and dynamic loss scale, flash SDPA,
  zero dropout, and no activation checkpointing.

The expected initial-parameter SHA-256 is
`ece58512e94ee2f97d17278fe8af4c1abef9c5f7f9dbdd4087e36d7f67d7af57`.
The expected schedule SHA-256 is
`f1755812b4f70806bd137ee900c9338f64c4c2074b6dd8b7661e6bde9b141faa`.

As in Run 004, this is a Transformers/PyTorch mapping of the Pythia recipe, not
a GPT-NeoX-bitwise reproduction. Within-block causal shifting supplies 2,047
loss targets per 2,048-token input block.

## FP16 OL1 boundary

Every boundary accumulates separately scaled task and pressure gradients over
the same 32 microbatches, then unscales both. The task gradient alone is
globally clipped to norm `1.0` and consumed by AdamW. The clipped task gradient
drives AdamW's moments and the OL1 task direction. The pressure gradient remains
unclipped, uses AdamW's task second moment, has only a globally conflicting
component projected away, is capped to a weighted direction ratio of `1.0`, and
is applied after AdamW.

If either gradient component is nonfinite, the whole boundary is atomic: both
AdamW and OL1 are skipped and the dynamic loss scaler is updated. No partial
pressure correction is permitted. Terminal verification requires zero skipped
boundaries.

This combines Run 005's operational OL1 definition with Run 004's FP16 recipe.
The manuscript describes the trust budget as optional, while the executing
bootstrap requires the approved positive value `1.0`. The clipped task gradient
drives both AdamW and OL1 geometry.

## Data and complete validation

The pinned train cache contains 1,491,711,416 tokens and 728,374 complete blocks,
with a 1,464-token tail. Its SHA-256 is
`da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c`.

Every ordinary and diagnostic validation pass covers all 500 validation
documents: 338 complete blocks and 692,224 input tokens. The 1,444-token tail is
excluded and reported. Ordinary validation runs after boundary 1 and from the
reloaded final checkpoint.

## Diagnostics and retention

Every train event records task/pressure losses, task-only clipping, raw gradient
interaction, FP16 scale/overflow, OL1 adaptive directions, projection, raw and
final trust ratios, LR, throughput, elapsed time, and peak CUDA memory.

The reloaded final checkpoint receives count-first exact-zero, `1e-3`, and
`1e-2` activation statistics plus RMS/L2 moments at `m`, `h`, `q_post`,
`k_post`, `v`, and post-`W_o` attention output. All named weight norms and the
six exact logical-product families are retained. `R_block`, `R_model`, and the
integer A1-H `R_model_max` are logical opportunities, not speedup.

Model snapshots are retained at boundaries `0,1,2,4,8,16,32,64,128,256,512,712`.
Optimizer, scaler, and RNG recovery state are included at `256,512,712`.
There is no clipping frontier. Gradient interaction is collected during
training because it cannot be reconstructed from checkpoints.

## Implementation ownership and verification

Run 009 owns its config, OL1/FP16 boundary, smoke/preflight, worker entrypoints,
verification, and deployment record. It loads Run 004's frozen recipe
initialization, orchestration, diagnostics, and checkpoint machinery under
private module names. Every reused file is included in Run 009's run-code
content hash, so this dependency is explicit and transferred with the source
packet.

Tests cover the four-condition matrix, absence of new controls, exact Run 004
recipe/schedule/diagnostic/checkpoint match, comparator hashes, task-only
clipping and AdamW moments, trust-budget enforcement, atomic nonfinite handling,
target-smoke health rules, and complete code inventory.

## RunPod execution

Four independent Secure one-GPU Pods are proposed. To pair hardware with Run
004 where live capacity permits, lambda `0.05`, `0.1`, and `0.5` request A100
SXM4 80 GB; lambda `1.0` requests A100 PCIe 80 GB. Placement may use any data
center. Capacity polling waits five minutes between read-only checks and
continues until an approved GPU appears or the user interrupts. Any accelerator
substitution still requires a new explicit decision.

Run 004 measured about 56.95 GiB peak reserved for its MB32/GAS32 pressure path,
and the same target failed on a 24 GB RTX 4090. Run 009 therefore requires an
exact OL1 target preflight. The preflight runs five complete lambda-1 boundaries,
checks initialization, cache and schedule hashes, rejects overflow/trust-budget
violations, and enforces 10 percent device-memory headroom. It creates no
scientific attempt.

The retained 100 GB Standard volume `9luykg5yc3` in `EUR-IS-1` remains untouched.
It will be used only if the first eligible placement happens to be in that data
center; otherwise a verified 5.97 GB cache copy is staged to the Pod volume.
All reusable files live under `/workspace`, which survives a Pod stop. After a
successful preflight, the Pod is stopped for at most 24 hours while awaiting
scientific approval. If restarted, it becomes the hardware-matched lambda `1.0`
worker after PCIe preflight or lambda `0.05` worker after SXM preflight, with no
second source/cache transfer. The other three workers require concurrently
staged and hash-verified copies. No new network volume is proposed.

Live prices, capacity, exact preflight resource, maximum cost, and termination
deadline belong in `prelaunch/launch-plan.json` and require the user's separate
approval before creation.

The exact preflight passed five lambda-1 MB32/GAS32 boundaries with a `5.7424`
second median, `56.953125` GiB peak reserved memory, zero overflow/skips, and a
final OL1 trust ratio of `0.441796`. The retrieved packet and all internal hashes
verified locally. Pod `ujafdg00hqcq3q` is stopped in `CA-MTL-3` and can become
the lambda-1 worker without another source, environment, or cache transfer.

The approved scientific packet restarts that Pod and creates three
Secure A100 SXM4 80 GB workers in any available data centers. At current prices
the combined rate is `$6.16/hour`; expected incremental compute is `$12.00` to
`$13.50`, with a hard `$15.50` scientific compute-plus-storage envelope and a
2.5-hour billable limit per Pod. The user approved this launch on 2026-08-30.
The retained lambda-1 worker began training, while the lambda `0.05`, `0.1`, and
`0.5` workers entered concurrent environment and cache staging. Live Pod IDs,
placements, deadlines, and readiness state are recorded in
`prelaunch/launch-plan.json`.

The first lambda `0.1` environment setup stalled during the PyTorch install:
the log made no progress for more than 15 minutes, the installer had no child,
and its four HTTPS sockets were all in `CLOSE-WAIT`. This was an infrastructure
failure before cache construction or scientific execution. Its original log
was retained, only the stuck installer was terminated, and setup attempt 002
was launched on the same Pod under
`prelaunch/scientific-workers/relu-ol1-0p1/setup-attempt-002/`. The approved
source, config, condition, and execution definition did not change.

Setup attempt 002 reproduced the same socket failure, so both logs were copied
and hashed locally under
`prelaunch/scientific-workers/relu-ol1-0p1/infra-attempt-001-bad-pod/` before
Pod `wr8s9u2geswh5o` was terminated. Replacement Pod `tjze8cu812heep` retained
the approved A100 SXM4 80 GB type and price, verified the exact source packet
and base commit, and began a fresh detached environment setup. The replacement
has its own 2.5-hour teardown deadline in `prelaunch/launch-plan.json`.

Replacement attempt 002 landed in `US-KS-2` again and exposed severe
persistent-filesystem latency: after five minutes it had not completed Python
`venv`/`ensurepip`, while both `US-MD-1` workers had finished the whole pinned
environment in about that time. Its empty pre-output log was retained and the
Pod was terminated before any cache or scientific attempt. Replacement attempt
003 was therefore constrained to `US-MD-1`; Pod `suq98ymotdpc0g` verified the
same A100 SXM4 80 GB, source packet, and base commit before starting setup.

The completed lambda `1.0` attempt exposed a deployment-verifier scope detail:
`03_verify.py` is the intended cross-Pod terminal verifier and requires all four
condition-order directories to be co-located, so it cannot run successfully on
one distributed worker alone. Before teardown, the lambda `1.0` archive hash,
internal transfer inventory, manifest, 712-event OL1 history, validation
coverage, diagnostics, checkpoint cadence/content, and final optimizer state
were instead verified locally with the existing verifier functions. The global
`03_verify.py` subsequently passed after all four attempts were retrieved.

## Scientific closeout

All conditions completed 712 optimizer boundaries without FP16 overflow or a
skipped boundary. Final validation used all 338 complete blocks and excluded the
declared 1,444-token tail.

| lambda | OL1 final validation loss | matched Run 004 L1 loss | OL1 minus L1 |
| ---: | ---: | ---: | ---: |
| 0.05 | 5.1980621885 | 5.2061491323 | -0.0080869438 |
| 0.1 | 5.1593908050 | 5.1654738505 | -0.0060830455 |
| 0.5 | 5.1102351409 | 5.1126882931 | -0.0024531522 |
| 1.0 | 5.1210295688 | 5.1022763422 | +0.0187532267 |

These are validation-loss coordinates, not by themselves a conclusion about the
quality--sparsity frontier. The retained activation and logical-product
diagnostics support a later approved analysis; the one-seed and cross-Pod limits
remain in force.

The four internal transfer inventories total 4,058,598,440 bytes. Each archive
matched its remote byte count and SHA-256 before extraction, and the existing
single-attempt verifier passed before its Pod was removed. Global
`artifacts/verification.json` records `status: verified` and `evidence_label:
valid`.

The final RunPod audit found zero active Pods and only the pre-existing 100 GB
Standard volume `9luykg5yc3`, which remains retained at its independent
`$7/month` cost. At `14:15Z`, posted Pod billing for the 2026-08-30 execution
window was `$12.2669929022`: `$12.2021780834` GPU and `$0.0648148188` Pod disk
across six Pods, including the reusable preflight Pod and the two failed lambda
`0.1` infrastructure placements. This is provisional because the final lambda
`0.1` hourly bucket had not fully settled; every Pod was nevertheless terminated
within its approved 2.5-hour guard and below the approved cost envelope. No new
network volume was created.

## Manuscript relationship and limits

This run directly exercises the methodology's AdamW-relative OL1 equations and
the candidate claim that conflict-aware pressure improves the quality--logical-
compute frontier relative to simpler local pressure. It is an `h`-only method
comparison, not evidence for an architecture-wide topology advantage.

One seed, one 14M model, one pass, and independently scheduled Pods remain
descriptive. GPU/kernel execution is scientifically matched but not bitwise
across machines. No sparse kernel or runtime speedup is measured. Results will
not be promoted to a manuscript claim without a separate user-approved finding.

## Approval record

- 2026-08-30: user approved the detailed design and explicitly selected Run
  004's lambda grid `{0.05, 0.1, 0.5, 1.0}` for the comparison.
- 2026-08-30: user approved one reusable preflight Pod in any data center,
  five-minute capacity retries, and a $3.40 maximum incremental envelope.
- 2026-08-30: user approved the four-worker scientific launch under the measured
  $15.50 maximum incremental compute-plus-storage envelope.
