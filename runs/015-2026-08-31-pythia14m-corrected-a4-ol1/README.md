# Run 015 - Corrected paper-scale Pythia-14M A4-OL1

Status: completed and verified. Five condition-parallel scientific attempts
completed, were verified remotely, transferred with matching archive hashes,
safely extracted, and independently verified locally. The cohort verifier
accepts all 3,560 optimizer boundaries and 20 complete validation passes. All
RunPod Pods were deleted; the pre-existing network volume is unchanged.

## Question and decision rule

Does adding operational OL1 pressure to every active A4-Z site improve the
matched Pythia-14M validation-loss versus measured-`R_model` frontier over the
same five threshold values as Run 011?

Support requires a useful paired change in complete-validation loss and
count-first `R_model` relative to Run 011 at one or more matched `kappa` values.
Refutation includes no paired improvement or a quality cost throughout the
grid. Either outcome remains limited to one seed, one 14M scale, and one full
MiniPile pass. The joint four-site pressure cannot identify a single-site
effect, and logical-product opportunity is not measured runtime speedup.

## Verified result

| `kappa` | corrected A4-OL1 loss | corrected `R_model` | A4 loss | A4 `R_model` | OL1 loss delta | OL1 `R_model` delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 5.458170 | 7.8100% | 5.470497 | 7.2120% | -0.012327 | +0.5980 pp |
| 0.01 | 5.458277 | 8.5867% | 5.466500 | 7.4137% | -0.008222 | +1.1730 pp |
| 0.05 | 5.489803 | 10.4564% | 5.434110 | 8.2059% | +0.055693 | +2.2506 pp |
| 0.1 | 5.548323 | 11.3943% | 5.419642 | 8.9530% | +0.128681 | +2.4413 pp |
| 0.5 | 6.037987 | 12.7134% | 5.659680 | 10.2155% | +0.378307 | +2.4979 pp |

The scoped hypothesis is supported in a threshold-dependent form. Corrected
four-site OL1 improves both measured outcomes at `kappa=0` and `0.01`. At
`0.05`, `0.1`, and `0.5`, it buys additional logical opportunity with an
increasing validation-loss cost. Among the ten matched Run 011/015 endpoints,
the corrected `kappa=0.05`, `0.1`, and `0.5` rows extend the nondominated
frontier; the Run 011 `kappa=0.1` row remains its lowest-loss point.

Every condition realized all 24 pressure tensors on all 712 boundaries, had
zero overflows or skipped updates, and kept the final pressure/task correction
ratio at or below the trust budget. The run-local candidate observation is in
`observations/O001-corrected-a4-ol1-versus-a4.md`. This does not restore
discarded Finding F001 or alter result-bearing TeX: a new numbered cross-run
analysis and explicit user approval are still required.

## Why this is a new run

Run 012's configuration and manifest declare pressure at `a,m,h,z`, but its
training implementation reuses Run 004's unchanged line:

```python
capture_context = ActivationCapture(model, ["h"], torch=torch) if pressure.enabled else nullcontext(None)
```

Consequently, the realized Run 012 pressure objective was the unweighted mean
of only the six `h.layer_*` tensors. Its gates and terminal diagnostics still
covered A4-Z, so the historical evidence is accurately described as **A4-Z
gates plus OL1@h**, not four-site A4-OL1. Run 012's verifier checked the declared
pressure config but did not record or validate the realized capture names.

This is a scientific-input correction, so Run 012 remains immutable and Run
015 is a new numbered run. The Run 012 numbers remain valid only for their
realized A4-Z + OL1@h intervention. They cannot support the declared A4-OL1
comparison or Finding F001.

## Corrected intervention

For every microbatch and every `kappa` in `{0, 0.01, 0.05, 0.1, 0.5}`:

1. apply the same one-sided threshold, with equality surviving, at
   `a`, `m`, `h`, and `z` in all six transformer layers;
2. capture each post-gate tensor, yielding exactly 24 canonical names;
3. compute one aggregate pressure loss: the unweighted mean of each tensor's
   mean absolute value;
4. differentiate that complete aggregate and apply operational
   `orthogonal_l1` with `lambda=1` and `step_budget=1`.

The expected names are `{a,m,h,z}.layer_{0..5}`. Their sorted newline-delimited
SHA-256 is
`8277a447cefd5c4a91533b9fe0dd59fddd0c7103e8b8cf984b76a16e0252add1`.
The boundary fails before differentiation when any name is missing or extra.
Every train event records the realized tensor count and name hash; preflight
and terminal verification reject anything other than all 24 tensors.

This matches the desired A4-OL1 description exactly: **A4-Z pressure is applied
at all sites in `{a,m,h,z}`, for every kappa, with `lambda=1`**. It is not four
separate site losses with four independent lambda values; the repository's
approved OL1 estimand is one unweighted aggregate over all site/layer tensors,
multiplied once by `lambda=1`.

## Matched scientific identity

Only the Run 012 capture bug, evidence guards, and a non-scientific separation
of preflight/scientific termination deadlines change. Run 015 otherwise locks
the valid Run 012 design and the Run 011 comparator:

| Field | Run 015 contract |
| --- | --- |
| Model | random-initialized `EleutherAI/pythia-14m-deduped` architecture config, revision `7386d9a4ae45aef494a6e704910394def3037fc5` |
| Initialization/data seeds | model `1234`; data order `1234` |
| Data | pinned MiniPile token cache; 2,048-token blocks; one scheduled full pass |
| Conditions | five independent `kappa` rows: `0`, `0.01`, `0.05`, `0.1`, `0.5` |
| Gate | A4-Z one-sided threshold at `a,m,h,z`; equality survives |
| Pressure | post-gate `a,m,h,z`; one unweighted 24-tensor mean; OL1 `lambda=1`, trust budget `1` |
| Budget | 712 optimizer steps; global batch 1,024; microbatch 32; accumulation 32; 1,493,172,224 input tokens per condition |
| Optimizer | mapped Pythia AdamW recipe; peak LR `1e-3`; minimum LR `1e-4`; betas `(0.9,0.95)`; weight decay `0.1`; task-gradient clip `1.0` |
| Numerics | dynamic FP16; FP32 parameters/optimizer; an overflow skips the entire AdamW+OL1 boundary atomically |
| Validation | all 500 documents, all 338 complete blocks, 692,224 input tokens; 1,444-token incomplete tail excluded and reported |
| Comparator | Run 011 at exactly matched `kappa`, initialization hash, schedule hash, validation cache, and recipe |

The mapped Transformers/PyTorch execution is not claimed to be bitwise
GPT-NeoX/DeepSpeed reproduction. Independently scheduled conditions share
scientific identities but are not presumed bitwise replicas.

## Diagnostics and retained artifacts

The launch retains training-time task/pressure gradient norms, dot/cosine,
conflict, projection, raw/final pressure ratios, trust scale/saturation,
overflow, and the realized pressure-capture count/hash at every boundary.
Terminal diagnostics cover count-first exact zero and near-zero mass at
`0`, `1e-3`, and `1e-2`, activation sums/RMS/L2 moments and nonfinite counts at
`a,m,h,q_post,k_post,v,z,attention_output`, all named parameter norms including
bias and normalization, and integer logical-product counters for `R_block`,
`R_model`, and the A4-Z `R_model_max` ceiling.

Model checkpoints are retained at steps
`0,1,2,4,8,16,32,64,128,256,512,712`; optimizer/scaler/RNG recovery state at
`256,512,712`, including the final checkpoint. Predictions and a post-hoc
clipping frontier are not part of this corrective run. Gradient interaction is
collected during training because it cannot be reconstructed later.

## Implementation and regression protection

- `run015_capture.py` converts the frozen Run 004 `h` request to exactly
  `a,m,h,z` and rejects any change to the inherited call shape.
- `optimizer_boundary.py` derives expected names from the realized model layer
  count, compares them with the actual capture after every forward pass, and
  only then builds the aggregate pressure loss.
- `smoke.py` requires the 24-name identity at both kappa endpoints for the exact
  target workload.
- `verification.py` requires that identity in every one of the 712 train events
  for every condition.
- `tests/test_run_015_corrected_a4_ol1.py` covers configuration, exact aggregate
  value, four-site adapter behavior, missing-site rejection, preflight
  rejection, terminal-verifier rejection, and code-identity coverage.

## Manuscript and workflow effect

This run is the corrective source intended for the Pythia-14M `A4-OL1` cell and
paired questions P06, P09, P10, P12, P14, and P16. Until Run 015 completes and
a new cross-run analysis is approved, Run 012's four-site interpretation and
Finding F001 are unsupported. Existing result-bearing TeX is not silently
rewritten by opening this run; the repository crosswalk records it as stale and
pending corrected evidence.

No scientific result will be promoted automatically after execution. A new
numbered analysis must reconcile Run 011 and Run 015 integer counts and obtain
user approval before any finding or manuscript claim is restored.

## Launch gate

The intended execution is condition-parallel rather than DDP: one exact
preflight first, followed only on success by five independent one-condition
GPUs. Launch requires separate explicit user approval.

Read-only discovery at `2026-08-31T17:12Z` found zero Pods and the pre-existing
100 GB Standard volume `9luykg5yc3` (`sparsity-spillover-shared`) in `EUR-IS-1`.
The volume remains independently billable and will not be created, resized,
written, or deleted by this run. Secure A100 SXM 80 GB with host CUDA 12.8 was
live at `$1.59/GPU-hour`, low stock in each of `EUR-IS-1`, `US-KS-2`,
`US-MD-1`, and `US-WA-1`. The catalog's aggregate availability was medium, but
five-GPU allocation is not guaranteed. Secure A100 PCIe 80 GB was `$1.39/hour`
with low CUDA-12.8 stock, but is not an automatic substitute because the exact
fit/performance preflight is locked to SXM.

The proposed maximum is one 1.5-hour preflight plus five 2.5-hour scientific
workers: `1.5*1.59 + 5*2.5*1.59 = $22.26` maximum compute. The launch approval
request uses a `$22.50` incremental cap to include a small ephemeral-storage
buffer; the already-retained volume's continuing charge is outside that run
cap. Each Pod uses a 30 GB container disk and 25 GB `/workspace` Pod volume.

The verified cache payload is 5,969,620,336 token bytes plus metadata per
worker. Placement may span data centers, so every worker receives and verifies
its own immutable cache rather than constraining the fleet to the retained
volume. Based on the matched historical cohort, return artifacts are about
1.015 GB per condition and 5.074 GB total, including all declared checkpoints,
events, manifests, metrics, diagnostics, and transfer inventories. Source,
cache, and returned artifacts are all hash-verified before use or teardown.

The local RTX 5070 Ti Laptop GPU has 12,227 MiB VRAM and cannot represent the
production MB32/GAS32 boundary with safe headroom. Its smoke was also blocked
because the local Torch build reports no Flash-attention kernel. This is an
environmental block, not a scientific-code failure; CPU regression tests cover
the exact aggregate and fail-closed capture behavior. The billable preflight
must demonstrate five finite exact boundaries at both threshold endpoints,
all 24 pressure tensors, and at least 10% memory headroom before the fleet is
authorized to proceed.

Historical matched/A7 runs put complete condition time between roughly 70 and
115 minutes on A100 SXM 80 GB. Run 015 therefore projects about 1.3--2.0 hours
per condition after setup, executed in parallel, with the 2.5-hour worker guard.
Preflight timing supersedes this estimate. Monitoring is read-only every five
minutes with the warning conditions in `DEPLOYMENT_PLAYBOOK.md`.

## Actual RunPod closeout

The exact non-evidence preflight passed on Secure A100 SXM 80 GB at the target
MB32/GAS32 boundary. Its maximum reserved memory was 67,125,641,216 of
85,093,777,408 bytes (78.9%), leaving 21.1% headroom. Both endpoint probes
completed five finite boundaries, used Flash SDPA, and realized the exact
24-tensor pressure identity.

Four infrastructure-only allocations were deleted without creating scientific
evidence: two preflight/control-plane attempts and two cross-region cache
transfers that were too slow for the approved window. Their immutable notes
are under `prelaunch/infrastructure-attempts/`. The successful preflight Pod
and all five accepted scientific Pods ran in `US-MD-1`; the scientific workers
completed in 5,628--6,078 seconds, with median throughput of
338,968--342,386 tokens/s.

The accepted transfer archives were 1,014,835,200 bytes for `kappa=0` and
`0.01`, and 1,014,845,440 bytes for the other three conditions. Every remote
and local SHA-256 matched; all archive paths were confined to the expected
single attempt directory; remote and local standalone verification passed;
and `artifacts/verification.json` then verified the complete five-condition
cohort. `prelaunch/scientific-transfer-closeout.json` records the per-Pod,
archive, readiness, log, billing, and teardown provenance.

The final resource audit found zero Pods and the one unchanged pre-existing
100 GB Standard volume `9luykg5yc3` in `EUR-IS-1`. There is no continuing Pod
compute charge; the retained volume continues its independent charge. Final
resource teardown is authoritative; the latest posted Pod billing and its
still-lagging final hourly bucket are recorded separately in the closeout JSON.
The approved `$22.50` incremental cap was not approached.

## 2026-09-01 billing refresh

The Status Report Number 2 RunPod REST v2 audit reconciled the ten Run 015 Pod
IDs to `$18.3562909383` GPU plus `$0.0923611165` temporary Pod-disk spend,
`$18.4486520549` total. This current Pod-ID total supersedes the lagging
closeout snapshot above and excludes the separately retained network volume.
