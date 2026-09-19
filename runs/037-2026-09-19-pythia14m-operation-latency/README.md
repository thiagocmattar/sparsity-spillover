# Run037: conditional operation latency at 14M T7/Pall, kappa=0.5

Status: design approved by the user on 19 September 2026; implemented and
CPU-verified, awaiting launch approval. **No new GPU measurements exist yet.**
The manuscript's current Table2 contains retained Run029 measurements, not
results from this run. Its revision is committed as `24b2854`.

## Question and approved contract

Measure the full-model contribution of skipping at a, m, h, z, QK and PV for
one final T7/Pall checkpoint. QK covers q and k jointly because they share
matrix instructions. See the complete
[approved design](../../analyses/024-2026-09-17-h-only-kernel-latency/PER-SITE-LATENCY-DESIGN.md).

The exact Run029 c30 final checkpoint is retained: random Pythia-14M,
seed1234, step712, 1493172224 training input tokens, original AdamW and data
order. One-sided gates at a,m,h,z; symmetric gates at q_post,k_post,v;
kappa0.5, training orthogonal-L1 pressure at all seven sites, lambda=b=1.
No optimizer or backward pass runs here. Weights, gates and thresholds never
change across execution controls. Weight SHA256:
`f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425`.

BF16, batch1, length2048, uncached causal full-model inference with all50304
logits. One RTX5090 with no concurrent scientific workload. Pinned Run029
Python3.12/PyTorch2.11.0/Transformers5.12.1/CUDA12.8 environment and exact lock.
Runtime seed2801; timing seed2504; same64 validation identities, seven paired
passes, three fresh processes. Correctness covers all338 complete validation
blocks from500 MiniPile documents:692224 input tokens,691886 prediction tokens,
1444-token excluded tail. Original logit atol0.25/rtol0.02/relative-L2 0.02
and pooled-loss atol0.001 apply without relaxation.

## Implementation and interpretation

- `01_prepare.py` copies the hash-verified Run029 archive and only c30's
  checkpoint/data provenance; creates explicit, count-checked source changes.
- `controls.py` defines ten modes: untouched frozen K050; all controls enabled;
  all off; projections only; and six leave-one-operation-out modes.
- `site_controls.py` and `candidate/` retain the joint h/z fusion and attention
  schedule. Independent compile-time h/z switches disable only the selected
  site's short-row/empty-tile path. QK/PV switches affect only their respective
  matrix instruction predicates. a/m use existing independent wrapper flags.
- `02_benchmark.py` preserves Run029 timing and eager-anchored qualification;
  `diagnostics.py` extends its independent counts to each switch combination.
- `06_cuda_checks.py` requires 20 synthetic h/z output/counter cases, including
  empty, short, mixed, dense and unsafe-small values; equal h/z settings must
  match the original kernel bitwise. Then `03_execute.py` requires all ten
  full-model smoke modes before the 30 scientific processes can run.
- `07_reduce.py` reports per-process spread, full/frozen fidelity, and
  conditional saved milliseconds and speedup ratios. Non-overlapping observed
  full/frozen process ranges require review before K050 attribution; overlap
  is descriptive, not an equivalence proof. Conditional contributions do not
  sum to total gains. The run does not isolate individual memory/softmax/check
  overheads or allocate overlapping QK savings separately to q and k.

Positive h/z effects and null/negative QK/PV effects would support the proposed
explanation for this checkpoint/kernel/workload. Opposite signs, negligible
effects relative to process variation, or failure to reproduce frozen K050
limit or refute it. No other recipe, threshold, model size or device is tested.

## Verification before launch

- Focused CPU tests: **19 passed** (1.00s), covering masks, signed conditional
  effects, independent sparse/short-row count cases, checkpoint identity,
  coverage and the exact process matrix.
- Full bootstrap suite: **242 passed** (8.06s).
- Original source/input and derived-code hash checks pass; bundle inventory
  and receipt are retained under `bundles/`.
- CUDA compilation/direct tests/smokes have **not run**: this Windows machine
  has an RTX5070Ti Laptop with12GB, no WSL distribution, and does not match the
  approved RTX5090 hardware. GPU qualification is the first bounded cloud phase.
- Planned smoke:20 direct h/z cases, then10 full-model processes, each with
  four timing inputs/two passes, eight correctness blocks and eight diagnostic
  blocks. Smoke observations will never be promoted as paper results.

## Proposed launch envelope

Live read on19 September: no active Pods; Community RTX5090 stock NONE.
Secure RTX5090 stock LOW at **USD0.99/hour**, in EU-RO-1 and EUR-NO-1.
The existing100GB `sparsity-spillover-shared` volume in EUR-IS-1 is left unchanged.
Raw catalog responses are retained under `prelaunch/`.

One Secure RTX5090, GPU count1, minimum host CUDA12.8, proposed name
`run037-operation-latency-001`;20GB container plus25GB Pod volume mounted at
`/workspace`. Use the previously verified Run033 image:
`runpod/pytorch@sha256:0a360022e8de4375af99430f84e8b38951acc397252163a37ceac7204d01be35`.
Install the exact archived environment in a run-local venv. Only SSH is exposed.
No network volume, endpoint, template or registry is created.

Expected ETC **25-50 minutes**, dominated by environment installation and
compiling the independent controls. Historical c30 scientific processes took
14.0-21.9s, including7.8s for a full diagnostic pass;30 warm processes therefore
need roughly8-12min with all ten diagnostic passes, before setup and transfer.
This is an evidence-based estimate, **not a calibration of the new CUDA code**.
Peak historical allocation was3.034GiB, giving ample headroom on32GB.
Recompute ETC after the smoke phase; investigate if the90min envelope is at risk.

Proposed maximum: **90 aggregate GPU-minutes and USD2 incremental total**,
including temporary storage and infrastructure retries. At0.99/hour,90min
GPU costs1.485;45GB at0.10/GB/month adds about0.009 for that duration.
Expected cost is approximatelyUSD0.42-0.83. RunPod charges no ingress/egress
fees ([provider pricing](https://docs.runpod.io/pods/pricing), checked19 September).
Existing unrelated volume charges are not part of this incremental cap.

## Monitoring, retention and teardown

Run setup/execution detached with logs on the Pod volume. Arm independent
on-Pod (`10_deadline_guard.py`) and hidden local (`11_local_stop_guard.ps1`)
stop guards for the absolute90min deadline; reserve the last10min for retrieval.
Guards stop compute and preserve volume data; verified normal teardown deletes
the new Pod. A stopped Pod's storage still costs money, so resolve retrieval
and deletion immediately rather than leaving it unattended.

Monitor every60s (sooner if ETC is shorter): completed processes/blocks,
loss and numerical errors, throughput, memory and refreshed ETC. Stop progression
on nonfinite outputs, counter mismatch, failed qualification, CUDA errors,
stale progress, or a projected budget overrun. No scientific retry or source
change is silently substituted into this run after execution.

Retain raw paired host/device timings, all per-block output checks, pooled
loss, per-operation issued/bypassed/scalar counts, exact/near-zero counts,
activation RMS, row occupancy, weight norms, source/environment/config and
checkpoint/cache hashes. The final checkpoint remains verified locally;
gradient interactions are training-only and are not reconstructed here.
Transfer all completed/failure outputs, source/config snapshots, logs,
diagnostics and reductions. `08_collect.py` packages even incomplete outcomes;
`09_verify_retrieval.py` verifies archive and every file before teardown.
Confirm required counts/coverage, then delete only this new Pod and check for
unintended billable resources. Preserve the pre-existing network volume.

The launch approval should also confirm this diagnostic inventory is sufficient
or name any additional measurement needed before the GPU is released.
