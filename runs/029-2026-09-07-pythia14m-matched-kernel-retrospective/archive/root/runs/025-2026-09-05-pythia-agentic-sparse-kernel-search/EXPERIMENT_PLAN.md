# Scientific design and execution sequence

Status: proposed, not implemented or launched. The human may refine this
design before the evaluator and experiment code are written.

Revised for the user's $40 total ceiling and official-Sakana starting point.
This replaces the larger-budget, six-trajectory design before any execution.

## 1. Question and matched scientific contract

Can a bounded agent-driven specialization of Sakana-derived sparse execution
produce correct full-model speedups on the completed Pythia ladder, including
an attention contribution, and do those gains transfer across checkpoints,
model shapes, and GPUs?

| Field | Proposed contract |
| --- | --- |
| Models | Pythia 14M, 70M, 410M; retained final checkpoints only |
| 14M shape | 6 layers, hidden 128, FFN 512, 4 heads of dimension 32 |
| 70M shape | 6 layers, hidden 512, FFN 2,048, 8 heads of dimension 64 |
| 410M shape | 24 layers, hidden 1,024, FFN 4,096, 16 heads of dimension 64 |
| Shared workload | Vocabulary 50,304; causal length 2,048; full LM logits |
| Primary / optional | Batch 1 prefill; batch 32 sentinel checks only after core evidence, if memory/budget permit; no KV-cache decoding |
| Precision | BF16 inputs/weights as in the prior adapter; preserve declared accumulation policy and numerical gates |
| Model state | Eval/inference mode; dropout off; no optimizer or backward pass |
| Training provenance | Seed-1234 random initialization from pinned configs; approximately 1.493B tokens; exact source metadata/hashes retained |
| Variable under search | Kernel algorithm, layout, fusion, launch configuration, and legal dispatch/fallback |
| Fixed | Weights, tokenizer, data identities, gate sites/operators/thresholds, topology, causal mask, RoPE, validation coverage |
| Budget | $40 total; one trajectory, at most 20 RTX-5090 GPU-hours / 40 attempts; final validation and transfer reserved first |

A0 remains native GELU without imposed gates; A1-H retains its trained ReLU
activation at h. A4-OL1 retains its trained a/m/h/z one-sided gates. A7-OL1
also retains the signed symmetric q_post/k_post/v gates, with exactly the
operational placement in `research/METHODS.md`. OL1 is training provenance,
not an inference-time loss term. All original optimizer, LR, pressure sites,
and initial-parameter identities must be carried into the manifest; there are
zero new optimizer steps and no new model initialization.

Do not create additional sparsity, change kappa, lower precision, drop tokens,
truncate sparse rows, or substitute released pretrained weights. Compile-time
specialization on shape/dtype/GPU is allowed; memorizing checkpoint IDs,
validation token IDs, or precomputed activations/output is not.

## 2. Checkpoint and input partition

The selected ladder has 12 checkpoints per size: A0, A1-H, A4-OL1 at
`kappa={0,0.01,0.05,0.1,0.5}`, and A7-OL1 at that same grid.

| Size | Source of final weights | Search sentinels | Untuned checkpoint tests |
| --- | --- | ---: | ---: |
| 14M | Run 004 controls; Run 015 corrected A4-OL1; Run 014 A7-OL1 | 6 | 6 |
| 70M | Run 018 canonical-init ladder | 6 | 6 |
| 410M | Run 019 canonical-init ladder, not the later A0 LR screen | 6 | 6 |
| Total | 36 retained checkpoints; no retraining | 18 | 18 |

Each size's sentinels are A0, A1-H, and the kappa 0/0.5 endpoints of A4/A7.
The interior thresholds 0.01/0.05/0.1 are not queried for performance while
searching. Their historical quality and sparsity are already known, so this
is an untuned-checkpoint test, not a claim of a wholly unseen dataset.

Use 64 complete blocks sampled from the pinned **training** cache with seed
2500 for development; use the first 16 for the primary short timing suite and
all 64 in two batch-32 groups if optional checks are funded. Training here names the
input split, not new training. Keep the evaluator's input identity list fixed.

After freezing, evaluate quality on all 500 MiniPile validation documents:
338 complete 2,048-token blocks (692,224 input tokens), excluding and reporting
the 1,444-token tail. Use the existing shifted-label loss/token aggregation
contract, pooling integer counts rather than averaging batch percentages.

Final latency inputs are a prespecified seed-2504 sample of 64 validation
blocks at batch 1 and, if optional batch-32 checks execute, ten disjoint groups
of 32 blocks. The final
18 blocks still participate in complete quality/count evaluation but not the
fixed-shape batch-32 timing. Report timing coverage separately from complete
validation coverage. No final latency result may feed another candidate edit;
a scientifically changed search protocol needs a new numbered run.

The mandatory final matrix is all 36 checkpoints at batch 1 on the development
GPU, comparing strongest matched dense, the minimal Sakana adaptation, and the
frozen optimized descendant. H100 transfer and component ablations cover the
prespecified 18 endpoint sentinels. Batch-32 checks use those same 18 sentinels
only if the complete mandatory matrix and teardown are already funded; omitted
optional tests are explicitly reported, never selected by favorable timing.

## 3. Baseline audit before crediting any agent improvement

1. Start from official Sakana commit
   `661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5`. Keep its license/source intact
   and record every derived patch. Do not start a from-scratch replacement
   project or treat the old raw-ELL adapter as the only available upstream path.
2. Audit both the optimized TwELL path and the currently used raw-ELL path.
   Unsupported upstream configurations are recorded as such, not scored as
   a slowdown. Reproduce the official positive control on its supported H100
   setup; Blackwell compatibility must be tested, not assumed.
3. In the new harness, correct input rotation: Run 023's
   `03_benchmark.py:benchmark_full_model` captures `inputs[0]` during timing
   and assigns `inputs[-1]` only afterward. Its timer is fixed-input, despite
   preparing several groups. The original full-validation measurement is
   separate. Re-measure dense and sparse with actual paired input rotation;
   do not attribute this correction to search performance.
4. Re-run the 14M N=128 regression that failed in Run 022. The later narrow-N
   patch must pass the same case before any 14M performance claim.
5. Reconcile README, config, and terminal artifacts. In particular, Run 024's
   config still says `implemented_not_launched` although its README/artifacts
   record completion. Use verified artifact provenance, not that stale status.
6. Do not modify completed runs to make their historical benchmarks look like
   this one. Record all harness changes and rebuild both sides here.

Required source lineage:

- **U0:** unmodified upstream kernels and their supported official control.
- **P0:** minimum Pythia bridge and necessary correctness fixes, preserving all
  nonzeros and exact gate/attention semantics. Prefer upstream optimized TwELL
  where legal; if its assumptions require changes, list each change and why.
- **K001...K040:** agent optimization patches descending from P0. Separate
  correctness integration from performance changes and retain the actual
  upstream kernel files/configurations used by each path.

If an upstream path cannot preserve the exact contract, mark it unsupported
and use a documented exact Sakana-derived adaptation; do not silently truncate
or benchmark a known-wrong implementation. A newly added attention component
must be labeled as new code integrated with Sakana, not an upstream causal
attention kernel. The comparison does not call P0 an unmodified upstream run.

Baseline set:

- Native Hugging Face dense with the strongest correct supported fused SDPA.
- Dense adapter, to expose integration overhead.
- Optimized dense (`torch.compile` and/or graph capture where legal), with
  equal input/update and capture treatment for the sparse comparison.
- P0 minimal official-Sakana adaptation, including optimized TwELL where legal.
- Historical corrected raw-ELL as an additional diagnostic, if distinct from P0.
- A dense-only sibling of the winning adapter, retaining non-sparse fusions.

Primary speedup is `S = latency(strongest valid matched dense) / latency(candidate)`.
Report improvement over P0 and historical raw-ELL separately; beating a sparse baseline is
not beating dense. Select dense configurations on development inputs too,
then freeze them. Never silently use a slower dense baseline on the final set.

## 4. Step-by-step development ladder

### Phase A: local implementation and verification, no rented idle GPU

Create focused run-local preparation, evaluator, adapter, candidate, and
verification files after design confirmation. Reuse trusted source-run
metadata/loaders and mathematical tests, not a wholesale new model stack.
Pin an agent/model execution route and demonstrate one bounded dummy
propose/evaluate/record cycle. No generalized scheduler or research framework.

CPU tests cover manifest/hash checking, gate/hook placement, signed/all-zero
pack round trips, sparse-to-dense reconstruction, input rotation, causal
semantics, pooled metrics, candidate immutability, and artifact serialization.
Run the complete bootstrap suite before a launch proposal. CUDA tests occur
on the exact deployment hardware during the calibration gate.

### Phase B: capped calibration, at most $5 of the total $40

Allow up to two RTX-5090 hours and one H100 hour, including setup/transfer.
On RTX 5090, load one checkpoint at a time and measure memory/latency at all
three shapes at batch 1. Use A0 first for correctness, then A1-H and
high-threshold A7 for representative sparsity. Compile/test the inherited
operators and obtain a dense operator-time profile. On H100, reproduce the
upstream positive control and establish portable-kernel compatibility.

Measure setup/transfer time, compile time, one candidate evaluation, and full
validation throughput. Project the cost of the entire 36-checkpoint final
matrix before continuing. If 14M's removable dense-time fraction is tiny,
keep it as a measured limitation rather than silently dropping it.

Proceed only if the harness is trustworthy, the proposed final matrix fits
the budget, and profiling leaves a plausible optimization target. Otherwise
report a bounded negative feasibility result or return a revised design.

### Phase C: one logged, Sakana-derived closed loop

Use trajectory identifier 2501 and the existing authorized agent session.
Seal P0 and the evaluator before candidate 1. Record model/settings, prompts,
tool actions, token usage where available, human help, errors, and all trials.
Do not claim bitwise reproducibility of unseedable model generation; the
source artifact and measured result must be independently rebuildable.

Allow at most 40 candidate attempts and 20 RTX-5090 GPU-hours, counting builds,
failures, evaluation, and agent waiting time while the Pod is rented. Under a
budget-feasible RTX-PRO fallback this becomes at most 8 search hours, with the
same candidate upper bound. Never fund extra search from protected final-test
hours. Provisional allocation of the 20-hour primary search: 7 hours for W2,
4 for other projections, 6 for attention, and 3 for composition. Reallocate
between these targets only within the fixed overall budget and declared scope;
retain an attempted attention path even if its result is negative.

Provisional order, with early exits:

1. **FFN W2:** fix launch/allocation overhead, fuse exact packing where useful,
   test tiled/blocked layouts and dense crossover dispatch. This is the first
   practical target on A1-H and the cleanest connection to upstream work.
2. **Other projections:** W1, QKV, and attention output projection using the
   exact a/m/z operands. Cache static weight transforms; account for dynamic
   packing/gathering. Test GPU/shape-specific tiling only after a generic path.
3. **Actual attention:** optimize QK-transpose and softmax-weighted V within
   causal attention, not just its surrounding linear projections. Seek exact
   zero-operand/tile or special-row fast paths that preserve normalization and
   compete with fused dense SDPA. Keep a dense route for unsuitable inputs.
4. **Composition:** combine FFN/projection and attention candidates, fuse legal
   boundaries, and optimize the full model including dispatch. Do not accept
   a primitive win that disappears end to end.

Attention correctness is non-negotiable: a zero Q row has uniform probability
over causally valid keys, not zero output; a zero K vector still participates
in the softmax denominator; zero V does not authorize removing a key from
normalization. Preserve bias, residual, RoPE, and causal-prefix behavior.
Arbitrary elementwise zeros do not imply empty blocks or sparse softmax.
Approximate/top-k attention or new structured masks require a new design.

There is no no-performance-feedback arm or independent search replication in
the $40 design. This is an agent-assisted systems case study, not a test of
the causal advantage of agent feedback or superiority over humans/autotuners.
Preserve the full trajectory, including parameter-only sweeps. Test important
winning changes by rebuilding P0 and removing changes in separate frozen
ablation variants, not by discarding the agent's failed proposals.

### Phase D: freeze and untuned-checkpoint validation

Select the winner using development results only. Freeze code,
dispatch thresholds, dense baselines, and numerical tolerances. Evaluate all
36 checkpoints with full quality coverage for dense, P0, and the final path.
Protect up to eight RTX-5090 hours for this evidence phase; shorten search
before entering it if calibration forecasts a larger requirement.

On the prespecified 18 sentinels, run four contribution modes: dense-only,
FFN/projections sparse with dense attention, sparse attention with dense
projections/FFN, and combined. The LM head remains dense in every mode and
is included in latency. If exact sparse attention never beats SDPA, report
that failure; a fast FFN-only path does not validate an attention claim.

### Phase E: frozen hardware transfer, no retuning search

Reserve the remaining three H100 hours, after the one-hour calibration share,
for all 18 endpoint sentinels at batch 1: dense, P0, and the frozen portable
RTX-5090 winner, with full validation and paired timing. This allocation
includes setup/reconnect, transfer, verification, and teardown. Do not rent an
H100 merely to leave it idle while the coding agent develops on RTX 5090.

Cross-evaluate generic and specialized configurations on both GPUs where
their instruction sets are legal. Mark architecture-specific code as
unsupported rather than inventing a cross-device timing. Do not retune after
seeing H100 results. Separate algorithm portability, configuration transfer,
and measured speedup transfer. Do not compare a newly tuned Blackwell result
only with old H100 measurements.

### Phase F: verified closeout

Retrieve logs, candidate sources, measurements, profiles, and hashes before
teardown. Write an observation and PDF figures in this run folder, with
source scripts and an observations index. Report all failures/unsupported
conditions, selected trials, and cumulative cost. Consolidate a cross-run
analysis or manuscript result only after the human approves that next task.

## 5. Measurement, objective, and correctness gates

### Timers

Primary timer is synchronized host wall time around a complete forward pass
with resident GPU input and full logits, including dynamic gates, packing,
metadata updates, dispatch, sparse and dense operations. Declare this a
resident-input inference measurement, not service-level request latency.
Measure input transfer/model load/compilation/cold start separately for ETC.
Static reusable weight transforms are allowed, with their setup cost reported.
Input-dependent work cannot move outside the timer. Record CUDA-event timings
and kernel profiles as auxiliary diagnostics, not substitutes for that timer.

Rotate actual inputs and randomize dense/candidate order within each paired
block. Warm up every candidate and input/shape path. For final results use
seven paired passes over the declared input groups and three fresh-process
sessions for claimed winners. Record GPU UUID, software, power/clock/thermal
state, utilization, VRAM, and other device activity. No concurrent benchmarks
on one GPU. Keep capture/static-shape treatment matched between paths.

### Objective and promotion

The primary search score is geometric-mean full-model speedup across the 15
non-A0 endpoint sentinels at batch 1, weighting every size and condition
equally. A0 is a no-regression control: no more than 2% median slowdown after
confirmation, normally served by a legal dense fallback. Report all 18
sentinels, not just the ones in the objective. Batch 32 is secondary and its
regressions cannot be hidden by mixing batches into the score.

Provisional promotion requires a >=2% score improvement over the incumbent,
correctness, and reproduction in three fresh paired timing blocks. A final
practical speedup claim targets median >=1.05x with a paired-bootstrap 95%
interval above 1.0, resampling input groups/sessions rather than individual
dependent kernel invocations. Report uncertainty even if this threshold fails.
Five percent is a decision threshold, not an expected outcome. The primary
score, per-size scores, worst case, and sparse-executed fraction all remain
visible; a selective win does not imply every checkpoint sped up.

### Numerical gates

Use the same dense mathematical function and source checkpoint. Before the
search, seal adversarial primitive and composed-model tolerances in the
evaluator. Retain the prior relative-L2 ceiling of 0.02 for primitive checks
as an upper limit, supplemented with absolute-error tests calibrated against
the FP32 reference/BF16 dense rounding envelope; zero-reference cases need an
absolute criterion. Publish the exact numerical bounds before trial 1.

Check signed data, all-zero/dense inputs, ragged widths including N=128,
padding, biases, long causal prefixes, all-zero Q/K/V cases, and repeated
workspace use. Compare intermediate outputs and logits, not loss alone.
Full-model finalists must satisfy absolute mean validation-loss difference
<=0.001 nat/token against matched dense on complete validation coverage,
with no nonfinite outputs. A tighter preflight tolerance may be chosen before
search, never loosened after seeing a candidate. Any failed condition prevents
an all-checkpoint correctness claim; dense fallback must itself be tested.

## 6. Diagnostics and analysis

Retain exact and near-zero integer counts at a/m/h/z/q_post/k_post/v, activation
RMS/L2, per-layer weight norms, row/tile occupancy and length distributions,
packing bandwidth, workspace allocation counts/bytes, dispatch decisions,
per-operator latency, memory peaks, and complete validation loss. Retain the
existing declared near-zero threshold grid in the manifest; add thresholds
only before launch, not retrospectively to favor an interpretation. The
inherited Run-023 grid is absolute thresholds 0.001 and 0.01, alongside exact
zero at 0.0; preserve the source comparison operators in the counter tests.

Keep canonical `R_model`, `R_block`, their integer numerators/denominators,
and architecture/topology `R_model_max` counts/unit unchanged. Separately
measure actual BF16 operands and the opportunity covered by operations that
really execute sparsely (`R_covered`), including attention when implemented.
Record actual skipped work/branch coverage, not just a nominal module list.
Use measured dense operator-time shares for an Amdahl sanity check; the
logical product share is not itself a wall-time share.

Final tables include every checkpoint, kappa, loss, `R_model`, `R_covered`,
latency/speedup, GPU, batch, mode, correctness status, and uncertainty. Figures:

- Best-so-far full-model score versus cumulative GPU cost/time for the complete trajectory.
- Full-model speedup versus `R_model`, showing all realizations by size/GPU.
- Dense/raw-ELL/optimized/component-ablation latency and overhead breakdown.
- Untuned-checkpoint generalization and frozen-kernel hardware transfer.

Use the existing analysis-010 publication style; PDF only. Show descriptive
OLS R2 and Spearman association within size/GPU/batch and, where possible,
within topology across five kappas, with leave-one-out sensitivity. Small
samples and shared checkpoints limit inference; do not treat repeated timings
as new independent training seeds. Report quality trade-offs alongside speed.

No new training gradients, gradient conflict, or OL1 boundary metrics can be
recovered from inference. Link retained training diagnostics where available;
do not simulate their collection. No new TEAL clipping frontier is required
for this exact-kernel study; prior post-hoc quality frontiers remain separate.

Minimum paper package: a rebuildable U0/P0/final source chain; a full-model
table for all 36 primary checkpoints; complete quality/count provenance;
paired timing uncertainty and fresh-process confirmation; 18-sentinel
FFN/attention contribution tests; the matched fixed-kernel H100 comparison;
and a complete trial/cost history. If any mandatory element is not completed
within budget, label the study partial and narrow the claim. Do not promise
paper-level validation of every link merely because the budget was exhausted.

## 7. Decisions to confirm before implementation/launch

Confirm the exact-prefill scope (rather than a new decoding/pruning study),
the 18/18 checkpoint partition, and the reduced single-trajectory claim. Confirm
the diagnostic inventory above or add measurements needed later; original
final checkpoints and cache identities remain retained regardless.

Before launch, also pin the agent model/version, effort/context policy,
token/call accounting and zero-new-API-charge route (or an allocation inside
the same $40 ceiling); the CUDA/container
matrix; exact per-output numerical tolerances; measured memory fit; checkpoint
inventory; and the refreshed per-Pod cost/deadline. Unknown values here are
explicit implementation/calibration gates, not permission to choose different
scientific interventions silently.
