# Proposed 14M tile-bypass and short-row ablation

Status: design and diagnostic retention explicitly confirmed on 25 September
2026. [Run056](../../runs/056-2026-09-25-pythia14m-mechanism-ablation/README.md)
implements this contract. Separate launch approval was received and all 15
scientific processes completed and qualified on 25 September. Artifacts were
verified locally and the Pod deleted. See the [result and interpretation](../../runs/056-2026-09-25-pythia14m-mechanism-ablation/observations/002-interpretation.md).
The approved design below is retained.

## Question and manuscript connection

For the retained Pythia-14M T7/Pall checkpoint at kappa=0.5, how much does
empty-tile bypass reduce full-model latency conditional on short-row execution,
and how much does short-row execution help conditional on tile bypass?
Measure their joint benefit and interaction as well.

The affected claim is the h/z mechanism description and Table
`tab:kernel-paths` in [the kernel appendix](../../manuscript/draft/12-appendix-kernel-validation.tex).
[Run037](../../runs/037-2026-09-19-pythia14m-operation-latency/README.md)
toggles both mechanisms together at each site and cannot separate them.
This follow-up tests execution mechanisms jointly across h and z; it does not
repeat the per-site study or introduce a new training intervention.

The operational term is **short-row execution**, including zero rows that
produce bias-only projection outputs and rows with one or two nonzeros that
execute scalar products. Calling this entire path scalar arithmetic would
obscure the zero-row case. Removing short rows can make additional tiles empty;
mechanism effects therefore need not be additive.

## Fixed scientific contract

- Exact Run029 c30 final checkpoint, also used by Runs037/039: six layers,
  hidden width128, FFN width512, vocabulary50304. Its original pretraining used
  random initialization from the pinned Pythia architecture, seed1234,
  step712 and 1,493,172,224 input tokens. The archived initialization, data order,
  AdamW recipe and training provenance remain authoritative.
- Weight SHA256:
  `f83aef36ddbddbd94efb915d574c4645c687206bb68f51886ffc6e979985b425`.
  This is inference on an existing checkpoint: zero new training steps,
  optimizer updates, initialization draws or checkpoint selection.
- T7/Pall: one-sided threshold gates at a,m,h,z; symmetric threshold gates
  at q_post,k_post,v; kappa=0.5 throughout, with equality surviving. The
  checkpoint was trained with orthogonal L1 at all seven sites, lambda=b=1.
  No pressure is applied during inference. Gates and weights stay fixed.
- Start from frozen K050, including its K049 joint h/z implementation.
  Preserve the joint projection/residual fusion, 8-row grouping and padded
  matrix-instruction geometry, static weight representations, numerical guards,
  arithmetic order and rounding. All a/m/QK/PV and other model paths stay fixed.
- BF16, batch1, sequence length2048, uncached causal inference, full50304
  vocabulary logits at every token, CUDA graphs. Use one physical RTX5090 for
  the entire comparison, with no overlapping scientific GPU workload.
- Match the archived Python3.12/PyTorch2.11.0/Transformers5.12.1/CUDA12.8
  environment. Any necessary infrastructure difference must be documented.
- MiniPile and tokenizer identities follow [DATA.md](../../research/DATA.md).
  Validation cache SHA256:
  `51cd758fda72f14383da30c358a895d0223c0d1d80b31455d2d842c3656d0451`.
  Use all500 documents and all338 complete blocks:692224 input tokens,
  691886 next-token predictions, and a reported1444-token excluded tail.

## Independent mechanism definitions

**B: empty-tile bypass.** Inspect support and omit an empty 8x16 activation
tile from the remaining matrix path before requesting its weights or issuing
its matrix instructions. With short-row execution enabled, support is taken
over the rows still assigned to matrix execution; otherwise it includes all
rows. Retain the existing guarded fallback behavior for unsafe values.

**S: short-row execution.** Subject to the existing numerical guards, route
rows with zero, one or two nonzeros to the existing short-row calculation.
Exclude those rows from matrix operands. If an entire 8-row projection group
is handled this way, omit that projection's matrix fallback; if both h and z
groups are handled, retain the existing fused early return. These whole-group
eliminations belong to S by definition, and are counted separately.

With B disabled and S enabled, any projection group that still contains a
matrix-path row traverses every K16 step and issues the corresponding weight
loads and matrix instructions, even when a particular tile is empty. Do not
force matrix execution for a projection group whose outputs S already completed.
This defines S-only without an artificial redundant whole-group calculation.

Shared inspection is incurred when needed by either mechanism and is included
in latency. There is no attempt to assign its cost to one mechanism. Disabled
features may be compiled out; record compiler/resource differences rather than
assuming that source-level toggles preserve scheduling costs exactly.

| Mode | B | S | Meaning |
|---|---|---|---|
| t00 | off | off | Same fused implementation's dense matrix fallback |
| t10 | on | off | Tile bypass using all rows; no short-row substitution |
| t01 | off | on | Short rows plus unskipped matrix traversal for remaining groups |
| t11 | on | on | Both mechanisms, matching the retained K050 policy |
| frozen | original | original | Untouched K050 anchor for correctness and timing fidelity |

These switches apply jointly to h and z in all six layers. There are five
measured modes, including the anchor, not five training runs. No additional
per-site factorial or threshold sweep is included.

## Timing, qualification and estimands

Three fresh processes per mode:15 scientific processes. Reuse runtime seed2801,
the exact64 validation timing identities selected by seed2504, and seven paired
timing passes. Randomize process order with seed2504 and within-process paired
order with the inherited replicate seed convention. Each process also measures
the same checkpoint's native PyTorch/SDPA graph as an environmental reference.
The primary differences use measured custom-mode times, without normalizing
them to a historical Base latency.

Report geometric-mean synchronized host latency, raw host/device samples,
per-process means and observed ranges. Include embeddings, all blocks, final
normalization, full logits and input-dependent inspection. Exclude compilation,
static weight preparation, graph capture and equal input staging, matching the
paper. Instrumented work-count passes are separate from timings. Inner repeats
are not independent process replicates; range-based difference spans are
descriptive, not confidence intervals.

Every process covers all338 validation blocks against the unchanged eager
reference and qualified native graph. Retain the existing finite-output,
logit atol0.25/rtol0.02, per-sequence relative-L2<=0.02 and pooled-loss
absolute-difference<=0.001 requirements without relaxation.

For the new mechanism attribution, additionally require bitwise equality to
frozen K050 for full-validation logits and post-gate h/z operands/zero masks.
Check operands in untimed passes. A failure stops attribution rather than
silently accepting different threshold decisions. Require t11 to reproduce
the frozen anchor within observed process variation: non-overlapping process
ranges trigger fidelity review; overlap alone does not prove equivalence.

With tBS denoting full-model latency and positive differences denoting savings:

- Tile benefit with S enabled: `t01 - t11`.
- Short-row benefit with B enabled: `t10 - t11`.
- Joint benefit: `t00 - t11`.
- Standalone tile/short-row benefits: `t00 - t10` and `t00 - t01`.
- Interaction: `I = t10 + t01 - t00 - t11`. Positive I means the joint benefit
  exceeds the sum of standalone benefits. The two conditional benefits sum
  to the joint benefit plus I, not generally to the joint benefit.

Report signed microsecond differences first. Any percentage reduction names
its own comparator explicitly. Do not allocate a unique percentage of joint
savings to each mechanism or attribute the entire PyTorch gap to sparsity.

## Diagnostics and retention

Retain the existing inference diagnostic package and the following mechanism
details, by site/layer and pooled from integer counts over all338 blocks:

- Full row-nonzero histograms, explicitly separating zero, one, two and more
  than two nonzeros; numerical eligibility and guard rejection counts.
- Original empty-tile support and support after removing eligible short rows;
  original empty tiles versus tiles newly emptied by that removal.
- Whole projection groups completed by S, joint h/z early returns, issued
  matrix instructions, empty-tile omissions in remaining groups, scalar
  products and total padded matrix-work potential. Use disjoint execution
  categories with whole-group S elimination taking precedence; preserve the
  original support counts separately rather than treating them as additive.
- Independent operand-based verification of instrumented counters and control
  semantics. Retain source-level weight-request estimates with declared units;
  these are not measurements of cache misses, DRAM bytes or memory latency.
- Exact/near-zero activation counts at epsilon0,0.001,0.01; activation sums,
  squared sums, RMS/L2 and nonfinite counts; per-layer parameter weight norms.
- Preserve canonical logical-product evidence and its precision/provenance.
  Inherited BF16 activation-only opportunity diagnostics exclude weight zeros
  and probability underflow and must remain labeled as a lower bound, not
  renamed R_model. Kernel padded-work counters are a different estimand.
- Raw timing samples/order, per-block output checks and pooled loss, environment
  and compiler/resource reports, source/config hashes, checkpoint/cache hashes,
  logs, failed outcomes, and complete reduction provenance.

The original final checkpoint and cache remain retained locally. Training
gradient conflict/OL1 boundary records remain source-run artifacts; no backward
pass or reconstruction is proposed. Gate sites/thresholds stay fixed and no
new clipping frontier is included. Confirm whether any additional post-hoc
measurements are needed before launch.

## Decision criteria, deliverables and limits

Positive conditional savings robust to observed process variation support a
runtime contribution from the respective mechanism. Null or negative effects
limit or refute that contribution for this setting; interaction may explain
why standalone and conditional effects differ. Numerical, count or fidelity
failures prevent attributing results to the retained implementation.

Deliver a five-mode latency table, conditional/standalone/joint effects and
interaction, machine-readable reductions, and a compact PDF with its observation
and source script. Results are scoped to this checkpoint, device, fused layout
and full-sequence workload. The custom dense fallback includes padding and is
not claimed to be the best dense implementation. This is not a decomposition
of memory time versus arithmetic time. No manuscript or supplement edits are
automatic consequences of the experiment.

After design confirmation, create the next numbered run and implement only
these controls. Verify mathematical behavior, routing, counters, serialization
and full bootstrap tests; prepare GPU numerical/control smokes. Then present
resource fit, ETC, exact execution scope and a current launch envelope for
approval. RunPod availability/prices are queried at that stage; historical
budgets are not reused. Recover and hash-verify all agreed artifacts before
teardown. No cloud resource has been created for this proposal.
