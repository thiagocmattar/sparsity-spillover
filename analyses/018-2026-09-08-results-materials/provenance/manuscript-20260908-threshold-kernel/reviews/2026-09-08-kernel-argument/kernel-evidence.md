# Kernel argument: evidence audit

Read-only scientific audit, 8 September 2026. No experiments or manuscript TeX edits.
The current paper cohort is Analysis 018's 30 checkpoints (c01-c30); the source
Run 029 retains a historical 35-checkpoint cohort. Use the 30-checkpoint values
below in the current manuscript.

## Recommended reasoning arc

1. **Start from the implementation problem.** The scientific objective is to
   test whether architecture-level activation sparsity can yield full-model
   acceleration. Realizing this opportunity requires kernels matched to the
   model's shapes, signed activations, numerical behavior and hardware. This
   is a stronger opening than emphasizing the authors' lack of systems expertise;
   no evidence about personal expertise is needed.
2. **Give the concrete compatibility result.** The unchanged Sakana/TwELL
   control accelerated its supported SparseLM workload, but its small-width
   Pythia-14M primitive failed correctness. The existing adapted Pythia baseline
   also failed to supply broadly qualified acceleration. This motivates
   specialization without misrepresenting the published system.
3. **Introduce agents as the method.** A human-guided Codex loop proposed,
   implemented, tested and benchmarked specialized candidates, retaining only
   numerically qualified improvements. This demonstrates the practical use of
   coding agents for this study, without an unsupported agent-superiority claim.
4. **Report early gains and later refinement.** The retrospective matched
   incumbent reaches 1.6024x at iteration 11 and 1.7830x at iteration 42.
   Substantial gains appeared early; the final improvement arrived at the last
   eligible proposal. Do not call the curve early convergence to the best kernel.
5. **Connect to model-wide sparsity.** The frozen selected implementation
   qualifies on all 30 checkpoints, averages 1.2340x and exhibits a strong
   approximately linear association with model-wide sparsity (OLS R-squared
   0.8167). This supports the metric's relevance to realized execution under
   specialization. It is neither strict proportionality nor a causal speed law.
6. **End with the informative limitation.** Fusion and projection-side sparse
   paths provide the useful acceleration; attention skipping is slower despite
   substantial issued-instruction avoidance. Scalar sparsity and profitable
   hardware granularity are distinct. The current attention kernel skips when
   either entire operand fragment is zero; it does not require matching zeros
   in both Q and K. No measured break-even condition based on joint zero masks
   exists.

## Exact evidence and source map

### Sakana compatibility and P0

- `runs/022-2026-09-04-pythia14m-sparse-kernel-a0-baseline/README.md`,
  execution outcome: supported SparseLM0.5B/H100 NVL control 476.9542 ms versus
  620.0687 ms, 1.3001x. This is the upstream benchmark workload, not the current
  full-vocabulary Pythia timer. The exact raw ELL primitive at M=256, K=512,
  N=128 had relative-L2 error 0.69055, versus the declared 0.02 bound. Execution
  stopped before Pythia validation or timing; there is no unchanged Pythia-14M
  full-model speedup from this run. Widths N=256/512/2048 worked; padding and
  slicing 128 to 256 worked but changes the operation and doubles output work.
- `runs/025-2026-09-05-pythia-agentic-sparse-kernel-search/SOURCE_AUDIT.md`:
  unchanged non-gated T2D dispatch assumes N=2048 and K=5632/8192; D2T uses
  positive-output packing and Hopper WGMMA. P0 adds signed exact packing,
  full payload capacity, predicated shape tails, current-stream launches,
  bias/accumulation adaptations. P0 is a descendant, not the published fused FFN.
- `runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/observations/04-sakana-compatibility.md`:
  upstream benchmark excludes the LM head; P0 retains dense SDPA QK/PV and
  uses native projection fallback on A0. Upstream commit is
  `661f1fc841ed84d92f3fca5e5a94cffc5bf00ee5`.
- Current `analyses/018-2026-09-08-results-materials/figure_data.json:runtime.points`:
  P0 qualifies on 4/30: c01/A0 (1.001212319x, native fallback),
  c08/A1-H-OL1 lambda .1 (0.874976406x),
  c20/A4-OL1 kappa .5 (0.879446489x),
  c30/A7-OL1 kappa .5 (0.817960663x).
  Qualified-subset geometric mean is 0.890976175x. The other 26 fail
  the elementwise logit tolerance, with finite outputs and acceptable pooled
  loss/relative L2. Do not call these execution failures or infer an all-30
  qualified P0 mean. At c30, P0's 0.8180x versus K050's 1.7832x is a matched
  implementation comparison; the original H100 control is not.

### Search progression and attribution

- `runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/observations/01-matched-autoresearch-progress.md`
  and current `figure_data.json:runtime.progress` agree:

| Iteration | Candidate | Qualified incumbent |
| ---: | --- | ---: |
| 0 | Native | 1.000000000x |
| 9 | K017 | 1.057045558x |
| 10 | K018 | 1.097161155x |
| 11 | K019 | 1.602385781x |
| 42 | K050 | 1.783029124x |

- There are 42 eligible proposals from 50 archived K proposals, with 8
  other-size/standalone exclusions. K011's 12 masks share an ordinal; 53
  eligible K configurations plus P0 were measured. The curve is a common-protocol
  retrospective replay of an adaptive human-guided history, not its original
  online feedback trace or 42 independent trials. Its monotonicity follows
  from plotting a cumulative maximum. The 1.6024x point accounts for about
  76.9% of the final speedup above 1x, but stating the two actual endpoints is
  clearer than introducing that derived statistic.
- `runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/observations/06-implementation-and-claim-audit.md`
  records configured Codex default `gpt-6-astra`, without per-request served-model
  attestation or a human/non-agent comparison. Human guidance and inherited
  FlashAttention/CUTLASS/PyTorch code should remain explicit where method is
  described. Neither agent superiority nor from-scratch kernel authorship follows.

### Current regression and ablations

All figures below were independently recalculated from current Analysis 018
`figure_data.json:runtime.points`, not copied from the historical 35-row table.

| Implementation | Qualified | Geometric mean speedup |
| --- | ---: | ---: |
| K050 | 30/30 | 1.233991707x |
| K050, attention skipping disabled | 30/30 | 1.250558753x |
| K050, sparse paths disabled | 30/30 | 1.182857977x |
| P0 | 4/30 | 0.890976175x on those four only |

- K050 range: 1.005938495-1.783174564x. The lower endpoint is close to noise.
- Unweighted OLS with intercept: speedup = 0.951295128 +
  3.770390864 * S_model (fraction), R-squared = 0.816717670, n=30.
  There are 24 inverted checkpoint pairs in this cohort; strict monotonicity
  is false. Use "strong positive association" or "approximately linear
  association," not "scales linearly" as a law.
- K050 / no-skip geometric ratio: 1.043228968x; helps 14/30. At c30,
  1.783174564 / 1.360004958 = 1.311153x. This supports a conditional sparse-path
  contribution, while fusion already provides gains. Checkpoint topology,
  trained weights, quality and native costs vary; canonical sparsity uses FP16,
  timing BF16. The fit alone does not isolate sparsity causally.
- The attention-dense control improves every checkpoint. At c30 it reaches
  1.802220272x. Thus "best" in the figure refers to the searched K candidates;
  the later ablation is faster. Disabling attention skips retains the same
  custom attention kernel rather than replacing it with stock SDPA.
- Contract: one RTX5090, BF16, B=1, uncached T=2048, all 50,304 logits; 64
  identical validation inputs, seven paired passes and three fresh processes,
  1,344 paired ratios/checkpoint. Numerical qualification covers all 338
  validation blocks. Ratios between ablation speedups use separate processes.
- Source reduction and caveats:
  `analyses/018-2026-09-08-results-materials/observations/O007-kernel-realization.md`,
  `evidence.py:runtime_subset`, and Run 029 Observation 03.

### Attention skip predicate and what it means

Frozen source prefix:
`runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/archive/root/runs/028-2026-09-06-pythia14m-all-site-sparse-kernels/`.

- `candidates/k035/sparse_gemm.h`, lines 8-37: `run028_nonzero` ORs all local
  BF16 magnitude bits and uses a warp vote. An MMA instruction is issued only
  when `active_a[m] && active_b[ns]`; otherwise it is bypassed. A fragment has
  nonzero activity if any value in its whole warp-distributed operand is nonzero.
  Therefore complete zeros in either fragment suffice. Partially sparse
  fragments still execute even where scalar products would be zero. Matching
  zero locations in both Q and K are neither required nor a measured criterion
  for profitability.
- `candidates/k035/flash_fwd_kernel.h`, lines 884-886 and 960-962 apply this
  primitive to Q and K; lines 932 and 997 apply it to P and V. Causal masking
  and softmax still execute. A zero QK score is not a zero softmax probability.
- `candidates/k035/candidate.py:27` requires shape (1,4,2048,32).
  `kernel.cu:45` uses Flash traits with head dimension 32, query tile 64,
  key tile 256, four warps; two key/value splits are fixed. The current policy
  disables the experimental zero-query prefix shortcut.
- Run 029 Observation 03 retains c30 counters: 57.9952% QK and 67.2084% PV
  MMA atoms bypassed, despite negative latency contribution. Recurring fragment
  tests, warp votes and branch handling are inside execution. This is direct
  evidence that reduced issued work can fail to repay skip overhead. It does
  not isolate the exact overhead component or establish a minimum profitable
  mask pattern. Do not claim a proved Q/K joint-zero break-even rule.
- Longer sequences, alternative fragment patterns, other devices and a
  stronger compiled-dense baseline are outside this measured result. Longer
  sequences increase attention's work share, but also repeat detection/dispatch
  overhead; no crossover or generalization has been measured.

## Compact manuscript-ready paragraph arc

Our aim is to determine whether the activation sparsity produced by these
interventions can translate into full-model acceleration. This requires an
implementation matched to the model's shapes and sparsity structure: the
published Sakana kernel accelerated its supported workload, but its unchanged
Pythia-14M primitive failed numerical qualification, and the adapted Pythia
baseline did not provide broadly qualified gains. We therefore used a
human-guided coding-agent loop to develop and evaluate specialized kernels.
A matched retrospective of 42 proposals reaches 1.6024x by iteration 11 and
1.7830x at iteration 42. The selected implementation qualifies across all 30
checkpoints, averages 1.2340x, and exhibits a strong approximately linear
association between speedup and model-wide sparsity (R-squared 0.8167).
Ablations show why specialization matters: fusion and projection-side sparse
paths provide the gain, whereas attention skipping adds overhead even while
bypassing 58.0% of QK and 67.2% of PV matrix-multiply instructions at the
search checkpoint. The attention path can skip when either complete operand
fragment is zero; the limitation is that its measured savings do not repay
the cost of detecting and handling those fragments in this workload.

This paragraph needs the existing protocol sentence and appendix references
nearby; the more detailed compatibility numbers and implementation predicates
fit the appendix better than the main narrative.
