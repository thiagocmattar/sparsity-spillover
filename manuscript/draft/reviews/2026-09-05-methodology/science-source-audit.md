# Scientific source audit for the methodology draft

5 September 2026. Read-only audit of the old manuscript methodology against
`research/METHODS.md`, `research/METRICS.md`, the definitions, and the current
implementation in `src/sparsity_research/`. No manuscript or scientific code
was changed. The paper may rename the reported metrics to a stylized S while
preserving the existing artifact keys and numerical definitions.

## Highest-priority pitfalls

1. **Metric scope is counted scalar multiplications, not all computation.**
   The numerator covers six block matrix-product families. The denominator
   adds the dense LM head, with no zero credit for that head. It omits
   normalization, activations, softmax, RoPE arithmetic, bias/residual additions,
   selection/packing overhead, and memory traffic. Avoid calling the fraction
   “removed FLOPs,” “all compute,” “speedup,” or an exact cost model.

2. **Observed sparsity is not bounded by the selected-site reach ceiling.**
   The measured numerator includes actual zero operands in all six block
   operations, including natural zeros outside the topology-selected sites.
   The ceiling is a different object: the all-zero operation reach of selected
   sites. A0 can therefore have positive observed sparsity despite zero reach.
   A ratio of observation to selected-site reach is not guaranteed to lie in
   [0,1] and should not be called bounded utilization.

3. **Pressure is applied to post-gate tensors and is tensor-weighted.**
   Capture records the actual output of the selected gate or identity port.
   L1 takes the mean absolute value within each site/layer tensor and then
   an equal mean across tensors. Larger tensors do not receive proportionally
   larger weight. Changing the pressure set therefore changes the objective
   and the weight of each previously selected tensor.

4. **Gate equality and gradients matter.** Fixed gates retain equality;
   their masks are detached. A surviving entry receives derivative one,
   including at the threshold. ReLU has derivative zero at zero, so one-sided
   thresholding at zero and ReLU agree in values but not at that boundary.
   Symmetric thresholding at zero is exactly the identity in values and
   gradients. At `h`, the selected gate replaces GELU, rather than following it.

5. **OL1 is a specific post-AdamW correction, not a guarantee about loss.**
   Task gradients are clipped before the task-only AdamW update. Updated task
   moments precondition the unclipped pressure gradient. A single global
   conflict-only projection and positive trust budget determine a correction
   applied after AdamW. The geometry excludes weight decay and per-group
   learning-rate multiplication. It is not raw-gradient orthogonality or
   first-order protection of the task loss.

## Main text versus appendix

| Subject | Main text must retain | Formal/reproducibility appendix should retain |
| --- | --- | --- |
| Model and sites | Randomly initialized Pythia-family causal models; parallel attention/FFN branches; exact identities of sites used in contrasts or a pointer to their diagram; gates and pressure targets are independently specified | Forward equations, shapes, distinct attention/MLP LayerNorms, QKV split, partial-RoPE order, `z` before output projection, topology table and per-site gate mappings |
| Gates | Gate family and threshold are distinct from topology; selected `h` replaces GELU; pressure is evaluated after the gate | Piecewise formulas, equality survives, detached-mask derivatives, ReLU-at-zero difference, symmetric-zero identity, mapping repeated in every layer |
| Pressure | L1 versus OL1 are different interventions; post-gate equal-tensor objective; OL1 uses task-only AdamW and a bounded correction | Exact averaging, accumulated microbatch gradients, clipping sequence, optimizer moments, eligible-parameter set, stabilized projection, trust ratio/scale, update order |
| Attribution | Matched contrasts state what differs; changing gate and pressure sets together identifies a recipe contrast | Condition/config crosswalk when result sets are introduced; exact matches, pressure weights and budgets; missing A7+OL1@4 contrast when relevant |
| Sparsity measure | A fraction of declared scalar multiplications with zero activation operands; all counted block families; dense LM head retained; exact zeros distinct from near-zero mass; count-first pooling | Linear, QK, and PV integer formulas; valid causal mask; actual post-RoPE operands; workload/precision; local zero and RMS formulas; operation denominators |
| Reach ceiling | Analytic selected-site reach, architecture/workload-dependent; distinct from observations and runtime; not an upper bound on all observed zeros | Reach union, no double-counting Q/K, V-to-context closure, block/model formulas, full-sequence assumption, explicit unit |
| Evaluation | Paired loss and logical count from the same checkpoint/forward configuration; measured runtime is separately evaluated | Full MiniPile coverage, shifts in next-token loss versus input-token accounting, exact cache identities, eager uncached unpadded attention, exclusions, condition-specific protocols |

The main text can be compact without sacrificing these distinctions. Most of
the old manuscript's forward equations and per-topology analytic table belong
in the appendix. A complete experimental schedule need not be invented before
the later results/experimental-setup stage.

## Exact definitions verified in code

### Sites and gates

Sources: `sites.py`, `pythia.py`, `capture.py`.

- `a` and `m` are separate LayerNorm/gate outputs in Pythia's parallel
  branches. `h` is the FFN nonlinearity/gate output before the down projection.
- `q_pre/k_pre` are immediately before partial RoPE; `q_post/k_post` are
  immediately after it. `v` is the value operand after splitting fused QKV.
  `z` is concatenated attention context immediately before the output affine
  projection. Captured values include their gates.
- One-sided gate: retain x when x >= kappa, otherwise replace it with zero.
  Symmetric gate: retain x when abs(x) >= kappa, otherwise replace it with zero.
  Kappa is finite and nonnegative. Neither is a top-k gate or straight-through
  estimator. Equality survives in both forward and detached-mask backward.
- Standard PyTorch ReLU supplies its own backward convention at zero. A gate
  at `h` replaces stock GELU. The symmetric gate special-cases kappa=0 as an
  exact identity.
- PRE-RoPE marginal zeros cannot be credited directly to QK; actual post-RoPE
  zeros must be counted. An isolated zero can be repopulated by rotation with
  its partner coordinate, while a completely zero Q or K remains zero.

### Activation objective and optimizer boundary

Sources: `pressure.py`, `optimization.py`, `capture.py`.

For J captured site/layer tensors A_j, the pressure objective is

`L1 = (1/J) * sum_j mean(abs(float32(A_j)))`.

The objective uses post-gate activations. Consequently rejected gate entries
have no input gradient through the mask, and the absolute-value derivative at
zero is also zero. The capture is not a pre-gate penalty hidden behind a
post-gate diagnostic.

- **Naive L1:** each microbatch contributes task loss plus lambda times L1;
  accumulated combined gradients are globally clipped before AdamW. Separate
  task and unweighted pressure gradients are retained for diagnostics.
- **OL1:** task and pressure gradients are accumulated separately over the
  same microbatches, dividing each microbatch contribution by their number.
  Task gradients alone are globally clipped and consumed by AdamW. The
  current reference clipping threshold is one; run-specific configurations
  remain the authority for executed settings because shared code also permits
  explicit clipping disablement.
- After AdamW updates its task-only moments, reconstruct
  `u = m_hat / (sqrt(v_hat) + adam_eps)` and
  `v = g_pressure / (sqrt(v_hat) + adam_eps)`.
- Only parameters with task gradient, pressure gradient, and initialized
  AdamW state are eligible. The pressure gradient has no first-moment buffer.
- Let `c = <u,v>` and `q = ||u||^2`, pooled over eligible parameters. If
  `c < 0` and `q > eps`, apply `v_safe = v - c*u/(q+eps)`; otherwise retain v.
  The epsilon means even the conflicting case is only approximately
  orthogonal: its exact remaining dot product is `c*eps/(q+eps)` in exact
  arithmetic, before floating-point error. Do not assert a strict
  nonnegative-dot guarantee.
- Let `r = lambda*||v_safe||/(||u||+eps)` and, for r>0,
  `s = min(1, budget/(r+eps))`; when r=0 use s=1. The current config parser
  requires a strictly positive budget for OL1. The correction is
  `theta_p -= learning_rate_p * lambda * s * v_safe_p` after AdamW.
- Projection and norm bounds concern the adaptive vectors before group
  learning rates. Decoupled weight decay is part of AdamW's preceding update
  but excluded from the projected task direction. Do not call this the exact
  full parameter displacement or assume different group learning rates
  preserve the same global angle.

### Exact-zero product counters

Sources: `metrics.py`, `logical_capture.py`.

For X with shape [n,p] feeding a linear projection with q outputs:

`Z = q * count(X == 0)` and `N = n*p*q`.

Weight zeros are not counted. For QK and PV, both operands are activations and
the zero event is their logical OR, counted once when both operands are zero.
Only pairs u<=t are valid; masked future positions are excluded from numerator
and denominator. Valid probability zeros, including floating-point underflow,
are counted. Value coordinates at earlier key positions participate in more
valid products, so unweighted local value sparsity does not determine PV
sparsity. The attention-output projection uses actual post-gate context zeros,
not a propagated estimate from partial V masks.

The six block families are QKV projection, QK scores, probability-value
products, attention output projection, FFN up projection, and FFN down
projection. If Z sums their zero-product counts and N sums their complete valid
product counts, then the existing artifact definitions are

`R_block = Z/N`,

`R_model = Z/(N + N_input_tokens*d*V)`.

The manuscript can display these as stylized S with the same subscript and
units. They are fractions in [0,1]; multiply by 100 only when displaying
percentages. Do not silently combine the old manuscript's percent-valued
formulas with the artifacts' fraction-valued numbers.

### Analytic selected-site reach

Source: `ceilings.py`.

For one uncached full sequence, one layer's denominator is

`T*(4*d^2 + 2*d*d_f) + d*T*(T+1)`.

For L layers, add the dense head `T*d*V`. The two attention terms each contain
`d*T*(T+1)/2` products. Pythia uses d_f=4d in the studied configurations.

The reachable set is the union of these rules:

| Site(s) identically zero | Counted operations reachable |
| --- | --- |
| a | QKV projection |
| m | FFN up projection |
| h | FFN down projection |
| q_pre, q_post, k_pre, or k_post | QK, credited once |
| v | PV and attention output projection |
| z | attention output projection |

V=0 implies context PV=0 and hence zero input products in the attention output
projection. The output projection's bias does not invalidate its own
zero-product credit but prevents interpreting its entire affine output as
zero. Likewise a=0 does not imply Q=K=V=0 in the presence of QKV biases;
q=0 yields zero scores rather than zero softmax probabilities. No additional
transitive reach is assumed. PRE and POST have the same all-zero reach despite
different partial-mask behavior.

The ceiling is reachable numerator divided by the same model denominator. It
does not depend on a checkpoint or quality constraint. It is not the maximum
achievable sparsity at acceptable quality, a universal observed upper bound,
or a hardware ceiling.

### Validation

Source: `evaluation.py`, operational data/metrics contracts.

The default diagnostic evaluates all 338 complete 2,048-token blocks from all
500 MiniPile validation documents, accounting for 692,224 input tokens and
excluding the 1,444-token tail. With labels=input_ids, the causal task loss
uses next-token shifts within each block; matrix products still process all
input-token positions. The code weights each batch mean loss by its number
of equal-length sequences. Count pooling sums integer counts before division.
Loss and product counts used for paired claims should come from the same
eager logical pass; a terminal loss from a different attention implementation
is not automatically identical.

## Old-manuscript discrepancies to correct in the new draft

- The old source header cites another repository and historical commit. The
  new manuscript should use the current repository's operational contract and
  provenance notes, without implying that old commit defines current behavior.
- The old OL1 equations/algorithm omit the explicit task-gradient clipping
  step. Current execution includes it before task-only AdamW.
- The old manuscript makes the OL1 trust budget optional. Current parsing
  requires it to be positive.
- The old manuscript describes AMSGrad fallback. Current `build_adamw` does
  not enable AMSGrad, and current correction code reads `exp_avg_sq`, not an
  AMSGrad maximum buffer. Do not copy the unsupported generalization.
- The old topology/analytic table is labeled “executed” even though supported
  topologies and analytically evaluated architectures are not all equivalent
  to the eventual result cohort. Label a registry/map as such and distinguish
  it from condition coverage.
- The old appendix's `U_arch` terminology lacks the current warning that the
  numerator includes natural zeros outside selected reach. Omit the ratio
  unless scientifically necessary or retain the explicit unbounded-ratio
  qualification.
- Old R formulas are percent-valued. New stylized S definitions should declare
  fractional units consistently with artifacts and convert explicitly in plots.

## Review rubric for the actual new draft

No draft score is assigned in this source audit. The next two passes will use
the requested unchanged 1–5 anchors and five equally weighted criteria:

1. operational fidelity;
2. metric/ceiling precision;
3. intervention identifiability;
4. sufficient reproducibility split between main text and appendix;
5. claim rigor for later results.

Maximum 25 points. A source-audit correction does not itself authorize a new
experiment, a change to executed scientific inputs, or a new result claim.
