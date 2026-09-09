# Scientific audit for splitting definitions from experimental setup

5 September 2026. Read-only audit before the structural revision. Sources:
current manuscript methodology and appendix, the ladder source, Analysis 013
README, Analysis 011 README, and the top-level configs of Runs 004, 009, 011,
013, 014, 015, 018, and 019. No manuscript, result, or scientific-code change
was made by this reviewer. Config details below are verified available setup
facts, not a selection of a new result cohort or permission for experiments.

## What belongs in each section

**Methodology:** independently defined gate and pressure sets, generic gate
families and post-gate pressure, naive L1 versus OL1, the declared
zero-multiplication estimand and pooling, and selected-site all-zero reach.
These definitions should remain interpretable before a reader knows A0/A1/A4/A7.

**Experimental Study:** Pythia and MiniPile, the concrete A* recipes, which
comparisons they support, same-size matching, the actual scope of the size
extensions, and validation/setup choices. A compact appendix can retain
condition tables, full training settings, exact config revisions, and coverage
details. Merely splitting sections does not authorize creating a new design.

**Formal appendix:** gate derivatives, tensor averaging, AdamW correction,
integer attention/linear counters, reach union and closure, and denominator
derivation. It may refer forward to concrete recipe definitions rather than
duplicating them.

## Safe concrete recipe definitions

| Recipe | Gate-site set and operator | Pressure target/method |
| --- | --- | --- |
| A0 | No selected gates; stock GELU at h | None |
| A1-H | ReLU replaces GELU at h | None |
| A1-H-L1 | Same ReLU at h | Naive L1 at h |
| A1-H-OL1 | Same ReLU at h | OL1 at h |
| A4 | One-sided threshold gates at a,m,h,z | None |
| A4-OL1 | Same four one-sided gates | OL1 on a,m,h,z |
| A7 | A4's gates plus symmetric gates at post-RoPE q,k and v | None |
| A7-OL1 | Same seven gates | OL1 on all seven sites |

All active sites repeat across layers. A4 is paper shorthand for A4-Z; A7 is
paper shorthand for A7-Z-POST. Pressure applies to the post-gate tensor, with
equal means across targeted site/layer tensors. These definitions must not be
confused with historical Run 012, which realized A4 plus OL1 only at h. The
corrected four-site A4-OL1 evidence is Run 015.

The ladder enumerates separately pretrained alternatives. It is not a sequence
of continued-training stages applied to one checkpoint. Some rows add a gate,
some add pressure, and some change the pressure method or remove pressure.
“Progressively broader interventions” is an organizational description, not a
literal monotonic accumulation of components down every adjacent row.

## Verified intervention settings available for setup writing

The 14M cohort represented by the eight main ladder rows has:

- A0 and A1-H controls from Run 004;
- four A1-H naive-L1 weights, lambda in {0.05, 0.1, 0.5, 1.0}, from Run 004;
- the same four A1-H OL1 weights, with trust budget one, from Run 009;
- A4 and A7 threshold grids, kappa in {0, 0.01, 0.05, 0.1, 0.5}, from Runs
  011 and 013;
- corrected A4-OL1 and A7-OL1 at the same threshold grid with lambda=1 and
  trust budget=1, from Runs 015 and 014.

Analysis 013 maps 30 main 14M endpoints from these runs, with the historical
h-only A4 pressure conditions separately sourced from Analysis 009. Whether
all those endpoints will appear in the eventual paper remains an editorial
cohort decision; no new result table is required by this split task.

Runs 018 and 019 extend only A0, A1-H, A4-OL1, and A7-OL1 to 70M and 410M.
Each uses the five thresholds for the two pressured recipes, with lambda=1
and trust budget=1. They do not constitute a full replication of every ladder
row at every size. The ladder's ceiling columns are analytic and do not imply
otherwise.

## Matching and contrast interpretation

Same-size paired contrasts share initialization, realized data order, and
training budget, as recorded in the comparison identities and Analysis 013.
Parameter initializations cannot literally be identical across different model
shapes. Seeds are one per size (model and data-order seed 1234 in these
configs). Distinct threshold or lambda levels are treatments, not independent
replicates.

| Contrast | Defensible interpretation | Limit to preserve |
| --- | --- | --- |
| A0 versus A1-H | Replace FFN GELU with ReLU under the matched recipe | Not an isolated pressure intervention |
| A1-H versus its L1/OL1 variants | Add a stated pressure method and weight at the same h gate | Naive L1 and OL1 change optimization differently; same lambda is not equal realized correction strength |
| A1-H-L1 versus A1-H-OL1 | Compare pressure handling under the same pressure target and selected weight | Do not equate a trust cap with tuning-free or maximal loss-preserving pressure |
| A4 versus A4-OL1, at matched kappa | Add OL1 at four fixed gated targets | Effect is conditional on the given gate/target configuration |
| A7 versus A7-OL1, at matched kappa | Add OL1 at seven fixed gated targets | Its pressure objective differs from A4's |
| A4 versus A7 without pressure | Extend the gate set to internal attention operands | At kappa=0 the added symmetric gates are identities; at positive kappa they change the actual forward map |
| A4-OL1 versus A7-OL1 | Compare full recipes with separately stated gate and pressure sets | The missing A7+OL1@4 condition prevents fixed-objective gate-by-pressure attribution |

Two subtleties should remain true even if they stay in the appendix:

1. At kappa=0, A7's added symmetric gates are identity in both values and
   gradients. The unpressured A4 and A7 mathematical forward maps coincide,
   even though their selected-site analytic ceilings differ. Their pressured
   versions still differ because pressure targets and equal-tensor
   normalization change. Thus it is too blunt to imply that added gate sites
   always change the executed activation map.
2. A1-H's ReLU and A4's one-sided kappa=0 gate agree in h forward values but
   differ in their derivative at zero. A1-H to A4 is not perfectly described
   as changing only the number of sites.

These are interpretation constraints, not proposals for new experiments.

## Cross-size scope

Analysis 011 reports one MiniPile pass per size, 712 optimizer boundaries,
2,097,152 input tokens per boundary, and 1,493,172,224 processed input tokens.
The audited configs use 1,024 sequences of length 2,048 per global batch.
This supports “fixed token exposure” or “equal-token size extensions.” It
does not support equal exposure per parameter, convergence-matched training,
cross-family transfer, or a scaling law.

An unqualified “same training protocol across sizes” would be inaccurate:
14M and 70M peak at learning rate 1e-3 and end at 1e-4, whereas 410M peaks at
3e-4 and ends at 3e-5. Microbatch decomposition and initialization realization
also differ. These differences need not clutter the compact main, but any
claim of matching must state its actual dimensions.

The completed full-pass cohorts use FP16 dynamic-loss-scaling training with
FP32 parameters, not the generic BF16 reference recipe in METHODS. If precision
is introduced in Experimental Study, use actual cohort configs; do not copy
the shared-method default. Optimizer details also follow the run-specific
mapped recipe, including bias/LayerNorm weight-decay exclusions, rather than
assuming the simplest shared optimizer builder was the full execution path.

## Common evaluation facts safe to move to setup

- Randomly initialized pinned Pythia architectures; released weights are not
  used. The family name identifies architecture, not continuation pretraining.
- MiniPile validation uses all 500 documents, packed into 338 complete
  2,048-token blocks, with 692,224 input tokens and a 1,444-token excluded tail.
- Each block contributes 2,047 next-token targets; matrix-product accounting
  includes all 2,048 input positions.
- Quality and logical counts used together come from the same eager,
  uncached forward configuration, not separately collected loss and sparsity
  that happen to belong to the same checkpoint.
- The scalar multiplication fraction remains hardware-independent accounting;
  timing a sparse implementation is a separate experiment/workload.

The existing selected-size evidence also contains uniform TEAL-style post-hoc
references. If mentioned, distinguish uniform calibration/clipping from TEAL's
optimized allocation; do not claim a full TEAL reproduction. Adding a new
post-hoc design or runtime setup is unnecessary for this structural task.

## Scientific guardrails during shortening

Keep the following visible somewhere near the resulting section boundary:

- separately controllable design choices need not have additive effects;
- A4/A7 pressured comparisons change the pressure objective;
- the ladder is a map, not complete execution coverage or a training curriculum;
- the size extensions are selected, one-seed, fixed-token comparisons;
- model-wide sparsity counts only the declared multiplication families plus
  denominator-only dense head, not all compute;
- selected-site reach does not bound observed zeros from all sites;
- no achieved interaction, mechanism, transfer, or speedup is newly claimed.

## Review rubric for the new split draft

No score is assigned to this preparatory audit. The two requested actual-draft
passes will use equal 1–5 scores for operational fidelity, metric/reach
precision, contrast identifiability, setup/coverage honesty, and claim rigor.
Anchors: 1 blocking, 2 major revision, 3 competent, 4 strong, 5 exceptional;
maximum 25. Scores will assess the requested structural manuscript stage
without demanding or inventing results.
