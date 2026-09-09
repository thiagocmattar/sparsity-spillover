# Revision v2: an intervention-centered, explanatory paper
## Coding-agent task file for *Activation Sparsification in Transformers: Pressure, Thresholding, and Site Placement*

**Base manuscript:** `main(20260909-211032).pdf` (28 pages).  
**Base PDF SHA-256:** `b0686105e14298cb7b329a2f14c5aa2ed820ae56474a31474dc3bb970172b0b2`.  
**Earlier visual reference:** `main(20260909-182851).pdf`, especially original Figure 5 on p. 10.  
**Purpose:** Make a focused second revision of the existing LaTeX paper and figure-generation code. Explain the experimental choices, restore the cross-size visual argument, and turn observations into useful scientific lessons. Do not restart the entire paper or turn it into a leaderboard.  
**Default mode:** `EXISTING_EVIDENCE`. Read repository code and records, regenerate figures, perform CPU analyses and mathematical unit tests, and edit `.tex`/`.bib`. New training, checkpoint reevaluation, GPU measurements, and remote paid jobs require explicit authorization.  
**Source convention:** `[M]` denotes the base manuscript; `[O]` the earlier manuscript; `[Q]`, `[S]`, and `[P]` the primary literature identified in the source register. Page numbers are PDF pages in these source versions. Find edits by section, paragraph opening, and LaTeX label—not by changing page numbers.

---

## 0. Author decisions: implement these, do not reopen them by default

- [ ] **Restore the cross-size quality–model-wide-sparsity comparison to the main text.** It is evidence that a particular intervention relationship is observed beyond 14M. Keep the 14M/70M/410M side-by-side view and the useful distinction between absolute and normalized comparisons.
- [ ] **Remove the “Cohort means” subplot.** Preserve its numerical attribution evidence in prose and the implementation table; do not replace it with another average-speedup bar chart.
- [ ] **Remove the main-text “Maximum observed sparsity under same-size A0” table and its winner-selection paragraph.** Retain the underlying reduction in the supplement. Use the freed space for intervention comparisons and their explanation.
- [ ] **Explain why OL1 was chosen.** Present its protected-direction geometry, relative norm cap, and conditional saturation with respect to the regularization weight. Treat it as a fixed experimental instrument, not a newly validated optimizer.
- [ ] **Restore the motivation for fixed thresholding.** Connect projection-input sparsification to Q-Sparse and the selection-overhead question to Spark; then explain why this study lets learned distributions determine the surviving count under a fixed cutoff.
- [ ] **Present architectural reach as an a priori planning tool.** Explain what can be calculated before training, including width, depth, sequence length, and dense-head effects.
- [ ] **Analyze ReLU-trained checkpoints plus post-hoc clipping explicitly.** Add the A1-H clipping trajectories to the cross-size comparison, subject to verification of the existing records.
- [ ] **Investigate the 410M training-budget explanation using retained logs.** Treat the author's gradient-norm observation as a hypothesis to check, not an established result to insert.
- [ ] **Write an informative paper, not an extended disclaimer.** Each major finding should tell the reader what changes, what explains the change, and what experimental choice it informs.

These instructions supersede conflicting editorial choices in `activation_sparsification_agentic_rewrite_plan.md`, especially requirements to prioritize a quality-budget winner table or an aggregate-speedup panel. They do **not** supersede numerical provenance, fair comparisons, or honest reporting of unfavorable results.

### The desired argument

> We examine how pressure, thresholding, and placement interact. A priori operation accounting explains the opportunity associated with each placement; paired training comparisons show how the intervention changes loss and zeros; selected cross-size comparisons show which relationships recur; and kernel controls distinguish counted opportunity from profitable execution.

This is an **intervention study with an execution case study**. It is not primarily a search for the best checkpoint under a chosen quality budget, a claim of universal optimizer superiority, or a general scaling law.

### Three distinctions that the revision must make positively

1. **Design rationale versus empirical validation:** A well-motivated fixed OL1 configuration is legitimate without a full optimizer sweep. A mathematical saturation property is not evidence that every training run operated in that regime.
2. **Cross-size recurrence versus universal scaling:** A named comparison recurring at 14M, 70M, and 410M is meaningful. It does not imply that every threshold ordering or dense-relative loss penalty improves with size.
3. **Intervention comparison versus winner selection:** Show the loss–sparsity response of each intervention, including trade-offs and reversals. Avoid converting the discussion into retrospective winners at arbitrary budgets.

---

## 1. Execution order, priorities, and completion rubric

### 1.1 Work in small, restartable batches

| Batch | Task IDs | Main deliverable | Effort |
|---|---|---|---|
| 0 | B00–B02 | Source map, evidence ledger, baseline build | Low |
| 1 | O01–O03, T01–T02, A01 | Verified method rationale and analytic checks | Low; some diagnostics may require mid effort |
| 2 | R01–R03, G01–G02 | Cross-size/ReLU analyses and training-log audit | Low with retained records; reevaluation is mid |
| 3 | V01–V03 | Restored cross-size figure, revised runtime figure, table relocation | Low |
| 4 | W01–W06 | Paragraph-level setup and results edits | Low |
| 5 | W07–W10 | Related work, discussion, introduction, abstract | Low |
| 6 | Q01–Q03 | Numeric, mathematical, visual, and prose verification | Low |
| Optional | E01–E04 | Approved evidence that answers a specific unresolved question | Mid to high |

Implement evidence-bearing figures and their analyses before writing the abstract. Complete two to four related tasks per batch, inspect the diff, update the state, and build. Do not make a single sprawling unreviewable rewrite commit.

### 1.2 Rubric

Score each dimension from 0 to 3: **0** missing/wrong; **1** partial; **2** adequate and verifiable; **3** especially clear and well supported. Do not average away a factual or mathematical failure.

| Dimension | Required for 2 or higher |
|---|---|
| Design justification | OL1, thresholds, and reach each have a stated purpose, an exact definition, and a supported scope. |
| Intervention insight | Results explain named comparisons, not just maximum values or winners. |
| Cross-size evidence | The main figure shows all three sizes, the recurring relationship, and the deviations. |
| Scientific precision | Adaptive-direction protection is not actual-loss protection; clipping timing is correct; gradient norms are not convergence certificates. |
| Runtime attribution | Removing the means plot does not remove the fusion and sparse-path controls. |
| Human readability | Stable terminology, explicit subjects/comparators, short explanations before qualifications. |
| Traceability | Every new number, graph, and empirical claim maps to a verified source or reproducible reduction. |

**Highest return at low effort:** Restore the cross-size figure; justify the methods; replace winner-focused prose; extract the ReLU/clipping response; correct the scale narrative.  
**Highest return at mid effort:** Missing checkpoint evaluations, targeted repeatability, and a diagnostic test selected after inspecting OL1 saturation and training logs.  
**Highest return at high effort:** A matched longer-training continuation only when it directly tests the remaining 410M budget hypothesis. Do not automatically run another full large-model sweep.

A completed rewrite can materially improve the submission. It cannot establish repeatability or convergence from prose alone. The handoff must distinguish editorial completion from evidence still needed for the strongest claims.

---

## 2. Writing contract

### 2.1 Language and readability

- Use consistent, professional English, preferably the manuscript's existing American-English conventions. Correct `GeLU` to **GELU**, `atention` to **attention**, and `accross` to **across**. Preserve proper method names.
- Prefer a concrete subject and verb: “Adding pressure increases…” rather than “An increase can be observed…”.
- Explain the experimental choice before listing its limitations. “We use a fixed relative norm budget to…” is more informative than an isolated “This does not guarantee…”.
- Keep one principal claim per sentence. Aim for roughly 3–6 sentences per results paragraph, but do not enforce a mechanical word-count rule.
- Define technical shorthand on first use. Use “speedup over standard PyTorch execution” before “native-relative speedup”; “same fused implementation with skipping disabled” before “all-skips-off”.
- Keep legitimate technical terms. Do not replace “orthogonal,” “preconditioned,” “causal,” or “geometric mean” with vague language. Explain the coordinate system or denominator when it matters.
- Do not invent jargon: avoid “sparsity leverage,” “quality elasticity,” “frontier robustness,” and similar phrases unless a quantity is formally defined and genuinely necessary.
- Do not use “scales better,” “improves,” “preserves the pattern,” or “does not hurt” without naming the axis, comparator, and setting.
- Do not repeatedly apologize for the same experimental boundary. State it at the first relevant comparison and in the limitations; repeat locally only when a reader could otherwise misunderstand the figure.
- Do not call a numerical diagnostic, extra-seed run, or checkpoint reevaluation “just a rewrite.” Label new evidence accurately.

### 2.2 The informative paragraph

A useful paragraph usually follows:

> **Question or design choice → named comparison → quantitative observation → supported explanation → implication.**

Add the necessary boundary beside the claim it limits. Do not end every paragraph with a generic disclaimer. A subsection should leave the reader with a useful consequence, for example:

- Choosing a fixed threshold lets the active count respond to training; it does not impose a constant sparse compute budget.
- Adding pressure targets changes the direction of the pressure correction, even when a norm cap largely removes its overall scale.
- A larger fraction of model-wide zero products can arise from a changing operation denominator, not only from more zero activations.
- A training nonlinearity changes the representation to which post-hoc clipping is later applied.
- A relationship that recurs across model sizes can coexist with a failure of absolute quality to improve uniformly.

Do not add a speculative mechanism merely to make a paragraph sound insightful. Mark a hypothesis as a hypothesis and name the test that would distinguish it.

### 2.3 Required terminology and symbols

| Use | Meaning / restriction |
|---|---|
| **pressure** | Training-time activation regularization, not the thresholding map itself. |
| **naive L1** / **OL1** | Existing update-rule labels. Expand OL1 as orthogonal L1; explain that projection is conflict-conditioned. |
| `\mathcal L_{\mathrm{task}}`, `\mathcal L_1`, `\lambda` | Task objective, activation-pressure objective, pressure weight. |
| `b` | Relative pressure-update norm budget, fixed to 1 in the reported OL1 conditions. Not the task-gradient clipping threshold. |
| `u`, `w`, `\widetilde w` | Adaptive task vector, preconditioned pressure vector, and projected pressure vector, as in Appendix A.2. |
| `G_{+,\kappa}(x)=x\mathbf1\{x\ge\kappa\}` | One-sided, value-preserving thresholding. Not `ReLU(x-\kappa)`. |
| `G_{\pm,\kappa}(x)=x\mathbf1\{|x|\ge\kappa\}` | Symmetric magnitude thresholding. |
| `\mathcal N`, `\mathcal P` | Thresholding sites and pressure sites, respectively. They need not coincide. |
| `p` | Post-hoc calibration quantile target. Not achieved model-wide sparsity or fraction of surviving entries removed. |
| **post-hoc magnitude thresholding (clipping)** | Evaluation-only zeroing after training. “Clipping” here is zeroing small values, not saturating large values. |
| **gradient clipping** | Norm limiting before the optimizer, distinct from activation clipping and the OL1 correction cap. |
| `S_{\mathrm{model}}` | Counted zero-operand scalar multiplications / declared full-model multiplication workload. |
| `R_{\mathrm{arch}}(\mathcal N)` | A priori architectural reach for the declared sites, graph, and workload. |
| `S_{\mathrm{block}}` | Zero-product fraction using the block-only denominator; do not call it ceiling utilization. |
| `\Delta\mathcal L` | Always specify treatment minus reference. Use same-size A0 for dense-relative cross-size comparisons. |
| **percentage points (pp)** | Differences between sparsity percentages. Do not confuse with relative percent changes. |
| **trade-off curve** | Connections between evaluated settings. “Observed frontier” requires actual nondominance calculation; it is not a fitted curve. |
| **complete-recipe comparison** | A comparison that changes more than one intervention component. Explain once rather than using it as a repeated warning label. |
| **14M, 70M, 410M** | Nominal Pythia-family labels; architecture dimensions and actual parameter counts remain documented. |

Preserve raw data keys and run IDs. For example, `R_model_max` may remain an archival key mapped to manuscript `R_{\mathrm{arch}}`; do not rename historical datasets merely to match prose.

---

## 3. Baseline, repository discovery, and evidence controls

### B00 — Establish the editable baseline

**Priority:** P0. **Effort:** Low. **Sources:** [M], [O].

- [ ] Read the draft README and the existing figure/table regeneration README, including the manuscript's referenced Analysis 019 instructions if present.
- [ ] Locate the root `.tex`, included section files, bibliography, figure source scripts, generated assets, supplementary records, training code/configs, and logs.
- [ ] Record `git status` and the current commit. Preserve unrelated user edits. Do not reset, clean, or overwrite the repository.
- [ ] Build the current paper with the repository's documented command. Save the baseline PDF, build log, and relevant rendered pages outside tracked source directories unless repository conventions say otherwise.
- [ ] Record actual paths in `revision-v2/source-map.md`. The paths in this task file are intended outputs or search anchors, not claims that files already exist.

Suggested searches, adapting to available tools:

```bash
rg -n -g '*.tex' 'SPARSIFICATION|Sparsification|architectural reach|Architectural reach' .
rg -n -g '*.tex' 'Maximum observed sparsity|Cohort means|Transfer across model sizes' .
rg -n -g '*.tex' 'This does not guarantee|pressure.*normalization|Retrospective quality' .
rg -n 'all.skips.off|statistical.top|trust.budget|grad.norm|gradient.norm' \
  --glob '*.py' --glob '*.json' --glob '*.yaml' --glob '*.yml' .
```

**Done when:** The agent can identify the source block and generating script for every edited figure, table, and paragraph. Missing logs do not block unrelated text and figure edits.

### B01 — Create a restartable task and claim ledger

- [ ] Create `revision-v2/state.md` with task statuses: `TODO`, `IN_PROGRESS`, `DONE_VERIFIED`, `BLOCKED_MISSING_DATA`, `NEEDS_AUTHORIZATION`, `NOT_APPLICABLE`.
- [ ] Create `revision-v2/claims.csv` with fields: `claim_id`, `claim_text`, `type`, `source_path`, `run_ids`, `precision`, `comparison`, `manuscript_location`, `verification`, `allowed_wording`.
- [ ] Use types `OBSERVED`, `DERIVED`, `DESIGN_RATIONALE`, and `HYPOTHESIS`.
- [ ] Record the author's OL1 robustness and 410M gradient observations as rationale/hypothesis until the required checks establish more.
- [ ] Maintain `revision-v2/changes.md`: task ID, paragraph anchor, files changed, numerical changes, build/test result, and remaining dependency.

**Done when:** A fresh agent can resume without rereading the entire conversation or inferring that pending experiments already happened.

### B02 — Freeze the scientific invariants

- [ ] Preserve the realized 54-condition primary study and its identities unless new conditions are explicitly added as a separate analysis.
- [ ] Preserve one-seed reporting. Neither 29 paired contrasts nor three model sizes are independent seed replications.
- [ ] Preserve all evaluated thresholds, clipping targets, unfavorable points, and full-range supplementary figures.
- [ ] Do not alter training code, checkpoint semantics, objective normalization, or threshold derivatives to make the method match a preferred explanation.
- [ ] Establish one source of truth for numerical table/figure data. Generate new summaries from records, not from rounded PDF entries.
- [ ] Require an explicit evidence update before changing cohort sizes, evaluation counts, or the statistical status of a finding.

**Done when:** Editorial changes cannot silently change what experiment is being described.

---

## 4. Method rationale: precise positive arguments

### O01 — Audit the implemented OL1 rule before describing its geometry

**Priority:** P0. **Effort:** Low. **Locate:** §3.1 paragraph beginning “Pressure uses the mean absolute activation”; Appendix A.2, p. 11. **Source:** [M, Appendix A.2].

Verify against the training implementation and run configs:

- [ ] The task gradient is accumulated, unscaled if AMP is used, globally clipped, and supplied to AdamW separately from the pressure gradient.
- [ ] `u` uses the updated, bias-corrected task moments; `w` uses the pressure gradient and the task second-moment preconditioner.
- [ ] Projection is applied only when `u · w < 0`, subject to the actual small-norm safeguard. It is not unconditional orthogonalization.
- [ ] The correction norm is capped relative to `||u||`, over the documented eligible parameters, before group learning rates.
- [ ] Record the actual treatment of decoupled weight decay, parameter groups with different learning rates, absent gradients, zero vectors, numerical stabilizers, and skipped optimizer steps.
- [ ] Verify that all reported OL1 runs use `b=1`, while the local study sweeps `lambda` and the multisite study fixes `lambda=1`.

**Deliverable:** `revision-v2/ol1-audit.md`, with implementation locations and the implemented formula. Do not call `lambda=1` “large enough to saturate” solely because it is the largest tested value.

**Done when:** The main-text rationale and appendix describe the rule that produced the checkpoints, not an improved rule the agent wishes had been used.

### O02 — Replace the unsupported-disclaimer paragraph with the protected-direction argument

**Priority:** P0. **Effort:** Low. **Dependencies:** O01. **Source:** [M, Appendix A.2]; the derivation below is new mathematical reasoning, not a new experiment.

#### Exact editorial target

Replace the main-text sentence beginning **“This does not guarantee preserved task loss”** as the paragraph's explanatory conclusion. Do not simply delete the boundary and substitute an unconditional claim that task loss is preserved.

Use two short paragraphs after the pressure-objective definition:

**Paragraph O-P1: why this pressure implementation.**

- Task optimization remains the primary update.
- The pressure correction uses the task optimizer's coordinate scaling.
- Under conflict, remove the component opposing the adaptive task direction.
- Bound the correction using that direction's norm instead of letting a large pressure weight produce an arbitrarily large correction.

**Paragraph O-P2: why a fixed configuration.**

- In the cap-active regime, increasing `lambda` no longer materially increases the correction magnitude.
- This motivates holding `b` and the multisite `lambda` fixed to keep the study focused on thresholds and placement.
- Describe this as a design rationale and a conditional geometric property, not measured hyperparameter robustness unless O03 provides evidence.
- Say the budget extends the **norm-limiting principle** of the Pythia training recipe to an auxiliary update. Do not say update-norm capping is mathematically identical to Pythia's gradient clipping.

#### Analytic explanation to add to Appendix A.2

Keep the exact stabilized implementation first. Then introduce an explicitly **idealized** version, ignoring numerical stabilizers and using a common learning rate for the displacement interpretation.

For a nonzero adaptive task direction `u`, define

\[
\widetilde w =
\begin{cases}
w-\frac{u^\top w}{\|u\|^2}u, & u^\top w<0,\\
w, & u^\top w\ge0.
\end{cases}
\]

The ideal pressure descent vector is

\[
d_{\mathrm p}
=\min\!\left(\lambda,\frac{b\|u\|}{\|\widetilde w\|}\right)\widetilde w,
\]

with `d_p=0` when `w_tilde=0`. The added parameter displacement is `-alpha d_p`.

This gives three useful properties:

1. **Adaptive-direction protection:** `u^T d_p >= 0`, and equality holds in the nondegenerate conflicting case after projection. Thus `u^T(u+d_p) >= ||u||^2`. The added direction does not cancel progress measured along `u`.
2. **Relative norm control:** `||d_p|| <= b ||u||` in the idealized rule. At `b=1`, the auxiliary vector cannot exceed the adaptive task vector in this norm. The combined vector is not itself constrained to the task-vector norm.
3. **Saturation in lambda:** once `lambda ||w_tilde|| >= b ||u||`, `d_p = b ||u|| w_tilde / ||w_tilde||`. The ideal vector is then independent of further increases in `lambda` at that fixed optimizer state and gradient pair.

Optional, if a compact formal statement improves the appendix: with `rho=b||u||` and nonzero projected direction, the cap-saturated ideal vector solves

\[
\underset{d}{\operatorname{maximize}}\;w^\top d
\quad\text{subject to}\quad
u^\top d\ge0,\quad\|d\|\le\rho,
\]

The constraint defines the adaptive-task half-space. This is maximum alignment with the **preconditioned pressure direction** inside the norm ball and protected half-space. In the conflicting case, the optimum lies in the orthogonal subspace. In the nonconflicting case, retaining an aligned component is appropriate. In the exactly opposing degenerate case, the implemented zero correction is an optimum but need not be the only one.

Do not call that objective the true first-order pressure-loss decrease: it is `w^T d`, not generally `g_1^T d`.

#### The necessary boundary, stated constructively and once

An AdamW adaptive direction is not generally the current task gradient. The protected quantity above is its directional component, not the actual task loss. Actual first-order task loss depends on `g_task^T d_p`; finite-step behavior also depends on curvature. The implemented stabilizers make conflict removal approximate, and group learning rates/weight decay require the scope stated in O01.

Use wording such as:

> “The projection protects the adaptive task direction, and the relative norm cap makes the auxiliary step saturate with its weight. We therefore use a fixed budget as a controlled pressure mechanism rather than tuning an optimizer for each intervention. Appendix A.2 states the geometric property and its relation to the implemented update.”

Place the exact distinction from task-loss preservation in the appendix explanation, not as the only reason offered in the main text.

#### CPU verification tasks

- [ ] Test ideal projection on aligned, orthogonal, conflicting, exactly opposing, and zero pressure vectors.
- [ ] Test the ideal cap and the lambda-saturation boundary.
- [ ] Compare the stabilized implementation with the ideal rule and describe differences rather than imposing exact equality tests on an approximate projection.
- [ ] Include an internal negative test preventing a false task-loss claim: `u=(1,1)`, `w=(-2,0)`, `b=1`, cap active gives `d_p=(-1,1)`. For task gradient `g=(1,0)`, `u^T d_p=0`, but `g^T(u+d_p)=0` versus `g^T u=1`. Orthogonality to `u` can remove the Adam direction's predicted loss improvement. This is a verification example, not a necessary manuscript figure.
- [ ] Remove any statement that `b=1` makes the full update identical to a gradient-clipped Pythia step. The norm domains and sequence of operations differ.

**Done when:** The authors' intended rationale is explained positively, with a real, verified geometric result and no false loss-preservation theorem.

### O03 — Check saturation before asserting robustness or a normalization confound

**Priority:** P0 for analysis; empirical extension conditional. **Effort:** Low with logs; mid for a short approved diagnostic. **Dependencies:** O01–O02.

This task also corrects an over-simple interpretation in the previous revision advice: changing the mean's denominator changes the objective algebraically, but it need not change a saturated OL1 correction's magnitude.

- [ ] Search retained per-step records for `||u||`, `||w_tilde||`, `lambda`, cap multiplier, conflict indicator, and update norms.
- [ ] Compute the fraction of steps on which the cap binds and the distribution of `||d_p||/||u||`, separately by recipe, threshold, and early/late training phase.
- [ ] Define “cap active” from the actual implementation condition, not from a rounded correction norm alone.
- [ ] Report whether stored norms precede or follow learning-rate multiplication and which parameter set they cover.
- [ ] When records are absent, retain the conditional analytic rationale. Mark actual saturation as unknown; do not infer it from endpoint sparsity or the selected `lambda`.

#### Positive-scaling invariance check

At fixed `u`, multiplying the entire pressure objective by a positive scalar multiplies `w` and `w_tilde` by that scalar. If both corrections are cap-saturated, their normalized directions are identical in the ideal rule. The implemented epsilon terms make the equivalence approximate.

Consequently:

- A four-target mean extended to seven targets changes the mixture of site gradients. That directional change remains scientifically relevant even under saturation.
- For a **fixed expanded target set**, using `sum_7/(7L)` versus `sum_7/(4L)` differs only by an overall factor. A saturated normalization step can remove that factor's effect.
- Do not claim that the existing sites necessarily received a proportionally smaller **effective update** just because their coefficients in the written mean became smaller.
- Do not order a long “fixed coefficient” ablation that merely rescales the same always-saturated vector. First establish that it changes effective updates or a meaningful relative weighting.
- Independent weighting of old versus newly added sites can change direction; that is a different experiment from changing one global denominator.

**Approved wording without saturation logs:** “Expanding the pressure targets changes the aggregate direction and, when the cap is inactive, its scale. The comparison therefore evaluates a different pressure objective.”

**Approved stronger wording with logs:** Quantify the observed cap-active fraction and explain how often global rescaling is suppressed. Do not extrapolate to untested weights or training settings.

**Done when:** The paper neither asserts unmeasured hyperparameter robustness nor overstates normalization dilution as an established mechanism.

### T01 — Restore the Q-Sparse → Spark → fixed-threshold rationale

**Priority:** P0. **Effort:** Low. **Locate:** §3.1 opening paragraph; §2.1 paragraph beginning “Beyond FFNs”; original §3.1 first paragraph. **Sources:** [Q], [S], [O, §3.1].

Write a short motivation paragraph **before** the threshold equations:

- [ ] Describe this study as testing the broader idea of train-time activation sparsification motivated by projection-input selection, not as reproducing Q-Sparse.
- [ ] Cite [Q] for top-k magnitude selection and its straight-through training rule. The present rule changes both selection and gradient treatment; do not imply identical methods.
- [ ] Cite [S, §3] for the implementation-overhead motivation and statistical top-k alternative. Describe fixed thresholds as avoiding per-input selection and threshold-estimation work, not as a measured speedup over Spark.
- [ ] State the experimental hypothesis: with the cutoff held fixed, training and optional pressure reshape site distributions, so the surviving count is an outcome rather than a constraint.
- [ ] Connect this choice to the distribution study: the histograms investigate the representation learned with that intervention; they do not establish an entire time trajectory from initialization.

Suggested final logic, to adapt rather than copy mechanically:

> “We study a fixed-cutoff alternative to activation selection. The cutoff requires neither ranking nor a per-input estimate, and leaves the surviving count free to change as training reshapes the activation distribution. This makes the threshold and pressure separate experimental controls: the threshold defines what is zeroed, while pressure can change how much activation mass reaches it.”

**Literature precision:** Avoid saying all top-k algorithms require a full sort. Attribute the costly selection/sorting motivation to the implementation context in [S]. Spark's statistical operator, its attention-position selection, and this paper's feature-coordinate thresholding are not interchangeable mechanisms.

**Done when:** The reader understands why this simpler intervention is scientifically useful without being told it has already beaten selection-based methods.

### T02 — Keep the actual maps, site assignments, and derivatives exact

**Priority:** P0. **Effort:** Low. **Locate:** threshold equations; Figure 1 caption; Appendix A.1 and Table 4. **Source:** [M].

- [ ] Preserve the factor `x` in both maps: `G(x)=x * indicator`. A binary indicator alone is not the executed activation.
- [ ] Explain that retained values are unchanged. `G_{+,kappa}` is not `ReLU(x-kappa)`; at a positive cutoff the map is discontinuous at the boundary.
- [ ] Preserve the executed site map: **one-sided** at `a,m,h,z`; **symmetric** at `q,k,v` in A7.
- [ ] Do not simplify this to “one-sided at FFN sites and symmetric at all attention sites.” The attention projection sites `a,z` also use the one-sided map in the reported recipes.
- [ ] Describe `h` as replacing GELU; describe extra sites as adding the declared transformation. Q/K thresholding is after RoPE.
- [ ] Retain the detached mask convention and boundary derivatives. `G_{+,0}` and ReLU share forward values but differ at zero in the specified implementation; `G_{±,0}` is identity in values and gradients.
- [ ] Add a small method test or verify an existing test at `-kappa`, `0`, `kappa`, and just inside/outside boundaries.
- [ ] Keep any new all-attention symmetric-site ablation in the optional experiment list. Do not alter old recipes through prose.

**Done when:** The appealing rationale accurately describes what was trained.

### A01 — Make architectural reach useful before training

**Priority:** P0. **Effort:** Low. **Locate:** §3.2 paragraph beginning “Architectural reach”; Appendix B.2 and C.2. **Source:** [M, pp. 3, 12, 14]; algebra below is derived from its declared counts.

Replace the definition-only paragraph with **purpose → definition → analytic consequence**:

1. **Purpose:** Before training, estimate which fraction of the declared multiplication workload a proposed set of sites can structurally affect. This helps distinguish a narrow intervention from one that reaches substantially different operations.
2. **Definition:** Preserve `R_arch=N_reach(N)/N_model`, actual graph closure, union counting, and the dense-head denominator.
3. **Consequence:** Explain how changing width/depth/sequence length changes reachable work. Reach is architecture/workload potential, not a quality-preserving sparsity prediction or a measured speedup.

#### Analytic specialization to verify and retain in the appendix

For the manuscript's full-sequence causal Pythia graph, with `d_f=4d`,

\[
C_{\mathrm{model}}
=Td\,[L(12d+T+1)+V_{\mathrm{vocab}}].
\]

Let `D=L(12d+T+1)+V_vocab`. Then

\[
R_{\mathrm{arch}}(\{h\})=\frac{4Ld}{D},\qquad
R_{\mathrm{arch}}(\{m,h\})=\frac{8Ld}{D},
\]
\[
R_{\mathrm{arch}}(A4)=\frac{12Ld}{D},\qquad
R_{\mathrm{arch}}(A7)=\frac{L(12d+T+1)}{D}.
\]

The additional operation coverage from A4 to A7 is

\[
R_{\mathrm{arch}}(A7)-R_{\mathrm{arch}}(A4)
=\frac{L(T+1)}{D},
\]

which is the share of the two attention matrix products for this graph; A4 already reaches the attention-output projection.

Explain the distinctions:

- At fixed depth, vocabulary, and sequence length, FFN/projection work grows quadratically in width, while the two attention products grow linearly in width.
- At fixed architecture, projection work grows linearly with sequence length, while causal attention-product work grows quadratically. Thus long sequences increase the relative importance of internal-attention sparsification.
- More depth can reduce the dense head's relative share. “Model size” is not a single geometric variable; width, depth, and vocabulary affect the ratio differently.
- Attention **projections** also have width-quadratic work. Do not describe all attention operations as quadratic in sequence length.
- The statements concern full uncached sequences. Do not silently apply the same formula to cached decoding.

Use the existing architecture configurations to verify these approximate percentages at `T=2048`, `V_vocab=50304`:

| Size | h-only reach | A4 reach | A7 reach |
|---|---:|---:|---:|
| 14M | 4.278% | 12.833% | 29.952% |
| 70M | 12.354% | 37.063% | 49.424% |
| 410M | 24.925% | 74.776% | 87.245% |

The h-only ceiling rises substantially across these configurations even before observing any learned sparsity. This is a useful explanation for part of the cross-size change, not evidence that training itself improves monotonically.

- [ ] Add CPU tests against the original operation inventory, overlap counting, and `V=0 => PV=0` closure.
- [ ] Check width/sequence-length trends by evaluating the formulas; no training is needed.
- [ ] Keep `S_block=S_model/R_arch(A7)` only with the explanation that A7 reaches every counted block operation and its denominator is the same for all curves within a size.
- [ ] State once that natural zeros outside selected-site reach can contribute to `S_model`; do not reintroduce “universal sparsity ceiling.”

**Done when:** A reader can use the metric to reason about a proposed intervention before committing training compute.


---

## 5. Existing-data analyses that drive the rewritten results

### R01 — Assemble a precision-consistent cross-size comparison dataset

**Priority:** P0. **Effort:** Low. **Locate:** endpoint records; 540 clipping records; current Figure 8 and Figures 10–12 scripts. **Source:** [M, Tables 6 and 8; Appendix D.5].

- [ ] Join records by checkpoint content hash/run identity, size, recipe, trained `kappa` or `lambda`, clipping target `p`, and evaluation precision. Do not join by a display label alone.
- [ ] Retain each measurement's source and whether it is a training endpoint or an evaluation-only clipping result.
- [ ] Build `revision-v2/data/cross_size_interventions.csv` from all verified records relevant to A0, A1-H, A4-OL1, and A7-OL1 at all three sizes.
- [ ] Include absolute loss, same-size A0 loss difference, `S_model`, `S_block`, operation numerators/denominators if available, and architectural reach for the actual inference sites.
- [ ] For clipping trajectories, use their measured `p=0` values; do not silently substitute canonical endpoints when tiny evaluation differences exist.
- [ ] Keep a discrepancy report for endpoint versus `p=0`, including precision and implementation differences. Do not treat numerical drift as a substantive intervention gain.

**Done when:** The same data file can generate the cross-size figure, the ReLU analysis, and all quoted numerical comparisons.

### R02 — Test what ReLU changes about post-hoc clipping

**Priority:** P0. **Effort:** Low with the complete clipping records; missing reevaluations are mid. **Dependencies:** R01. **Source:** [M, Tables 6 and 1; Figures 10–12].

The author's proposed result is worth testing and showing. Its accurate description is **“ReLU pretraining followed by post-hoc clipping”**, not “train-time clipping,” unless a separate run actually clips activations during training.

#### Required comparisons

For each size, compare **A0 + clipping** with **A1-H + clipping**, using the same clipping sites and calibration procedure.

- [ ] Show the absolute loss versus achieved `S_model` trajectories. Include unmodified A0 and A1-H as visible reference points.
- [ ] Report the base cost of changing GELU to ReLU, before any extra clipping. Current endpoint deltas versus same-size A0 are about `+0.0611`, `+0.1230`, and `+0.1038` at 14M, 70M, and 410M, respectively. Recompute from unrounded records.
- [ ] Also compute within-checkpoint clipping increments:

\[
\delta\mathcal L_r(p)=\mathcal L_r(p)-\mathcal L_r(0),\qquad
\delta S_r(p)=S_r(p)-S_r(0),
\]

where `r` is A0 or A1-H. This separates the starting representation from sensitivity to the additional evaluation intervention.

- [ ] Compare matched `p` as a **matched calibration policy**, not as matched achieved sparsity.
- [ ] Find where the observed curves cross or change ordering. Summarize the region where ReLU-trained checkpoints have a favorable observed trade-off and the region where the GELU baseline retains lower loss.
- [ ] For a claim about comparable achieved sparsity, use verified observed points and report their sparsity mismatch. Where records do not give a close comparison, label the gap or request a small new evaluation. Do not invent equal-sparsity points through smooth interpolation.
- [ ] Report all sizes separately before describing any cross-size trend. Do not infer a monotonic tolerance improvement from an increasing raw sparsity percentage alone.

#### Interpretive checks

- Existing zero mass can make several quantile targets produce the same mask. A flat response over those targets does not by itself prove greater tolerance to removing additional nonzero activations.
- A1-H starts with exact zeros at `h`; adding four-site clipping changes its inference reach. **A1-H + four-site clipping has the union of those four sites, not h-only reach.** Use the A4/clipping reach guide for that trajectory.
- At the same achieved total `S_model`, local zero allocations can still differ. State which comparison is being made rather than claiming site-matched behavior that was not measured.
- Any increasing cross-size benefit must be distinguished from the dense head's shrinking contribution and the different block-operation weights.

#### Outcome-dependent wording

| Data outcome | Appropriate manuscript statement |
|---|---|
| A1-H + clipping gives lower loss at comparable achieved sparsity in a verified region at all sizes | State that region and the actual comparisons; describe a recurring benefit of the ReLU-trained representation for this clipping policy. |
| ReLU reduces incremental clipping damage, but total loss remains worse | Distinguish robustness to the added clipping from final quality; retain the base ReLU cost. |
| The favorable region expands in raw sparsity but not after block normalization | Explain the architectural contribution; do not call this improved intrinsic tolerance. |
| Curves cross or the pattern differs by size | Make the crossing or reversal the finding, not a failed story to suppress. |
| Records cannot resolve comparable sparsity | Show the trajectories and bound the claim; request E01 only if needed. |

**Deliverable:** `revision-v2/relu_clipping_findings.md`, containing exact run/target pairs, base costs, incremental responses, and proposed wording. Do not produce a new cross-recipe winner table.

**Done when:** The figure can support a specific statement about ReLU and subsequent clipping, rather than a visual impression of “scaling better.”

### R03 — Correct and sharpen the cross-size story

**Priority:** P0. **Effort:** Low. **Dependencies:** R01–R02. **Locate:** §4.4, Table 3, restored cross-size figure. **Source:** [M, Tables 6 and 8].

Separate **three questions** in the analysis and prose:

1. Does a specified **within-size intervention contrast** recur?
2. Does the **absolute loss** of a matched recipe improve with more parameters?
3. Does its **penalty relative to same-size A0** shrink?

They need not have the same answer.

#### Facts already visible in the base manuscript

Verify from source records, then use as safeguards against an overbroad story:

| Comparison | 14M | 70M | 410M | Correct interpretation |
|---|---:|---:|---:|---|
| A0 absolute loss | 5.2086 | 4.0998 | 4.5475 | Dense quality improves at 70M, then worsens at 410M in this protocol. |
| A7-OL1, `kappa=0.5`, absolute loss | 5.8294 | 5.2159 | 5.1207 | This high-threshold recipe's absolute loss improves at both size changes. |
| A7-OL1, `kappa=0.5`, `S_model` | 27.483% | 40.602% | 80.616% | Raw model-wide sparsity increases; accounting changes must also be shown. |
| Same recipe's loss penalty versus A0 | +0.6208 | +1.1162 | +0.5732 | The dense-relative penalty is not monotonic. |
| A4-OL1, `kappa=0.5`, absolute loss | 6.0380 | 5.3895 | 5.1910 | This recipe also improves in absolute loss from 70M to 410M. |

The claim “410M is worse than 70M” is therefore valid for A0 and several settings, **not every sparse recipe**. The 410M high-threshold behavior is part of the result, not an inconvenient exception to discard.

- [ ] State the recurring high-threshold A7-OL1 versus A4-OL1 contrast explicitly: lower loss and higher `S_model` at every tested size.
- [ ] Retain the zero/smaller-threshold order changes. A4-OL1 is better at zero threshold for 14M/70M, whereas A7-OL1 is better at 410M.
- [ ] Explain why 14M → 70M is a cleaner architecture/protocol extension: the recorded peak learning rates agree, whereas 410M also changes that setting. Do not call the transition fully controlled in every respect; tokens per parameter and shapes still differ.
- [ ] Do not describe better absolute quality in a larger sparse model as a reduced sparsification penalty unless same-size A0 differences show it.
- [ ] Do not infer greater compute efficiency merely because a larger model has lower loss and a higher sparsity percentage. The absolute multiplication workload and latency differ by size.
- [ ] Use “the high-threshold recipe relationship recurs beyond 14M” or equivalent positive wording. Do not reduce the transfer section to caveats.

**Done when:** The reader learns both what is reproducible across the three architecture sizes and what is not monotonic under the fixed budget.

### G01 — Audit the author's gradient-norm evidence

**Priority:** P1. **Effort:** Low with logs; otherwise blocked or mid for approved diagnostics. **Locate:** training logs and their generating code; new appendix training-dynamics subsection. **Source status:** The current PDF does not establish the proposed gradient-norm comparison; it must be verified in the repository.

- [ ] Find the exact logged gradient-norm field and its implementation.
- [ ] Determine whether it is task-only or combined-objective, before or after gradient clipping, before or after AMP unscaling, and per microbatch or accumulated optimizer step.
- [ ] Determine whether it covers all parameters, an eligible OL1 subset, or distributed shards; identify the reduction convention.
- [ ] Check whether logged norms after clipping are capped at 1. Such a curve cannot show the size of pre-clipping gradients beyond the cap.
- [ ] Align by actual input tokens and, where relevant, successful optimizer updates. Record skipped updates and missing log intervals.
- [ ] Compare matched recipes and thresholds. Start with A0 across all three sizes; add the same sparse recipes at the same threshold rather than comparing unrelated runs.
- [ ] Retain the actual learning-rate schedule and any recorded training/validation loss histories.

Do not infer nonconvergence from a larger raw global norm alone. Parameter count, parameter scaling, gradient aggregation, stochastic variation, and optimization settings can change that norm. Dividing by `sqrt(parameter_count)` is a supplementary descriptive normalization, not a universal convergence test.

**Deliverable:** A log-field provenance note and machine-readable aligned records. If the logs are absent, explicitly record that the author's observation was not verified and continue other revision tasks.

**Done when:** The agent knows what the plotted norm means and which cross-size comparisons are actually valid.

### G02 — Evaluate a finite-budget explanation without claiming it is proven

**Priority:** P1. **Effort:** Low from retained logs. **Dependencies:** G01. **Locate:** §4.4 final paragraph; Appendix C.4 / new training-dynamics appendix.

- [ ] Plot retained task loss and verified pre-clipping task-gradient norms against tokens, using the same chosen recipe set in all sizes. Preserve raw curves; any smoothing must be documented and must not replace the endpoint.
- [ ] Include learning-rate information. A declining norm or flattening loss under a decayed rate alone is not a fair convergence comparison across different schedules.
- [ ] Inspect a fixed late-run interval, defined before outcome-specific selection; for example, the final 20% of logged optimizer updates, if coverage is sufficient.
- [ ] Quantify the late training-loss trend and retained validation trend when available. Do not invent intermediate validation evaluations.
- [ ] Report whether 410M is still making observable progress at the stopping point and whether this differs from the smaller models.
- [ ] Keep the 712-update budget, 1.493B tokens, tokens-per-parameter differences, and lower 410M learning rate visible in the methodological context.

**Outcome-specific language:**

- Stronger evidence: “The retained trajectory continues to improve near the stopping point, and the task-gradient diagnostic remains elevated under the stated measurement. These observations are consistent with a training-budget limitation.”
- Norm alone: “The larger terminal norm motivates inspecting training dynamics, but does not establish undertraining by itself.”
- No corroboration: Report what the logs show and do not assign the loss reversal to premature stopping.

Do not write “410M failed because it did not converge,” “longer training would restore the expected scaling,” or “the sparse penalty is explained by undertraining.” A matched continuation is needed to test the causal explanation and may change dense and sparse recipes differently.

**Done when:** The budget hypothesis is either supported by specified diagnostics, retained as a clearly unverified interpretation, or removed. It is never inserted as an established explanation because it sounds plausible.

---

## 6. Figure and table edits

### V01 — Restore and expand the cross-size visual argument

**Priority:** P0. **Effort:** Low. **Dependencies:** R01–R03, A01. **Locate:** current Figure 8, p. 20; original Figure 5, p. 10; §4.4 reference. **Source:** [M], [O].

**Mandatory design:** Restore a main-text figure comparing 14M, 70M, and 410M side by side. Preserve the original figure's direct visual comparison rather than replacing it with a numerical ranking table.

- [ ] Top row: absolute validation loss versus `100 S_model` for each size.
- [ ] Bottom row: loss relative to **unmodified same-size A0** versus `100 S_block` for every recipe in that size. State the normalization; do not normalize each recipe by its own reach.
- [ ] Include A4-OL1 and A7-OL1 across all five trained thresholds.
- [ ] Include the full A0 + clipping and A1-H + clipping trajectories across all ten targets. Show their baseline points distinctly.
- [ ] Retain the A4/four-site-clipping and A7 architectural-reach guides in the raw view. A1-H + clipping uses the four-site guide; its unmodified endpoint has h-only intervention reach.
- [ ] Use a single consistent recipe-to-color/marker mapping across the paper. Retain the existing visual scheme unless there is a legibility problem. Use open markers for post-hoc trajectories and filled markers for trained endpoints.
- [ ] Mark or annotate `kappa=0.5` so the recurring comparison is immediately locatable. Also make the low-threshold reversal visible rather than drawing only favorable segments.
- [ ] Distinguish “separately trained thresholds” from “post-hoc targets on one fixed checkpoint” in the legend or caption.
- [ ] Use shared y-axis limits within a row where legible. Different raw x-limits by size are legitimate but must be apparent; the normalized bottom row should use a common x scale.
- [ ] Preserve full-range trajectories in the appendix. If a main-text view omits high-loss tails, mark and count omitted points in the caption; do not hide them silently.
- [ ] Recalculate the number of distinct evaluations. The previous caption's “60 evaluations” is no longer correct when A1-H clipping is added. Distinguish duplicate plot appearances and near-identical `p=0` records from new runs.

**Preferred title:** “Quality–sparsity trade-offs across the Pythia family.” The author's familiar “frontier” wording is acceptable only if the figure explicitly distinguishes an observed nondominated set from the paths connecting all evaluated settings. Do not manufacture fitted frontiers or hide dominated observations.

**Caption structure:**

1. What repeats at the high threshold, stated as a named comparison.
2. What each column and row measures.
3. Which curves are trained recipes and which are post-hoc clipping of GELU/ReLU-trained checkpoints.
4. Why architectural reach changes; where the complete ranges are retained.
5. One concise sentence on the fixed token budget and changed 410M learning rate.

**Space fallback:** First reduce redundant text or remove the now-unnecessary main-text boundary-threshold table. Keep a legible three-size raw figure in the main text even if the normalized row must remain in the appendix. Do not again demote all cross-size visual evidence to the appendix merely to preserve the systems figure.

**Done when:** Without reading the appendix, a reviewer can see the beyond-14M recurrence, the ReLU/clipping behavior, and the nonuniform 410M response.

### V02 — Remove “Cohort means” without losing execution attribution

**Priority:** P0. **Effort:** Low. **Locate:** current Figure 5, p. 7, subplot (a); Appendix Table 12 and Table 13.

- [ ] Remove the aggregate means panel and its subplot caption. Do not replace it with a differently named aggregate bar/lollipop plot.
- [ ] Preserve the three implementation summaries in §4.5 prose and the existing appendix implementation table: all-skips-off `1.1829x`, K050 `1.2340x`, attention-dense `1.2506x` over native execution, subject to source revalidation.
- [ ] Keep the incremental sparse-path factor `1.0432x` and count `14/30` in the result text. These are interpretation controls, not mandatory chart panels.

**Preferred replacement figure:** Two intervention-level views generated from retained records:

- **Left:** BF16 validation loss versus absolute latency for the measured 14M checkpoints. Show matched fused/skips-disabled and K050 results using a consistent execution marker convention. Identify A0 and the informative A4/A7 pair at `kappa=0.5`; do not mark a global “winner.” Put the attention-dense variant in the appendix if the main panel becomes crowded.
- **Right:** Retain the checkpoint-level incremental sparse-path factor versus `S_model`, with a one-factor reference line, recipe symbols, and an explicitly descriptive fit only if it adds information. Label the FP16-count/BF16-timing distinction.

**Permitted simpler layout:** A single checkpoint-level runtime panel, with attribution and absolute-latency examples in prose, if two panels crowd the restored cross-size figure. Do not sacrifice readable cross-size evidence to add more kernel panels.

**Measurement constraints:**

- Use BF16 quality values alongside BF16 timings, not FP16 endpoint loss values presented as identical.
- Do not time clipping results by assumption; the existing timing cohort contains unmodified trained checkpoints.
- A ratio of rounded latency summaries is not the geometric mean of paired timing ratios. Define plotted statistics correctly.
- Do not present a ratio of separately native-normalized speedups as a directly paired cross-implementation measurement.
- Keep the A4/A7 comparison concrete: at `kappa=0.5`, the displayed K050 summaries are approximately `0.4790`/`0.4789 ms`, while BF16 losses are `5.6623`/`5.7076` and their FP16 `S_model` values differ substantially. Describe the summaries as close, not as statistically equivalent.

**Done when:** The figure compares actual interventions and execution paths, not aggregate winners, while the text preserves the fusion/sparsity distinction.

### V03 — Relocate the quality-budget table and trim redundancy

**Priority:** P0. **Effort:** Low. **Locate:** current Table 1 and §4.1 paragraph beginning “Retrospective quality budgets change which endpoint is preferable.”

- [ ] Remove Table 1 from the main argument and remove its winner-centered paragraph.
- [ ] Keep the reduction code, source rows, and a supplementary table or machine-readable artifact. Relabel it as an optional retrospective quality-budget view, not the study's objective.
- [ ] Replace the paragraph with an intervention question: how does the representation produced by ReLU training change the response to subsequent clipping? Use R02's verified observation.
- [ ] Keep absolute quality costs visible in the surrounding text and restored curves. Removing winner selection must not become hiding unfavorable losses.
- [ ] Keep the operation-contribution Table 2 or integrate its two numbers into the distribution paragraph if space is tight; it explains a result rather than ranking arbitrary candidates.
- [ ] Consider moving current Table 3's boundary-threshold summary to the appendix once the restored main figure and a quantitative sentence carry the same comparison. Keep all five thresholds in the complete results.
- [ ] Update LaTeX labels and references by symbolic labels, not hand-edited figure numbers.

**Done when:** Main-text space is spent explaining intervention effects. No result records are deleted or selectively excluded.


---

## 7. Paragraph-by-paragraph LaTeX edits

The method paragraphs are specified in O02, T01–T02, and A01. The following tasks cover the rest of the paper. Resolve actual source files and labels through B00. Do not duplicate the same explanation in several sections.

### W01 — Experimental setup: make the comparisons easy to follow

**Priority:** P0. **Locate:** §4 opening, p. 3. **Sources:** [M, §4; Appendix C.4].

**Paragraph 1 — anchor: “We pretrain Pythia models from random initialization…”**

- [ ] Preserve sizes, conditions, dataset, fixed updates/tokens, final-checkpoint evaluation, validation definition, and single-seed status.
- [ ] Explain the budget as a shared experimental protocol, not evidence that the three sizes have reached an equivalent training stage.
- [ ] Leave detailed optimizer settings in Appendix C.4; retain a concise adjacent note about the lower 410M learning rate.

**Paragraph 2 — anchor: “Within each size, paired conditions share initialization…”**

- [ ] State what is fixed in the pressure on/off contrasts and what changes in the placement comparisons.
- [ ] Explain that the multisite OL1 settings are fixed to focus the sweep, linking to the new rationale rather than asserting they were tuned or validated as optimal.
- [ ] Update the normalization language using O03. Do not imply effective per-site dilution proportional to the written coefficient without inspecting the cap regime.
- [ ] Keep the independent-site-set definition and one compact recipe key accessible in the main text.

**Paragraph 3 — anchor: “The evaluation-only reference applies uniform magnitude clipping…”**

- [ ] Introduce clipping as a second intervention applied to already-trained representations, including A0 and A1-H.
- [ ] State the calibration policy, four sites, fixed weights, and target-versus-achieved-sparsity distinction.
- [ ] Retain its limited allocation scope, but do not make the paragraph mostly an explanation of why the baseline is inadequate.

**Done when:** A reader knows which differences can be attributed to a matched intervention and which are comparisons of different complete recipes.

### W02 — Quality–sparsity results: compare interventions, not budget winners

**Priority:** P0. **Locate:** §4.1, p. 4. **Dependencies:** R02, V03.

**Paragraph 1 — anchor: “Local pressure and broader nonlinearities occupy different regimes…”**

- [ ] Keep the useful distinction between local pressure and broader thresholding.
- [ ] Retain a concrete local-pressure result, with both loss and sparsity, without presenting L1 or OL1 as uniformly superior.
- [ ] Retain the aggressive A7-OL1 endpoint and its actual quality cost; maximum sparsity is context, not the sole contribution.

**Paragraph 2 — replace “Retrospective quality budgets change which endpoint is preferable…”**

- [ ] Introduce the ReLU/clipping comparison from R02.
- [ ] Name the exact clipping regime and size(s) where the verified pattern occurs. Use a quantitative pair or a clearly identified crossing rather than a generic “better frontier.”
- [ ] State the base ReLU cost separately from the added clipping cost when relevant.
- [ ] End with the implication: training the activation function changes the representation exposed to later thresholding; intervention timing and nonlinearity can interact.
- [ ] Point to the cross-size figure for recurrence and the full supplementary trajectories for unfavorable settings.

**Done when:** This section motivates understanding interventions without turning into a model-selection contest.

### W03 — Paired effects and the historical pressure-target audit

**Priority:** P0. **Locate:** §4.2, p. 5; historical paragraph in Appendix D.6, p. 25. **Dependencies:** O03. **Source:** [M, Tables 7–8; historical audit].

**Paragraph 1 — anchor: “Pressure’s additional value depends on threshold and targets…”**

- [ ] Preserve the ReLU cost and local L1/OL1 comparison.
- [ ] Explain that the optimizer comparison is between complete update rules. Do not demand an optimizer-contribution story or hide the `lambda=1` reversal.
- [ ] Replace any implication that OL1 is chosen because it wins the local sweep with the fixed-instrument rationale in §3.1.

**Paragraph 2 — anchor: “For A4, adding OL1 improves both axes…”**

- [ ] Preserve the quantitative A4/A7 pressure increments at the same threshold.
- [ ] State the useful finding first: the same pressure mechanism has a different marginal effect when the thresholded sites and target objective change.
- [ ] Explain the remaining distinction in terms of target-gradient mixture and cap-dependent scaling, following O03.

**Optional short paragraph 3 — promote the audited h-only pressure comparison.**

- [ ] Move the scientific portion of the historical audit out of kernel qualification and into a pressure-target appendix, referenced here.
- [ ] Verify all five original configs, checkpoint identities, and diagnostic precision before quoting comparisons.
- [ ] State that broad thresholding with pressure only at `h` was a distinct historical condition, not one of the original 54 primary conditions. Give it an unambiguous descriptive label such as “A4 with pressure at h only,” retaining its raw run ID.
- [ ] Use the observation to motivate separating thresholding placement from pressure placement. Do not make a causal claim about the changed denominator alone.
- [ ] Do not expand the 30-checkpoint timing cohort or the 54-condition headline silently. A separate audited comparison is sufficient.

**Done when:** The section explains a real design choice—where to apply pressure—without turning normalization arithmetic into an untested causal mechanism.

### W04 — Distribution results: connect zero mass to the measured computation

**Priority:** P1. **Locate:** §4.3, pp. 5–6. **Source:** [M, Figure 4; Tables 2, 9–11].

**Paragraph 1 — anchor: “More FFN zeros need not mean more model-wide sparsity…”**

- [ ] Keep exact-zero masses adjacent to the distributions and define the pooled `h,m` and `q,k,v` groups.
- [ ] Explain why a narrow nonzero peak is not the same as zero mass.
- [ ] Distinguish directly thresholded attention operands from nonlocal distribution changes under A4.

**Paragraph 2 — anchor: “Operation counts explain the ranking reversal…”**

- [ ] Preserve the operation-level explanation: increased QK/PV contributions outweigh reduced projection contributions.
- [ ] Tie this back to architectural reach: reach identifies which work can be affected; the measured counts identify how much of that work has zero operands.
- [ ] End with a concrete consequence: local FFN sparsity alone can reverse the apparent ranking of the complete recipes for model-wide zero products.

**Paragraph 3 — replace the current diagnostics-only disclaimer at the end.**

- [ ] State the verified all-zero-input result first: pooled `z` rows are entirely zero at about 97.15% for A4-OL1 and 91.20% for A7-OL1 at the high threshold in the BF16 diagnostic.
- [ ] Explain the algebraic consequence carefully: the affected output projection returns its bias on those rows, if that bias exists. This says nothing by itself about the entire residual stream or all context use.
- [ ] Give one useful open question: whether the remaining nonzero branch contributions retain meaningful context dependence.
- [ ] Keep detailed per-layer diagnostics in the appendix; do not allow a speculative collapse narrative to displace the intervention findings.

**Done when:** Readers understand what the distribution evidence explains and what the more extreme regime invites them to test.

### W05 — Cross-size section: restore a positive, quantitatively bounded argument

**Priority:** P0. **Locate:** §4.4, p. 6. **Dependencies:** R02–R03, V01, G02.

Replace the current two compressed paragraphs with approximately four short paragraphs:

**Paragraph 1 — recur beyond 14M.**

- [ ] Open with the named high-threshold A7-OL1 versus A4-OL1 relationship at all three sizes.
- [ ] Give its paired loss/sparsity differences compactly or point to visible annotations. Say this is evidence beyond the 14M architecture, not three independent replications.
- [ ] Immediately identify the lower-threshold ordering changes as the boundary of the recurring relationship.

**Paragraph 2 — separate learning from architectural accounting.**

- [ ] Explain the increase in reach and model-wide sparsity using the shrinking dense-head share and changed operation mix.
- [ ] Refer to the normalized row so readers can see what remains after removing head share.
- [ ] Do not describe the normalized plot as controlling for all architecture differences.

**Paragraph 3 — ReLU plus post-hoc clipping.**

- [ ] State the verified outcome from R02 across sizes, including any crossing or unfavorable region.
- [ ] Explicitly distinguish increasing structural reach, initial ReLU zero mass, incremental clipping sensitivity, and final loss.
- [ ] Do not assert a monotonic scale benefit unless the actual relevant metric supports it.

**Paragraph 4 — absolute quality and the training-budget question.**

- [ ] Explain that 14M→70M improves absolute quality for the compared recipes, but not necessarily their same-size dense-relative penalty.
- [ ] Name which 410M trajectories depart from that pattern. Do not apply A0's deterioration to the high-threshold sparse endpoints that actually improve.
- [ ] Include a concise training-dynamics observation only if G02 verifies it. Otherwise state the fixed-budget protocol and leave the premature-stopping explanation as an open question.

**Done when:** The section's main message is a useful cross-size observation, not a defense of why a scaling law was not established.

### W06 — Runtime section: explain implementation-dependent outcomes

**Priority:** P0. **Locate:** §4.5, pp. 7–8. **Dependencies:** V02.

**Paragraph 1 — replace the TwELL/provenance-led opening.**

- [ ] Open with the execution question: which counted zeros improve full-model latency once fusion and skip overhead are separated?
- [ ] State the workload, reference implementation, and numerical qualification concisely.
- [ ] Move detailed TwELL compatibility history, coding-agent model provenance, and the absence of human-only-development controls to their existing appendix/AI-use locations. They are not the main scientific result.

**Paragraph 2 — quantitative attribution, without a means plot.**

- [ ] Report native, matched fused/skips-disabled, and sparse-path comparisons, keeping the aggregate figures in prose.
- [ ] Define speedup as `native latency / candidate latency` for paired samples, and define the incremental factor as the ratio of native-relative speedups.
- [ ] Correct Table 14's “K050/native” wording if it names latency ratios in the wrong direction. Inspect the script before changing a formula.
- [ ] Keep the search result separate from the later final sweep and attention-dense ablation.

**Paragraph 3 — informative intervention example.**

- [ ] Use the A4/A7 high-threshold comparison: more counted zeros but nearly identical reported specialized latencies and different losses/native baselines.
- [ ] Explain why the baseline-normalized speedup can look better without selecting a better absolute-latency intervention.
- [ ] Avoid equivalence tests or causal runtime claims that were not performed.

**Paragraph 4 — attention skipping and the physical granularity.**

- [ ] Keep the instruction-skipping negative result and its measured skipped fractions.
- [ ] Explain the mismatch between scalar zero products and whole-fragment bypass, plus recurring detection/control costs.
- [ ] End with the implementation lesson: architecture-level reach can justify investigating attention, but profitability still depends on granularity, shape, and overhead in the measured workload.

**Done when:** The runtime subsection supports the intervention argument and preserves attribution without dominating the paper or relying on a bar chart of means.

### W07 — Related work: explain the design lineage, not a list of citations

**Priority:** P1. **Locate:** §2.1 and §2.2, p. 2. **Dependencies:** T01, final methods/results story.

- [ ] Keep the distinction between post-hoc intervention, train-time nonlinearity/pressure, and broader placement.
- [ ] Use the verified literature facts from T01 once; avoid repeating a full literature summary in both §2.1 and §3.1.
- [ ] State which idea motivated the current study and which implementation choices differ. Do not claim to evaluate Q-Sparse or Spark directly unless their actual methods are implemented and compared.
- [ ] Explain that this study uses a simpler fixed-cutoff mechanism to examine distributional adaptation and intervention interactions.
- [ ] Keep workload-specific kernel distinctions, but remove unnecessary detail that belongs in execution provenance.
- [ ] Do not add new efficiency records, dates, or state-of-the-art claims unrelated to the revision.

**Done when:** Prior work explains why the study's controlled choices are interesting rather than becoming a defensive enumeration of missing baselines.

### W08 — Discussion and conclusion: three lessons, not three more warnings

**Priority:** P0. **Locate:** §5, p. 8. **Dependencies:** verified result edits.

Write approximately three compact paragraphs:

**Paragraph 1 — intervention interaction.** State how the threshold and pressure objective change pressure's marginal effect. Add the verified ReLU/clipping implication. Avoid listing which checkpoint wins.

**Paragraph 2 — architecture and cross-size evidence.** Explain the practical use of reach before training and the particular relationship that recurs beyond 14M. State the differing low-threshold behavior as a boundary, not a contradiction to erase.

**Paragraph 3 — execution and next question.** Connect zero counts to the measured granularity/overhead result. End with the most informative unresolved question supported by the paper: pressure direction versus weighting, training-budget sensitivity, or larger-width execution, choosing only those still genuinely unresolved after the audits.

- [ ] Keep one concise scope statement covering single-seed evidence, selected larger-size recipes, clipping policy, and measured runtime workload.
- [ ] Do not end with “our results provide valuable insights.” State the actual lesson.
- [ ] Do not infer a universal recommendation from a complete-recipe observation.

**Done when:** A reader can name at least two experimental decisions they would now investigate differently.

### W09 — Introduction: make the intended contribution legible

**Priority:** P0. **Locate:** §1, pp. 1–2. **Dependencies:** all major results/visuals stable.

**Paragraph 1 — anchor: “Activation sparsity can reduce inference cost…”** Keep the practical question: how to place and combine interventions so that the resulting zeros affect useful computation at an explicit quality cost.

**Paragraph 2 — anchor: “Existing recipes combine these choices differently…”** Introduce pressure, thresholding, and placement. State the study's scope as examining interactions under a common protocol, not reproducing every complete prior method.

**Paragraph 3 — anchor: “We pretrain Pythia-family models from scratch…”** Give two quantitative intervention findings and their beyond-14M extension. Introduce reach as a planning quantity that can be computed before training. Include the ReLU/clipping result only at the strength supported by R02.

**Paragraph 4 — anchor: “A human-guided coding-agent case study then measures execution…”** Lead with the scientific role of the case study, not the development tool. Give a brief attribution result and the attention-skipping counterexample. Keep coding-agent details out of the contribution claim unless they are themselves evaluated.

- [ ] Remove “small quality budgets can favor clipped endpoints” as a headline contribution; replace it with the intervention-level observation.
- [ ] Keep the per-size and same-size quality comparisons unambiguous.
- [ ] Avoid copying the abstract's phrasing and numbers wholesale.

**Done when:** The introduction promises the paper the reader will actually encounter in the main figures and results.

### W10 — Abstract: update last

**Priority:** P0. **Locate:** abstract, p. 1. **Dependencies:** W01–W09, Q01 numerical checks.

Use roughly 7–9 sentences, adapted to the venue's actual space constraints:

1. The design question: inducing zeros is not sufficient without placement, quality, and execution context.
2. The controlled study and the division between detailed 14M comparisons and selected larger recipes.
3. The role of model-wide sparsity and a priori architectural reach.
4. A concrete conditional pressure effect, with a comparator.
5. The cross-size recurrence of the named high-threshold recipe relationship, not an unspecified “advantage.”
6. The ReLU/post-hoc finding only if verified and sufficiently central; otherwise keep it in the introduction/results.
7. The execution result with clear attribution to fusion versus incremental sparse paths, choosing only enough numbers for readability.
8. A substantive design implication, not a general statement that all sparse execution is useful.

- [ ] Do not use the abstract to advertise a theoretical actual-loss-preservation guarantee for OL1.
- [ ] Do not claim full convergence, universal scaling, or broad dominance of clipping baselines.
- [ ] Do not return to maximum sparsity as the only training result, or make aggregate means the only execution result.
- [ ] Keep one-seed and workload scope where necessary, without making the final sentence solely a disclaimer.

**Done when:** Each abstract finding has a visible main-text comparison and a traceable number or derivation.

---

## 8. Optional evidence: choose tests after the audits

Do not launch these automatically. Write an experiment request naming the unresolved claim, minimum conditions, stopping rule, evaluation protocol, expected cost, and how each possible outcome would change the paper. Short training runs are mid effort; aggregate packages may become high effort. Larger-model training and long continuations are high effort.

### E01 — Fill a ReLU/clipping comparison gap

**Effort:** Mid for short checkpoint evaluations. **Trigger:** Existing quantile targets do not resolve an important comparable-sparsity comparison in R02.

- [ ] Evaluate a small number of additional clipping targets on the fixed A0 and A1-H checkpoints using the same calibration/evaluation protocol.
- [ ] Keep calibration on training data; do not tune thresholds directly to validation labels/loss.
- [ ] Report achieved sparsity and both total and incremental loss costs.
- [ ] Add every evaluated target to the supplement, not only successful comparisons.

**Insertion:** R02, V01, W02/W05. No new training is needed for this specific question.

### E02 — Test the OL1 rationale with a small diagnostic, not a new optimizer campaign

**Effort:** Mid. **Trigger:** No saturation logs exist and an empirical robustness sentence is important to the narrative.

- [ ] Instrument a small, clearly labeled diagnostic at representative thresholds, retaining task/pressure vectors, cap state, and effective correction norms.
- [ ] Test weights below and above the observed saturation threshold at fixed gradient/optimizer states first. This is a one-step mechanism check, not proof of training-trajectory robustness.
- [ ] Only run a short multi-weight training comparison if a trajectory-level claim is intended. Use matched initialization/data order and report it as a diagnostic when its budget differs from the primary experiment.
- [ ] For target weighting, choose genuinely different relative old/new-site weights or a cap-inactive regime. Do not spend compute on a globally rescaled vector that remains saturated and directionally identical.

**Insertion:** O03 and the pressure-target appendix. This is not a prerequisite for an honestly stated fixed-design rationale.

### E03 — Replicate the central intervention contrast

**Effort:** Mid per short/small run; price the aggregate package. **Trigger:** The final abstract depends on unreplicated endpoint differences.

- [ ] Select the few conditions carrying the main claim and add independent seeds, preserving matched initialization and data order within each seed.
- [ ] Use the same training budget for confirmatory comparisons. Shortened runs can screen a hypothesis but cannot be substituted for matched-budget endpoint evidence.
- [ ] Report paired differences across seeds, not a misleading error bar over threshold settings or validation blocks.
- [ ] Run the zero-threshold A4/A7 forward/gradient identity check as a low-cost implementation test; repeat training only when needed to diagnose a discrepancy.

**Insertion:** paired/cross-size claim wording and complete results. A null or reversed effect changes the claim; it is not grounds to omit the seed.

### E04 — Test the 410M finite-budget explanation

**Effort:** High. **Trigger:** Retained logs motivate the explanation and the authors wish to make a stronger causal statement.

- [ ] Continue a matched dense baseline and selected sparse recipe(s), not only the condition expected to improve.
- [ ] Define the continuation token budget and learning-rate treatment before inspecting outcomes. Preserve optimizer state when the test requires continuation rather than restart.
- [ ] Track task loss, validation loss, sparsity, verified gradient diagnostics, and effective update size.
- [ ] Distinguish “more training improves this checkpoint” from “the original 70M→410M reversal is resolved” and from “the sparse penalty shrinks.”
- [ ] A consistent longer-budget training design is stronger than a one-off warm-restart rescue; document schedule changes and their interpretation.

**Insertion:** G02/W05 and a training-dynamics appendix. Do not delay the entire rewrite waiting for this high-effort option.

---

## 9. Final verification and author handoff

### Q01 — Numerical and mathematical verification

- [ ] Recompute every quoted delta from unrounded source values; preserve pp versus percent conventions.
- [ ] Verify `S_model` versus `S_block` and the common A7 normalization; do not divide each recipe by a different reach.
- [ ] Verify the expanded figure's condition/evaluation counts and the A1-H + clipping reach.
- [ ] Keep FP16 training/count results and BF16 timing/quality results distinct.
- [ ] Check all speedup ratios, especially the Table 14 caption and incremental-factor axis.
- [ ] Run OL1 direction, cap, saturation, degeneracy, and global-rescaling tests. Label exact ideal properties separately from the stabilized implementation.
- [ ] Verify the theoretical half-space result with a brief proof or independent calculation; do not present a stronger actual-loss guarantee.
- [ ] Verify every training-dynamics statement against a known log field and matched recipe.
- [ ] Confirm that removing Table 1 and the means subplot does not remove unfavorable quality results or runtime attribution.

### Q02 — LaTeX build and visual inspection

- [ ] Regenerate figures/tables through source scripts, not manual raster edits.
- [ ] Use the documented build command; inspect warnings for missing references, undefined citations, multiply defined labels, and overfull boxes.
- [ ] Render and inspect every changed page at readable size. Check legends, tick labels, markers, panel letters, clipping/reach guides, and placement relative to first reference.
- [ ] Confirm the cross-size figure is in the main text and readable at final printed width.
- [ ] Confirm “Cohort means” is gone and no equivalent aggregate panel has appeared under another name.
- [ ] Confirm the main-text quality-budget winner table is gone; the underlying data remain available.
- [ ] Do not reduce fonts or margins to fit prose. Trim duplicate scope statements, redundant tables, and development provenance first.
- [ ] Preserve the venue template and documented page limit. Do not assume a future limit from an earlier PDF's page count.

### Q03 — Human-reader and claim audit

Read only the main text and figures first, without consulting the appendix. Then answer:

- [ ] Why use a fixed threshold rather than choose a surviving count?
- [ ] Why use OL1 with a fixed relative budget, and what property is actually protected?
- [ ] What can architectural reach predict before training?
- [ ] Which pressure effect depends on the threshold or target mixture?
- [ ] How does ReLU training change subsequent clipping, and what does “better” mean in the reported region?
- [ ] Which relationship recurs at all three sizes, and which one changes at 410M?
- [ ] What do the training logs support about the stopping point, if anything?
- [ ] Why can more model-wide zero products fail to reduce latency?

A “yes” requires a concrete answer, not a paragraph saying the issue is complicated.

Search for and manually inspect occurrences of:

```text
preserves task loss
maximum orthogonal step
does not hurt
hyperparameter robust
converged
premature stopping
scales better
consistent across all
all attention sites
train-time clipping
Cohort means
Maximum observed sparsity
winner
K050/native
```

These are audit triggers, not blind find-and-replace instructions. A source quotation or a clearly bounded discussion can legitimately contain them.

### Required handoff

Deliver:

1. Edited `.tex`/`.bib`, figure scripts/assets, and the compiled PDF, with actual paths.
2. `revision-v2/changes.md`, mapping each task to its paragraph/figure edits.
3. `revision-v2/claims.csv` and the derived data needed to reproduce new numerical comparisons.
4. A short **verified / conditional / unresolved** report: OL1 saturation, ReLU/clipping comparison, training dynamics, cross-size recurrence, and runtime attribution.
5. Exact regeneration/build/test commands, their outcomes, and any required data or environment dependencies.
6. A concise recommendation on additional evidence: no new runs needed for a particular wording, a targeted mid-effort test, or an optional high-effort continuation. Do not promise acceptance.

### Restart prompt for the next coding agent

> Read this task file and `revision-v2/state.md`. Locate the next unfinished task and its declared source blocks. Preserve the author's intervention-centered framing: restore the main cross-size figure, remove the means subplot and winner table, explain method choices, and keep empirical claims tied to verified records. Work in a small batch, regenerate affected assets, build, inspect the diff/render, and update the state. Do not run new training or GPU evaluations without explicit authorization. Do not replace conditional adaptive-direction geometry with an actual-loss guarantee, or a gradient-norm observation with a convergence conclusion.

---

## 10. Source register and claim provenance

The repository is the final source of truth for implementation details and unrounded measurements. The references below identify what was available when this task file was prepared. The task-file author has not inspected the repository's training code, raw clipping records, or gradient logs.

### [M] Base revised manuscript

*Activation Sparsification in Transformers: Pressure, Thresholding, and Site Placement*, `main(20260909-211032).pdf`, supplied by the authors.

| Location | Relevant evidence |
|---|---|
| §3.1 and Figure 1, pp. 2–3 | Threshold definitions, current pressure explanation, actual one-sided/symmetric site assignments. |
| Appendix A.2, p. 11 | Implemented OL1 preconditioning, conflict-conditioned projection, cap, and coordinate-system details. |
| §3.2; Appendix B.2/C.2, pp. 3, 12, 14 | Model-wide accounting, architectural reach, and operation formulas. |
| Table 1 and §4.1, p. 4 | Retrospective quality-budget presentation to relocate. |
| Figure 5 and §4.5, pp. 7–8 | Means subplot to remove and native/fused/sparse execution attribution. |
| Tables 6–8, pp. 16–17 | All endpoints, paired effects, and cross-size recipe contrasts. |
| Table 10, p. 19 | BF16 all-zero projection-input rows, not a direct context-sensitivity experiment. |
| Figure 8, p. 20 | Cross-size visual comparison to restore to the main text. |
| Figures 10–12 and Appendix D.5, pp. 22–24 | Complete post-hoc trajectories; raw per-target records must be inspected for new exact comparisons. |
| Appendix D.6, p. 25 | Historical h-only pressure audit. |
| Tables 12–14, pp. 26–28 | Runtime summaries, BF16 losses, and association checks. |
| Appendix C.4, pp. 14–15 | Realized budget, gradient clipping, optimizer settings, learning-rate differences, and tokens per parameter. |

Selected source citations supporting the main safeguards:

- OL1 implementation and geometry: fileciteturn3file0L1085-L1113
- Thresholding and actual site assignments: fileciteturn3file0L178-L185 fileciteturn3file0L264-L266
- Reach formulas and architecture configurations: fileciteturn3file0L1407-L1432
- Cross-size endpoint values: fileciteturn3file0L1625-L1679
- Historical pressure comparison: fileciteturn3file0L2519-L2527
- Clipping protocol and precision: fileciteturn3file0L2272-L2284
- Runtime summary definitions: fileciteturn3file0L2628-L2641 fileciteturn3file0L2659-L2665

The page/section descriptions remain usable when conversation-specific citation chips are unavailable to the coding agent.

### [O] Earlier manuscript and visual reference

`main(20260909-182851).pdf`, supplied by the authors, 27 pages. Use original Figure 5 on p. 10 as the visual reference, and original §3.1 on p. 3 for the thresholding motivation that was compressed out of the rewrite. Do not restore outdated ceiling terminology or broad dominance/scaling claims.

fileciteturn3file1L276-L292 fileciteturn3file1L1025-L1034

### [Q] Q-Sparse — external primary-source verification

Wang et al. (2024), *Q-Sparse: All Large Language Models can be Fully Sparsely-Activated*, arXiv:2407.10969, §2 and §4.1. Verified source for the method distinctions used in T01; it is not an experimental baseline reproduced by the supplied manuscript. Reuse the repository's existing BibTeX key after checking it.

Source location: `https://arxiv.org/html/2407.10969v1`

### [S] Spark Transformer — external primary-source verification

You et al. (2025), *Spark Transformer: Reactivating Sparsity in FFN and Attention*, arXiv:2506.06644, especially §3, equations 10–11, and §3.2. Verified for the statistical-selection motivation in T01. Do not equate its attention-position mechanism or threshold transformation with the current paper's feature-coordinate maps.

Source location: `https://arxiv.org/html/2506.06644v2`

### [P] Pythia — external primary-source verification

Biderman et al. (2023), *Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling*, ICML/PMLR 202, Appendix E, Table 6, PDF p. 22. The published configuration includes gradient clipping at 1.0. This supports the norm-limiting design lineage, **not** a claim that the manuscript's whole optimizer recipe is unchanged or that OL1 capping is the same operation.

Source location: `https://proceedings.mlr.press/v202/biderman23a/biderman23a.pdf`

### New reasoning versus unavailable evidence

The OL1 idealized saturation/half-space argument and the simplified architectural-reach formulas are mathematical derivations included in this task file for verification. They are not new experimental measurements. The hypothesized ReLU/clipping improvement region and the author's terminal gradient-norm comparison require the repository audits above before stronger empirical wording is approved.
