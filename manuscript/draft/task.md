# Agentic rewrite runbook
## Activation Sparsification in Transformers: Pressure, Thresholding, and Site Placement

**Purpose:** Turn the current manuscript into a more convincing, readable, and informative empirical paper through traceable, paragraph-level edits. Separate what can be improved using existing evidence from claims that require additional experiments.

**Source version:** `main(20260909-182851).pdf`, 27 pages, supplied by the authors. Page numbers below refer to this PDF, not a future compilation. Paragraph anchors quote the opening words of the original paragraph; paragraphs can continue across pages or around figures.

**Source SHA-256:** `a2aa5fefd6cfe0ccc51d025f9ec692db366f929a6cefbaabb401454dfecc2f1a`.

**Available for this plan:** The manuscript and its rendered pages. Authoring sources, checkpoints, training logs, supplementary data files, and timing records were not supplied here. Their existence in the manuscript is not proof that the rewriting agent has access to them.

**Recommended route:** Execute the low-effort rewrite and existing-data audit first. In parallel, obtain author approval for a small, decisive set of mid-effort controls and replications. Reserve high-effort work for a specific unresolved claim. A stronger presentation cannot substitute for a missing comparison.

**Default operating mode:** `CURRENT_RESULTS_ONLY`. Do not launch experiments, change the scientific conclusions, or imply that proposed experiments have been completed. Switch individual tasks to `NEW_EVIDENCE_APPROVED` only after approval and ingestion of verified results.

---

## Start here: the next agent's first actions

- [ ] **START-01:** Read Sections 1–4 of this runbook. Record the manuscript's intended contribution in two sentences before editing.
- [ ] **START-02:** Locate the authoring source and existing data. Create the inventories in Section 4. If only the PDF is available, produce anchored replacement passages and figure specifications rather than pretending to edit unavailable LaTeX.
- [ ] **START-03:** Freeze the original manuscript and create a paragraph-anchor map. Never use a changing PDF page number as the only locator.
- [ ] **START-04:** Obtain a decision on the proposed terminology change from “sparsity ceiling” to “architectural reach.” Preserve the mathematics either way.
- [ ] **START-05:** Execute the rewrite in this order: methods and setup → results → figures and tables → appendices → discussion → related work → introduction → abstract and title → verification.
- [ ] **START-06:** Work on one coherent batch at a time, usually two to four paragraph tasks. Update the state file after every batch. Do not rewrite the whole manuscript in a single pass.

**Success condition:** A reader can identify the comparison behind each conclusion, understand the size and cost of the effect, distinguish a measured result from an explanation or hypothesis, and leave with useful questions for designing their own sparsification experiments.

**Not the success condition:** Making every intervention look effective, maximizing the number of favorable statements, or eliminating all limitations through wording.

### Navigation

1. [Strategy and acceptance rubric](#1-strategy-and-acceptance-rubric)
2. [Writing contract: informative, precise, readable](#2-writing-contract-informative-precise-readable)
3. [Terminology, notation, and numerical conventions](#3-terminology-notation-and-numerical-conventions)
4. [Agent workflow, evidence inventory, and checkpoints](#4-agent-workflow-evidence-inventory-and-checkpoints)
5. [Paragraph-by-paragraph main-text rewrite](#5-paragraph-by-paragraph-main-text-rewrite)
6. [Figure and table surgery](#6-figure-and-table-surgery)
7. [Appendix and reproducibility edits](#7-appendix-and-reproducibility-edits)
8. [Optional experiments and exact insertion points](#8-optional-experiments-and-exact-insertion-points)
9. [Final verification and author handoff](#9-final-verification-and-author-handoff)
10. [Portable source register](#10-portable-source-register)

---

<a id="1-strategy-and-acceptance-rubric"></a>

## 1. Strategy and acceptance rubric

### 1.1 The contribution to preserve

The manuscript studies pressure, thresholding, and site placement under a shared experimental protocol. Its most promising current contribution is not a universally better sparse-model recipe. It is a set of informative comparisons connecting intervention choices to activation distributions, operation-weighted zero counts, and measured execution.

Build the paper around **two primary findings**:

1. **An intervention's marginal effect depends on what is already in the recipe.** Pressure does not have a single quality–sparsity effect across thresholds and target sets.
2. **The ranking can change between local activation sparsity, model-wide zero-product counts, and runtime.** Broader attention interventions increase counted opportunity, but the tested attention-skipping path is slower than its dense-attention ablation.

Use the selected larger-model results as a **bounded extension of complete-recipe comparisons**, not as proof that all individual effects transfer or that sparsification obeys a scaling law. These distinctions are supported by the current paired comparisons, operation breakdown, and kernel ablations. [S08](#s08) [S09](#s09) [S10](#s10) [S11](#s11)

### 1.2 Two legitimate paper directions

| Direction | What the paper must establish | Recommended now? |
|---|---|---|
| **Empirical design study** | Nontrivial, well-supported relationships between interventions; clear accounting; meaningful boundaries and counterexamples; appropriate repeatability. | **Yes.** This best fits the existing evidence. |
| **Practical sparsification method** | A defensible quality–latency advantage over competitive alternatives, with adequate implementation and training controls. | Only if new comparisons support it. Do not imply this result through maximum sparsity or same-checkpoint speedup alone. |

Narrowing the claim is not a license to make the paper uninformative. “Sparsity does not always give speedup” is too generic by itself. Identify **which operations, which controls, how much skipped work, and what overhead-limited behavior** make the observed case instructive.

### 1.3 Readiness rubric

Score each dimension separately. Scores are planning judgments, not estimated acceptance probabilities or an official venue rubric.

- **0 — Missing or contradicted:** The claim lacks the necessary evidence, or the manuscript says more than the result supports.
- **1 — Partial:** There is relevant evidence, but a major ambiguity, control, or scope issue remains.
- **2 — Adequate for the stated claim:** The comparison and its limits are explicit and defensible.
- **3 — Strong:** The evidence is replicated or independently reinforced and supports a useful, clearly bounded conclusion.

| Dimension | Evidence needed for a score of 2 or higher | What a rewrite alone can accomplish |
|---|---|---|
| Contribution clarity | Two memorable findings, each tied to a specific comparison and a reader-relevant implication. | Substantial improvement. |
| Comparison validity | Explicit separation of fixed-site pressure effects, unpressured placement effects, and complete-recipe changes; no causal attribution to untested components. | Clarify existing contrasts; cannot create missing controls. |
| Repeatability | Independent training replications for the headline contrasts, or a clearly limited claim acknowledging the single-seed evidence. | Honest qualification; cannot establish training variability. |
| Quality and baseline fairness | Visible quality costs; accurately scoped post-hoc baseline; no confusion between quality-matched model comparisons and same-checkpoint execution. | Reanalysis and better presentation, subject to data access. |
| Runtime attribution | Native, matched-fusion, and sparse-path comparisons distinguished; workload and precision stated. | Major improvement using the reported ablations. |
| Scope and transfer | Larger-size findings restricted to what was tested; no extrapolation to unmeasured workloads. | Major improvement. |
| Insight and readability | Results tell the reader what changes, why the evidence matters, and where the inference stops. | Major improvement. |

**Do not average away a blocking flaw.** A polished introduction does not compensate for a numerical error or an unsupported practical-efficiency claim. A replication with mixed results is not a failed experiment; it changes the claim the paper can make.

### 1.4 Effort and priority

**Effort definitions:** Low = writing, verification, or analysis of already available records. Mid = short new runs, targeted checkpoint evaluation, or short calibration experiments. High = long training/ablation packages, larger-model experiments, or substantial new kernel engineering. Reclassify a collection of individually short runs when its aggregate cost becomes high. No wall-clock or GPU-hour estimate is assumed here.

**Impact:** 5 = addresses an acceptance-blocking objection; 4 = materially strengthens the argument; 3 = supporting improvement. Rank by relevance to the chosen paper direction, not by the possibility of obtaining a favorable result.

| Package | Highest-value tasks, in order | Expected effect on the acceptance case |
|---|---|---|
| **Low** | L1 (5/5): claim and narrative reset; L2 (5/5): quality and speedup attribution; L3 (4/5, potentially higher): audit omitted historical controls; L4 (4/5): terminology, figure, and reproducibility repair. | Makes the current paper substantially more credible, but is not a reliable standalone route to acceptance because design and repeatability gaps remain. |
| **Mid** | M1 (5/5): fixed-weight crossed placement/pressure controls; M2 (5/5): headline-contrast replication; M3 (5/5 when claim retained): stronger post-hoc comparison; M4 (4/5): precision-consistent and functional diagnostics. | The recommended investment. Can put an empirical-design paper into plausible acceptance territory if the findings remain informative and claims follow the outcomes. |
| **High** | H1 (4/5): longer-training confirmation at 70M; H2 (4/5, or 5/5 for an efficiency headline): one larger-model quality–latency study. | Strengthens generality or practical relevance. Worth doing only after the central comparison is sound and the remaining objection is specific. |

**Low-effort workstream map:** L1 = RW-40/RW-42/RW-10/RW-ABS; L2 = DATA-01/02 and RW-41/RW-45; L3 = DATA-03; L4 = STYLE/VIS/APP/QA tasks. These are workstream aliases, not additional experiments. Each paragraph task below remains the unit of execution.

**Candid decision:** Low-only is a defensible fallback, not the preferred bet. Low + targeted mid is the best expected improvement per unit effort. High effort is conditional, not an automatic requirement or a guarantee of acceptance.

---

<a id="2-writing-contract-informative-precise-readable"></a>

## 2. Writing contract: informative, precise, readable

### 2.1 Language and audience

- [ ] **STYLE-01:** Write the paper in consistent, natural academic English. Prefer US spelling unless the authoring source clearly follows another convention; choose once and apply consistently.
- [ ] **STYLE-02:** Write for an ML researcher who understands transformers but has not memorized this paper's recipe IDs or custom accounting. Define task-specific terms before relying on them.
- [ ] **STYLE-03:** Prefer concrete subjects and verbs: “adding pressure increases…” rather than “the incorporation of pressure yields an enhancement…”. Use active voice for experimental choices; passive voice is acceptable when the procedure matters more than the actor.
- [ ] **STYLE-04:** Keep technical precision without making every sentence carry every qualification. State a shared protocol once, then repeat only the details necessary to interpret a local comparison.
- [ ] **STYLE-05:** Prefer one central comparison per paragraph. As a soft guide, aim for three to six sentences and inspect sentences longer than about 30 words. These are editing prompts, not rigid limits.
- [ ] **STYLE-06:** Use prose for the argument. Reserve compact lists and tables for experimental inventories, definitions, and contrasts. Do not turn the paper into this runbook's checklist format.

### 2.2 Avoid jargon that obscures the experiment

Use the established term when it is precise; explain it at first use. Do not invent a vocabulary that makes simple observations sound like a new theory.

| Avoid or qualify | Prefer |
|---|---|
| “forces activations to zero” for the L1 objective | “encourages smaller activation magnitudes”; exact zeros are measured after the relevant transformations. |
| “synergy” without a specified interaction contrast | “adding pressure at this threshold changes sparsity by… and loss by…”. |
| “disentangles all three factors” | State which contrasts hold which factors fixed, and identify complete-recipe comparisons separately. |
| “near-optimal sparsity” | “close to the selected-site architectural reach,” with quality cost and denominator stated. |
| “FLOP savings” when reporting this paper's count | “zero-operand scalar products” or “logical multiplication opportunity.” |
| “preserves quality” based on a small single-seed difference | Report the observed loss change; add uncertainty when available. |
| “dominates the frontier” without checking the domain and endpoints | Name the comparator, evaluated region, and nondominance criterion. |
| “predicts speedup” for the existing cross-checkpoint fit | “is associated with measured speedup in this cohort.” |
| “attention sparsity is ineffective” | “the tested attention-skipping implementation is slower on this workload.” |
| “spillover mechanism” without an isolating test | “a distribution change at sites not directly targeted”; mechanism remains a hypothesis. |
| “best kernel” | “selected searched implementation,” or identify the faster ablation explicitly. |
| “robust,” “substantial,” “negligible,” or “competitive” without support | Give the measured magnitude, comparator, and relevant uncertainty or budget. |

Retain standard technical terms such as matrix multiplication, causal masking, softmax, geometric mean, and gradient projection when needed. Plain language does not mean removing the information needed to reproduce the method.

### 2.3 The informative-paragraph pattern

Use this pattern for substantive results, but vary the prose so the paper does not feel templated:

> **Question or contrast → observed result → explanation supported by measurement → implication with a boundary.**

A paragraph should usually answer:

1. What comparison is being made, and what is held fixed?
2. What changes, in the units the reader needs?
3. What does the evidence explain? What remains only a hypothesis?
4. What should a researcher examine or reconsider because of this result?

Not every methods paragraph needs a takeaway. Not every paragraph needs all four elements. Apply the test at the subsection level when a single paragraph would become overloaded.

**Bad use of “insight”:** Add speculative explanations, promotional adjectives, or a sentence saying “this provides valuable insights.”

**Good use of insight:** Reveal an unexpected reversal, explain a denominator effect, distinguish two apparently similar comparisons, or identify the operation-level bottleneck that changes a design decision.

### 2.4 Examples of the intended approach

**Example A — Distinguish local sparsity from model-wide opportunity.**

Weak: “A7 achieves superior sparsity through broader placement.”

Better: “At the largest tested threshold, the seven-site recipe has fewer pooled FFN zeros than the four-site recipe, yet more model-wide zero-product counts. The operation breakdown attributes the increase to QK and PV products. In this setting, choosing a recipe from FFN sparsity alone would select the wrong recipe for maximizing counted model-wide opportunity.”

Boundary to retain: this is a complete-recipe observation, not proof of a pure pressure-site effect or a latency advantage. [S09](#s09)

**Example B — Separate observed acceleration from its source.**

Weak: “Our sparse kernel gives a 1.78× speedup.”

Better: “The selected implementation is 1.78× faster than native execution at the search checkpoint. Against a matched fused implementation with sparse paths disabled, the additional factor is 1.311× at that checkpoint and 1.043× geometrically across the cohort. The native-relative result therefore combines fusion with a checkpoint-dependent contribution from sparse execution.”

Boundary to retain: the ratios compare execution of the same checkpoints, not different models at matched quality. [S11](#s11)

**Example C — Report a conditional pressure effect without overselling a tiny difference.**

Weak: “OL1 preserves quality and unlocks attention sparsity.”

Better: “For A7 at κ = 0.1, adding OL1 changes loss by +0.0008 and model-wide sparsity by +1.372 percentage points in the reported run. At κ = 0.5, the corresponding changes are +0.1265 and +12.096 percentage points. The additional sparsity and its quality cost depend on the operating point; independent seeds are needed before interpreting the smallest loss difference.” [S08](#s08)

These are examples of sentence structure and evidential restraint, not instructions to paste the same text into multiple sections.

### 2.5 Build an insight ledger, not a list of attractive claims

Before polishing a results subsection, fill one row in the ledger:

| Candidate takeaway | Existing evidence | Reader implication | Boundary or next question |
|---|---|---|---|
| Pressure has a threshold-dependent marginal effect. | Fixed-recipe pressure contrasts in Figure 3 / Table 5. | Evaluate the addition at the intended operating point, rather than assuming a fixed benefit. | Current cross-target contrasts also change objective normalization; replication is limited. |
| Local FFN sparsity can rank recipes differently from model-wide counts. | Figure 4, Tables 7–8. | Report operation-weighted counts alongside site sparsity. | More counted opportunity does not establish lower latency. |
| The dense-head fraction changes the interpretation of cross-size sparsity. | Section 4.4, Appendix C.2. | Inspect the denominator before attributing raw sparsity increases to a better learned representation. | Block-only normalization still retains differing operation weights. |
| Substantial instruction skipping can lose to dense execution. | Appendix D.4 and Table 9. | Measure the overhead and granularity of the actual sparse path. | No general break-even point or impossibility result is established. |
| A recipe can beat another sparse recipe while remaining much worse than A0. | Tables 1, 4, and 6. | Compare against both the paired recipe and the dense-model reference. | Same-checkpoint acceleration does not answer the model-selection question. |

Keep two primary takeaways and a few supporting ones. Do not manufacture an “insight” for every sweep value.

### 2.6 Precision and reader trust

- [ ] **STYLE-07:** Distinguish **measured**, **derived**, **inferred**, and **proposed** statements in the agent's evidence ledger. In manuscript prose, use natural signals such as “we measure,” “the accounting implies,” “is consistent with,” and “we leave untested.”
- [ ] **STYLE-08:** Put the decisive caveat beside the claim it limits. A limitation on page 26 does not repair an unqualified claim in the abstract.
- [ ] **STYLE-09:** Do not turn descriptive data into causal or statistical language through editing. “Significant” requires the relevant statistical evidence; “isolates” requires a suitable design.
- [ ] **STYLE-10:** Do not hide unfavorable or dominated settings. Keep complete results accessible and show the region relevant to the stated question in the main text.
- [ ] **STYLE-11:** Define the reference whenever using “improves,” “more,” “lower,” “faster,” or a delta. Recipe IDs alone are not an explanation.
- [ ] **STYLE-12:** Use exact benchmark scope. “Full-model” here is not cached token-by-token decoding, a production serving stack, or an all-hardware result.
- [ ] **STYLE-13:** Remove self-assessment such as “these comprehensive experiments convincingly demonstrate.” Replace it with the actual comparison or omit it.
- [ ] **STYLE-14:** End the paper with what the reader has learned, not with a larger list of claims than the experiments support.

---

<a id="3-terminology-notation-and-numerical-conventions"></a>

## 3. Terminology, notation, and numerical conventions

### 3.1 Freeze the vocabulary before editing

Create a shared glossary and use it in prose, equations, figure labels, captions, tables, and supplement documentation. Preserve the original recipe IDs; their names identify experimental records.

| Concept | Canonical expression | Required distinction |
|---|---|---|
| Training regularizer | **activation pressure**; introduce it as L1 pressure | Pressure encourages small magnitudes; it is not the thresholding map. |
| Plain regularized optimizer | **naive L1** in the methods, or **ordinary L1** if changed consistently | Keep `A1-H-L1` and data identifiers unchanged. Do not alternate labels without explanation. |
| Projected update | **orthogonal L1 (OL1)** | This is the specified update rule, not a guarantee of task-loss preservation. |
| Sparsifying function | **thresholding nonlinearity** | One-sided and symmetric variants are different transformations. |
| Site choice | **site placement** or **thresholding sites / pressure sites** | The two site sets need not coincide. |
| Zero proportion at a site | **activation sparsity at site s** | Never silently substitute pooled FFN sparsity. |
| Count-based model metric | **model-wide activation sparsity**, then **model-wide sparsity** | A fraction of declared matrix-product multiplications, not all operations or time. |
| Structural accounting quantity | **architectural reach** (proposed replacement for “sparsity ceiling”) | Not a universal upper bound on observed sparsity and not a quality-feasible target. |
| Reference model | **same-size A0 reference** | Distinguish from native execution of a sparse checkpoint. |
| Execution reference | **native PyTorch/SDPA execution of the same checkpoint** | This is an implementation comparison. |
| Fusion control | **matched fused implementation with all sparse paths disabled** | Preserve graph, checkpoint, nonlinearities, and workload. |
| Evaluation-only comparator | **uniform post-hoc magnitude clipping** | Not a reproduction of optimized TEAL allocation. |
| Cross-size result | **complete-recipe comparison across model sizes** | Not transferred weights or an isolated pressure effect. |

Use **GELU**, **ReLU**, **nonlinearity**, **QK**, **PV**, **FFN**, **LM head**, **BF16**, and **FP16** consistently. Introduce “feed-forward network (FFN)” and “language-model (LM) head” before relying on abbreviations. Preserve formal matrix transposes where mathematical notation requires them.

### 3.2 Notation ledger

| Symbol | Meaning and convention |
|---|---|
| `\mathcal{N}` | Set of thresholding/nonlinearity sites. |
| `\mathcal{P}` | Set of pressure sites; visually distinct from attention probability matrix `P`. |
| `a, m, h, q, k, v, z` | Site labels from Figure 1 / Table 2. Keep placement relative to LayerNorm, RoPE, nonlinearities, and projections unchanged. |
| `G_{+,\kappa}` | One-sided threshold, retaining values satisfying `x >= κ`; removes all negative values. |
| `G_{\pm,\kappa}` | Symmetric threshold, retaining values satisfying `|x| >= κ`. |
| `\kappa` | Trained threshold, fixed within a condition. |
| `\lambda` | Pressure coefficient. Do not interpret equal λ as equal effective pressure across different objectives. |
| `b` | Positive OL1 trust budget; retain its existing meaning. |
| `p` | Post-hoc calibration target, not achieved model-wide sparsity and not `P`. |
| `Z_s` | Exact-zero fraction at site s. |
| `N_0`, `N_{\mathrm{model}}` | Zero-operand numerator and declared multiplication denominator. |
| `S_{\mathrm{model}} = N_0/N_{\mathrm{model}}` | Fraction in [0,1]; a percentage is `100 S_{\mathrm{model}}`. |
| `S_{\mathrm{block}}` | Same zero numerator divided by block-only counted products. |
| `R_{\mathrm{arch}}(\mathcal{N})` | **Proposed notation change:** exactly the quantity currently written `S_{\mathrm{model}}^{\max}`. Keep the original definition and workload dependence. Author approval required. |
| `U_{\mathrm{arch}}` | Existing A7-reference normalization. In this graph, it equals `S_{\mathrm{block}}`. Prefer the simpler established block-only interpretation where practical. |
| `L_{\mathrm{val}}`, `\Delta L_{\mathrm{val}}` | Optional consistent labels for validation cross-entropy and a stated-reference difference. Do not confuse with training objectives. |
| `t_{\mathrm{native}}, t_{\mathrm{fused}}, t_{\mathrm{sparse}}` | Optional notation for latency of matched implementations at one checkpoint and workload. |

**Notation-change protocol:** Do not silently redefine an existing quantity. Record old symbol → new symbol → unchanged definition → affected source files. Update all equations, labels, captions, tables, and supplement field descriptions together. Keep original data field names and document the presentation alias when renaming would break provenance.

**Fallback if the authors decline the reach rename:** Retain `S_{\mathrm{model}}^{\max}` but explicitly call it selected-site reach at first use and remove language implying a universal or quality-feasible maximum. Do not clamp observed values or ratios to one.

### 3.3 Nonnegotiable mathematical distinctions

- `G_{+,0}` and ReLU share forward values but differ in the stated derivative at zero. `G_{\pm,0}` is the identity in both values and gradients. Preserve the original equality and derivative conventions. [S04](#s04)
- Pressure averages site–layer means equally in the existing runs. Adding sites changes the existing terms' weights. Do not describe those runs as having fixed per-site pressure. [S04](#s04)
- Each scalar multiplication receives at most one zero credit even when both activation operands are zero. Future masked positions are excluded. The dense head receives no zero credit. [S05](#s05)
- Natural zeros outside selected-site reach can contribute to observed model-wide sparsity. Reach is therefore not an upper bound on every observation. [S05](#s05)
- Sparsity at a projection input, zero fragments, and skipped tensor-core instructions are different measurements. Preserve the distinction throughout the systems section. [S11](#s11)
- A zero attention score does not imply a zero softmax probability. A zero projection input does not necessarily imply a zero output when a bias is present. Preserve the declared graph rather than assuming unrestricted zero propagation. [S05](#s05)

### 3.4 Numerical reporting

- Use one source of truth per reported number: full-precision experimental records when available, reconciled against the original tables. Do not silently replace a canonical paired delta with a difference of rounded endpoints.
- The main text can use two decimals for sparsity percentages and three or four decimals for loss when useful; complete tables may retain their existing precision. Round at display time, not before computing deltas or ratios.
- A change in a sparsity percentage is in **percentage points (pp)**, not percent. Make `100 ΔS_model` explicit when needed to avoid mixing fractions and percentage units.
- Define all loss deltas as treatment minus a named reference. Negative is better. Do not compare cross-size deltas to different A0 models as though they were absolute quality.
- Report a geometric mean as a geometric mean, with the population and weights specified. A range across checkpoints is not a confidence interval.
- For optional perplexity reporting, confirm natural-log cross-entropy and use `exp(ΔL)`. Mark ratios computed from displayed loss values as approximate. This is explanatory re-expression, not additional experimental evidence.
- A native-relative speedup is `t_native / t_sparse`. The incremental sparse-path factor is `t_fused / t_sparse`. Do not add multiplicative factors or attribute the entire native-relative factor to sparsity.
- Use raw timing records to obtain absolute latency. The PDF's speedup ratios alone do not determine milliseconds. Do not synthesize latency bars from an assumed baseline.
- Keep FP16 quality/count measurements distinct from BF16 timing/qualification measurements until precision-consistent evaluation has actually been performed.

<a id="4-agent-workflow-evidence-inventory-and-checkpoints"></a>

## 4. Agent workflow, evidence inventory, and checkpoints

### 4.1 Preparation tasks

The paths below are **proposed working outputs**, not files assumed to exist in the author's repository. Reuse an equivalent existing organization rather than imposing a new one unnecessarily.

- [ ] **PREP-01 — Inventory the source.** Locate the manuscript entry point, included section files, bibliography, figure scripts, and supplement. Record the actual paths in `revision/source_map.md`. Do not invent filenames or repository locations.
- [ ] **PREP-02 — Freeze the starting version.** Save a source-control revision or immutable copy, record the PDF hash, and compile the untouched source when available. Identify discrepancies between that compilation and the supplied PDF before editing.
- [ ] **PREP-03 — Map paragraphs.** For every `RW-*` task below, record the actual file, heading, opening anchor, and current paragraph boundary. Include text that continues across a page break. Anchors supersede page numbers after rewriting.
- [ ] **PREP-04 — Inventory evidence.** Check access to the manuscript's described training records, integer operation counts, clipping evaluations, histograms, figure coordinates, runtime reductions, protocol, and hashes. Mark each `AVAILABLE`, `MISSING`, or `UNVERIFIED`.
- [ ] **PREP-05 — Create a claim ledger.** For each proposed main-text claim, record the comparator, held-fixed factors, result, source, limitation, and whether new evidence is needed. Never cite a previous review as the experimental source.
- [ ] **PREP-06 — Create a numerical ledger.** Store exact record IDs, checkpoint hashes, precision, evaluation split, units, and display rounding. Track derived quantities separately from measured ones.
- [ ] **PREP-07 — Confirm editorial decisions.** Obtain decisions on paper direction, the reach terminology, figure budget, whether post-hoc superiority remains a headline, and the permitted experimental budget. Do not assume a current venue page limit from memory.

### 4.2 Existing-data audit with the highest potential return

- [ ] **DATA-01 — Recover per-checkpoint runtime records.** Determine whether absolute native, all-skips-off, K050, and attention-dense timings are available. Align records by checkpoint, workload, precision, and measurement protocol. Missing raw latency blocks the absolute quality–latency plot; it does not block reporting the existing ratio summaries.
- [ ] **DATA-02 — Recompute a transparent quality-budget view.** Use all eligible evaluated endpoints. As a descriptive analysis, report the greatest observed sparsity under chosen same-size A0 loss-increase budgets, for example 0.05, 0.10, and 0.20. These budgets are new presentation choices, not previously prespecified tests. Show “no eligible sparse endpoint” when appropriate. Include A0 as the reference rather than selecting a weak sparse reference.
- [ ] **DATA-03 — Audit the five historical A4/h-pressure checkpoints.** Appendix D.4 says they were excluded because pressure targeted only h. Locate them and verify initialization, token budget, optimizer, thresholds, evaluation, and identities. They may provide useful additional comparisons; they do not automatically supply the proposed fixed-weight four/seven-site design. Do not silently change the 30-checkpoint manuscript cohort to 35. [S11](#s11)
- [ ] **DATA-04 — Separate numerical and substantive frontier changes.** Use the complete clipping records and their actual p = 0 measurements. Document how tiny rerun differences are treated; do not cherry-pick strict nondominance caused by rounding or numerical drift. [S07](#s07)
- [ ] **DATA-05 — Inspect existing per-site records before requesting new evaluations.** Some h/m/q/k/v distribution information may already exist. Token-level all-zero-vector statistics cannot be reconstructed from an aggregate histogram alone. Mark unavailable diagnostics as M4 rather than inventing them.

### 4.3 Seed values to check against the evidence ledger

This table is a verification aid, not a replacement for machine-readable records. Quality gaps below are arithmetic differences of the displayed Table 4 losses and should be treated as approximate until checked against full precision.

| Item | Reported value or derived check | Source |
|---|---|---|
| 14M A0 loss | 5.2086 | Table 4, p. 19 |
| 14M A1-H-L1, λ = 1 | Loss 5.1023; sparsity 3.949% | Table 4, p. 19 |
| 14M A7-OL1, κ = 0.5 | Loss 5.8294; sparsity 27.483%; Δloss vs A0 ≈ +0.6208 | Table 4, p. 19 |
| 70M A7-OL1, κ = 0.5 | Loss 5.2159; sparsity 40.602%; A0 loss 4.0998; Δloss ≈ +1.1161 | Table 4, p. 19 |
| 410M A7-OL1, κ = 0.5 | Loss 5.1207; sparsity 80.616%; A0 loss 4.5475; Δloss ≈ +0.5732 | Table 4, p. 19 |
| A4 pressure addition, κ = 0.5 | +2.498 pp sparsity; +0.3783 loss | Table 5, p. 20 |
| A7 pressure addition, κ = 0.5 | +12.096 pp sparsity; +0.1265 loss | Table 5, p. 20 |
| A7 pressure addition, κ = 0.1 | +1.372 pp sparsity; +0.0008 loss | Table 5, p. 20 |
| Pooled FFN zeros at κ = 0.5 | A4-OL1 99.31%; A7-OL1 93.64% | Table 7, p. 21 |
| K050 vs native | Geometric mean 1.2340× across 30; retrospective search maximum 1.7830× | Section 4.5, p. 11 |
| K050 vs matched all-skips-off | 1.311× at search checkpoint; geometric mean 1.0432×; helps 14/30 checkpoints | Section 4.5 / Appendix D.4 |
| Attention-dense ablation vs native | Geometric mean 1.2506×; faster than K050 on all 30 | Table 9 / Appendix D.4 |
| All-skips-off vs native | Geometric mean 1.1829× | Table 9, p. 27 |
| Attention instruction skipping at search checkpoint | 58.0% of eligible QK and 67.2% of PV MMA instructions bypassed, but net slowdown | Appendix D.4, p. 26 |

Do not replace the reported 1.7830× search maximum with the 1.7832× final-sweep maximum without explaining the different measurements. Do not call 1.8022× the original search result; it belongs to the attention-dense ablation's range. [S11](#s11)

### 4.4 Batch execution protocol

For each task:

1. **Locate:** Read the whole original paragraph, its caption/table dependencies, and the neighboring paragraphs.
2. **Verify:** Check the claim ledger and numerical source. Mark unsupported requested content as blocked.
3. **Patch:** Make the smallest coherent edit that accomplishes the task. Do not change unrelated notation or insert new literature while editing results.
4. **Check locally:** Read the preceding paragraph, edited passage, and following paragraph continuously. Remove duplicated definitions or abrupt transitions.
5. **Check globally:** Find every occurrence of the changed headline, symbol, statistic, and figure reference. Update related claims deliberately, not through unreviewed global replacement.
6. **Save state:** Record completed task IDs, evidence used, unresolved questions, and the next task. Compile when source is available.

**Completion states:** `TODO`, `IN_PROGRESS`, `DONE_VERIFIED`, `BLOCKED_DATA`, `BLOCKED_AUTHOR`, `DEFERRED_EXPERIMENT`, `NOT_APPLICABLE`.

A task is `DONE_VERIFIED` only when its evidence and local consistency checks pass. A rewritten paragraph with an unresolved numerical placeholder is not complete.

### 4.5 Minimal state template

```yaml
mode: CURRENT_RESULTS_ONLY
source_revision: <actual source revision or immutable PDF identifier>
paper_direction: empirical_design_study
current_batch: <task IDs>
completed_verified: []
blocked:
  - task: <ID>
    reason: <missing record, author decision, or unsupported claim>
    safe_fallback: <existing-evidence wording or omit the addition>
notation_decisions:
  reach_name: <approved term>
  reach_symbol: <approved symbol>
evidence_added: []
claims_strengthened: []
claims_narrowed: []
next_tasks: []
```

Keep state concise. The full evidence ledger, not accumulated chat history, should carry provenance. When restarting, read the state, glossary, claim ledger, and current task—not a fresh speculative summary of the paper.

---

<a id="5-paragraph-by-paragraph-main-text-rewrite"></a>

## 5. Paragraph-by-paragraph main-text rewrite

**How to use this section:** The task IDs identify stable units of work. “Keep” means preserve the substance, not necessarily the exact wording. “Change” is a proposed editorial action. “Done when” is the check before marking completion. New paragraphs are explicitly labeled; optional evidence must not appear in the current-results draft.

### 5.1 Section 3.1 — Sparsification interventions

#### RW-31-01 — Explain the fixed-threshold design without claiming unmeasured savings

**Locate:** p. 3, paragraph beginning “Q-Sparse selects the top-k activation magnitudes.”

- [ ] Keep the distinction between top-k selection and a fixed cutoff that leaves the number of retained activations free to vary.
- [ ] Reduce literature detail that is already in Section 2. Make the design choice, not an implied speed advantage over prior methods, the paragraph's subject.
- [ ] State that the fixed cutoff avoids ranking and the specified input-dependent estimates, but do not translate this implementation property into an unmeasured end-to-end benefit.

**Reader takeaway:** A fixed threshold allows training to change how many activations survive; it does not enforce a sparsity budget.

**Done when:** The paragraph answers why this nonlinearity is studied without claiming superiority to top-k. **Evidence:** [S04](#s04).

#### RW-31-02 — Make the two placement decisions explicit

**Locate:** p. 3, “A sparsification intervention specifies where to set activations to zero…”

- [ ] Keep `\mathcal{N}` and `\mathcal{P}` distinct and define them before the recipe IDs.
- [ ] Add one plain-language sentence: choosing where to apply a threshold and choosing where to apply pressure are separate decisions, even though many evaluated recipes align them.
- [ ] Point to Figure 1 for assignments. Do not describe the full cohort as a fully crossed factorial design.

**Done when:** The reader can distinguish “adding Q/K/V thresholds” from “adding Q/K/V pressure” before reaching the results. **Evidence:** [S04](#s04).

#### RW-31-03 — Preserve sign treatment and boundary conventions

**Locate:** p. 3, “We use ReLU and two thresholding nonlinearities…”

- [ ] Keep both equations and the distinction between one-sided and symmetric thresholding.
- [ ] Say explicitly that the one-sided map removes all negative values, not only small-magnitude activations.
- [ ] Keep the κ = 0 forward-map interpretation; retain the Appendix A.1 pointer for the derivative-at-zero distinction.
- [ ] Avoid introducing “soft thresholding,” “shrinkage,” top-k, or a straight-through estimator: those would change the specified method.

**Done when:** A reader could implement the forward maps correctly from the paragraph and find the exact backward convention in the appendix. **Evidence:** [S04](#s04).

#### RW-31-04 — Separate a training hypothesis from what distributions demonstrate

**Locate:** p. 3, “The threshold is fixed within each training condition.”

- [ ] Keep the fixed-threshold protocol.
- [ ] Retain adaptation as a hypothesis being examined, not a guarantee of quality preservation.
- [ ] Describe the distribution analysis as evidence about the learned endpoints. Do not imply it directly measures the entire training trajectory.

**Done when:** The paragraph does not turn endpoint distributions into proof of a causal adaptation mechanism. **Evidence:** [S04](#s04), [S09](#s09).

#### RW-31-05 — Split pressure definition from optimizer implementation

**Locate:** p. 3, “Pressure uses L1: we take the mean absolute activation…”

- [ ] Split into two short paragraphs if needed. First define the pressure objective, including post-nonlinearity measurement and equal site–layer weighting.
- [ ] Add the consequence next to the definition: adding targets changes the objective and the weights of existing targets in the reported protocol.
- [ ] In the second paragraph, summarize naive L1 and OL1. Keep task-only AdamW state, conflict projection, and the trust-budget role; leave exact update geometry to Appendix A.2.
- [ ] Remove any implication that OL1 is the central novelty or uniformly improves naive L1. Do not say it guarantees task-loss preservation.

**Done when:** The reader understands what changes between the pressure variants and why equal λ across target sets does not imply equal per-site pressure. **Evidence:** [S04](#s04), [S08](#s08).

### 5.2 Section 3.2 — Sparsity and architectural reach

#### RW-32-01 — Define exactly what the model-wide metric counts

**Locate:** p. 4, “Activation sparsity is the fraction of exactly zero entries at a named site.”

- [ ] Keep the local definition, then give `S_model = N0 / N_model`.
- [ ] Explain the atomic unit in one accessible sentence: one scalar multiplication in a declared matrix product receives credit if an activation operand is zero.
- [ ] Keep the dense LM head in the denominator with no zero credit, count-pooling, and causally valid full-sequence scope.
- [ ] End by distinguishing this quantity from total FLOPs, memory traffic, and elapsed time.

**Done when:** A reader cannot mistake 27% model-wide sparsity for a measured 27% runtime reduction. **Evidence:** [S05](#s05).

#### RW-32-02 — Repair the “ceiling” interpretation, not the measurement

**Locate:** p. 4, “The model-wide sparsity ceiling…”

- [ ] Apply the approved reach terminology and symbol consistently.
- [ ] Define the quantity as the fraction structurally reached when the selected sites are entirely zero, with overlap counted once.
- [ ] State that it depends on architecture, site set, and sequence length, but not a checkpoint or a quality constraint.
- [ ] Retain the natural-zero caveat: observed sparsity can include zeros outside selected-site reach.
- [ ] Do not change the numerator, normalize every recipe by a new denominator, or reinterpret closeness to reach as practical optimality.

**Reader takeaway:** The quantity explains the scope of the intervention in the declared workload; it does not say how much sparsity can be achieved without unacceptable loss.

**Done when:** Abstract, introduction, equations, figure labels, and appendices use the same interpretation. **Evidence:** [S05](#s05).

### 5.3 Section 4 opening — Experimental protocol and comparison map

#### RW-40-01 — Establish the common evaluation and the actual scale of the study

**Locate:** p. 4, “We pretrain randomly initialized Pythia-14M, 70M, and 410M models…”

- [ ] Keep random initialization, MiniPile, and the detailed-14M / selected-larger-recipes distinction.
- [ ] State validation cross-entropy as the quality measure and explain “lower is better” once.
- [ ] Keep the complete validation-block count, sequence length, and shared loss/count pass; move packing detail to the appendix only if needed for flow, not to conceal the evaluation scope.
- [ ] Make clear that recipe rows identify separately pretrained models.

**Done when:** The reader does not infer fine-tuning of released Pythia weights or a large-model intervention sweep. **Evidence:** [S06](#s06).

#### RW-40-02 — Turn matching claims into an explicit comparison map

**Locate:** p. 4, “Within each size, matched contrasts share initialization, data order, and training budget.”

- [ ] Keep those matching properties.
- [ ] Distinguish three cases: pressure on/off at fixed nonlinearities; unpressured A7 versus A4 at fixed κ; A7-OL1 versus A4-OL1 as a complete-recipe comparison.
- [ ] State that the last case changes both target placement and objective normalization.
- [ ] Qualify A4 versus A1-H: beyond the κ = 0 boundary details, this changes extra sites and the h threshold, not only the number of sites.
- [ ] Add a compact comparison map in text or a table if the figure alone is insufficient. Avoid repeating all 29 deltas here.

**Done when:** Every later use of “effect of placement” can be traced to a contrast that actually holds the relevant factors fixed. **Evidence:** [S06](#s06), [S04](#s04).

#### RW-40-03 — Bring the single-seed and budget limitations into the setup

**Locate:** p. 4, “The cohort contains 30 conditions at 14M and 12 at each larger size.”

- [ ] Keep cohort sizes, update/token budget, final-checkpoint selection, and swept λ/κ values.
- [ ] Add a short sentence stating one training seed per condition. Shared initialization is a paired design feature, not independent replication.
- [ ] Keep the larger-size learning-rate difference and tokens-per-parameter details in Table 3 / Section 4.4; cross-reference them here as needed.
- [ ] Do not call the 29 comparisons 29 independent trials.

**Done when:** The reader knows the unit of replication before interpreting small differences. **Evidence:** [S06](#s06).

#### RW-40-04 — Define the post-hoc control narrowly and transparently

**Locate:** p. 4, “For the evaluation-only reference, we apply uniform magnitude-clipping targets…”

- [ ] Keep the four sites, separate site–layer calibration, ten training blocks, and fixed weights.
- [ ] Use “uniform post-hoc magnitude clipping” as the main label. Keep TEAL-style provenance without implying a full optimized-TEAL reproduction.
- [ ] Distinguish calibration target p from achieved sparsity.
- [ ] State that comparing this four-site reference to A7 also changes reach and, depending on the contrast, the final transformation at h.

**Done when:** A reviewer can identify exactly what this baseline does and what superiority over it would not establish. **Evidence:** [S06](#s06), [S07](#s07).

### 5.4 Section 4.1 — Quality–sparsity trade-offs

#### RW-41-01 — Lead with the low-loss operating regime, not only high sparsity

**Locate:** p. 4, “Local pressure and broader nonlinearities offer different quality–sparsity trade-offs…”

- [ ] Keep the observation that local pressure can lower validation loss while introducing modest model-wide sparsity in the reported runs.
- [ ] Mention the strongest observed naive-L1 local endpoint, not only the selected OL1 endpoint: Table 4 gives loss 5.1023 and sparsity 3.949% at λ = 1.
- [ ] The existing OL1 versus A0-clipping contrast can remain as an illustration of different quality at similar, not identical, sparsity. Do not call it an exactly matched-quality or matched-sparsity experiment.
- [ ] Avoid inferring a statistically established regularization benefit until replicated.

**Reader takeaway:** The low-quality-cost regime and maximum-sparsity regime answer different design questions.

**Done when:** Readers do not come away believing that the main empirical benefit is unique to OL1. **Evidence:** [S07](#s07), [S08](#s08).

#### RW-41-02 — Attach a visible quality cost to the aggressive operating point

**Locate:** pp. 4 and 6, “Broader interventions reach much greater sparsity.” Include the continuation beginning “the internal attention operands.”

- [ ] Keep the 14M A7-OL1 κ = 0.5 result, but report its loss increase relative to same-size A0 alongside 27.483% sparsity.
- [ ] Keep the four-site versus seven-site reach distinction next to the clipping comparison.
- [ ] Replace broad “dominates the frontier” language with a statement about the tested comparator and region, only after checking the actual endpoint/frontier records.
- [ ] End with the unresolved design question: how much of the increased opportunity is available within a specified loss budget?

**Done when:** The reader sees that “more sparsity,” “better than another sparse recipe,” and “better quality–efficiency than A0” are separate statements. **Evidence:** [S07](#s07), [S05](#s05).

#### RW-41-03 — Add a quality-budget summary only from available records

**Locate:** New paragraph or compact table immediately after RW-41-02.

- [ ] Insert the DATA-02 summary: greatest observed sparsity within stated same-size A0 loss budgets, with recipe and κ/λ.
- [ ] Include a note that the budgets describe a retrospective summary of evaluated points; they were not the original selection criterion.
- [ ] Preserve unfavorable or empty regimes. Never interpolate an untrained model between endpoints.
- [ ] Add absolute latency only when DATA-01 provides valid measurements. If only speedup ratios exist, keep the latency addition blocked rather than relabeling ratios as quality–latency data.

**Done when:** A reader can identify a feasible tested operating point without treating all sparsity increases as useful. **Dependency:** DATA-01/02 as applicable. **Evidence:** [S07](#s07).

### 5.5 Section 4.2 — Paired intervention effects

#### RW-42-01 — Explain the local comparison without an optimizer-superiority story

**Locate:** p. 6, “Pressure's additional value depends on the threshold and target set…”

- [ ] Keep the ReLU-versus-GELU contrast and the local-pressure comparisons.
- [ ] Report naive-L1 versus OL1 as update-rule comparisons, not as an isolated test of conflict projection.
- [ ] Retain the reversal at λ = 1 and explain that small differences at other weights are single-seed endpoint differences.
- [ ] Use the complete Table 5 deltas rather than recomputing from rounded Table 4 entries.

**Done when:** The paragraph supports conditional behavior, not a universal ranking of optimizers. **Evidence:** [S08](#s08).

#### RW-42-02 — Make the threshold dependence quantitatively memorable

**Locate:** pp. 6–7, “For A4, adding OL1 improves both loss and sparsity at κ = 0 and 0.01.” Include its continuation after Figure 3.

- [ ] Keep one low/moderate-threshold comparison and one high-threshold comparison. Reduce the recital of sweep values that are already visible in the figure.
- [ ] Pair every sparsity increment with its loss change and the exact pressure-on versus pressure-off comparator.
- [ ] Preserve the different A4 and A7 outcomes at κ = 0.5, but immediately identify their different pressure objectives.
- [ ] Treat +0.0008 loss as a reported difference, not evidence of equivalence or loss preservation.

**Reader takeaway:** The value of pressure is conditional on the base recipe and operating point; its cost is part of the result.

**Done when:** The reader can explain one meaningful change of behavior without memorizing every point. **Evidence:** [S08](#s08).

#### RW-42-03 — Replace broad isolation language with an exact scope statement

**Locate:** p. 7, “Pressure therefore needs to be evaluated at its actual threshold and sites.”

- [ ] Keep the existing normalization caveat; it is essential, not expendable detail.
- [ ] State what is learned now: pressure on/off effects within each defined objective vary across thresholds.
- [ ] State what is not learned now: the larger A7 pressure increment cannot be attributed solely to broader nonlinearity placement or solely to pressure at q/k/v.
- [ ] Transition to distributions as a description of complete learned recipes, not a mechanism-identifying test.

**Done when:** This paragraph matches the comparison map in RW-40-02. **Optional extension:** M1 adds a separately labeled fixed-weight comparison; it does not retroactively change the meaning of the original runs. **Evidence:** [S08](#s08).

### 5.6 Section 4.3 — Reshaping activation distributions

#### RW-43-01 — Make the zero mass visible and name the aggregation

**Locate:** p. 7, “More FFN zeros need not mean more model-wide sparsity…”

- [ ] Keep the 99.31% versus 93.64% pooled-FFN comparison and the corresponding attention distinction.
- [ ] Use “pooled FFN elements at h and m” rather than suggesting the percentage describes every site or layer.
- [ ] Put the zero masses directly in the revised figure; retain Table 7 for complete values and tails.
- [ ] Explain once that a sharp nonzero peak near zero is not exact-zero mass.

**Done when:** The central sparsity information is not absent from the main distribution figure. **Evidence:** [S09](#s09).

#### RW-43-02 — Describe nonlocal changes without claiming a mechanism

**Locate:** pp. 7–8, “A4 does not directly threshold or pressure q, k, v…”

- [ ] Keep the observation that distributions at untargeted q/k/v sites change under the complete A4-OL1 recipe.
- [ ] Prefer a direct description over the undefined term “spillover.” If that term remains, define it descriptively and keep the causal caveat.
- [ ] Keep the distinction between indirect distribution changes in A4 and direct intervention in A7.
- [ ] Do not infer that pressure alone caused the change relative to A0.

**Done when:** A reader can separate the measurement from a proposed explanatory mechanism. **Evidence:** [S09](#s09).

#### RW-43-03 — Explain the ranking reversal using operation counts

**Locate:** p. 8, “The operation counts show why this distinction matters model-wide.”

- [ ] Keep the QK/PV contribution comparison and the fact that some projection contributions fall under A7-OL1.
- [ ] Prefer one net explanation over a long list: the added attention contribution outweighs the reduced projection contributions in the total.
- [ ] Reference a compact operation breakdown in the main text or main figure, derived from Table 8 / Figure 7.
- [ ] End with the bounded implication: local FFN sparsity is insufficient to rank these recipes by counted model-wide opportunity. Do not imply it ranks their latency.

**Done when:** The reader can reconstruct why the local and aggregate metrics disagree. **Evidence:** [S09](#s09).

#### RW-43-04 — Optional insertion: investigate extreme branch-input sparsity

**Locate:** New paragraph after RW-43-03, only after verified DATA-05/M4 results.

- [ ] Report h and z separately, with per-token and per-layer nonzero counts or all-zero-vector rates when actually available.
- [ ] Relate pooled statistics to their aggregation weights; do not substitute pooled h/m percentages for h alone.
- [ ] Distinguish an all-zero projection input from the entire residual branch's output, since biases may remain.
- [ ] Discuss context sensitivity or branch-function tests only when performed. Do not diagnose “collapse” from rounded product contributions alone.

**Safe current-results fallback:** No new functional claim. Retain the pooled-distribution limitations and place the diagnostic question in the discussion or revision notes. **Dependency:** M4.

### 5.7 Section 4.4 — Transfer across model sizes

#### RW-44-01 — Define exactly which ordering persists

**Locate:** p. 8, “The high-threshold A7-OL1 advantage persists at all three model sizes…”

- [ ] Keep the same-κ A7-OL1 versus A4-OL1 comparison at κ = 0.5 and the smaller-threshold reversals.
- [ ] Use “the ordering of these complete recipes recurs” rather than a broad statement that “the method transfers.”
- [ ] Make clear that improving both axes relative to A4-OL1 does not imply improving either axis relative to A0.
- [ ] Consider renaming the subsection **“Complete-recipe comparisons across model sizes”** to match its actual evidence.

**Done when:** The reader knows the comparator and boundary threshold without consulting a footnote. **Evidence:** [S10](#s10).

#### RW-44-02 — Turn normalization into an explanatory result

**Locate:** p. 8, “Raw sparsity must also be read against the changing workload.”

- [ ] Keep the changing dense-head share and the reach values, using the approved terminology.
- [ ] Explain the common A7-reference denominator. Do not normalize A4 points by A4 reach in one panel and by A7 reach in another without explicit labeling.
- [ ] Prefer identifying the normalized axis as block-only sparsity, since the current graph gives `U_arch = S_block`.
- [ ] State that this removes the dense-head share, not differences in the relative operation mix inside blocks.

**Reader takeaway:** A change in the denominator can make cross-size sparsity percentages look more different than the block-level counts alone.

**Done when:** Figure 5's axis and its explanation use exactly the same quantity. **Evidence:** [S10](#s10), [S05](#s05).

#### RW-44-03 — Keep the training-budget limitation adjacent to the scale claim

**Locate:** p. 8, “This is transfer of a complete-recipe comparison.”

- [ ] Keep the absence of no-pressure A4/A7 controls at larger sizes.
- [ ] Keep the common token budget, unequal tokens per parameter, lower 410M learning rate, and the poorer 410M A0 loss relative to 70M under this protocol.
- [ ] Describe longer training as an untested explanation/check, not as a proven correction for the results.
- [ ] Do not call the observed curves a quality scaling law or conclude that sparsification becomes intrinsically more beneficial with scale.

**Done when:** The subsection's last paragraph narrows only what the experiment does not test, without retreating from the actual same-size comparisons. **Evidence:** [S10](#s10), [S06](#s06).

### 5.8 Section 4.5 — Specialized inference kernels

#### RW-45-01 — Present compatibility testing without attacking the upstream method

**Locate:** p. 11, “Realizing activation sparsity requires kernels matched to the model's shapes…”

- [ ] Keep the unchanged primitive's numerical qualification failure and P0's limited qualified subset.
- [ ] State that these are compatibility results for the tested implementation and shape, not proof that TwELL is generally inaccurate or slower.
- [ ] Do not compare P0's four-checkpoint geometric mean with K050's 30-checkpoint mean as a fair aggregate ranking.
- [ ] Shorten this opening if it delays the central execution finding; leave primitive errors and adaptation details in Appendix D.4.

**Done when:** The section motivates specialization without claiming an unsupported victory over prior work. **Evidence:** [S11](#s11).

#### RW-45-02 — State the workload and development scope precisely

**Locate:** p. 11, “To make this specialization practical within an architecture study…”

- [ ] Keep the human-guided coding-agent loop and retrospective evaluation as implementation-development details.
- [ ] Keep BF16, batch one, uncached 2,048-token inference, full vocabulary logits, one RTX5090, and numerical qualification.
- [ ] Avoid claiming a controlled evaluation of agent productivity or a verified served model for every request; the appendix describes one development trajectory and a default configuration.
- [ ] Move detailed search chronology to the appendix if necessary to give attribution and operation-level findings more main-text space.

**Done when:** The reader can identify both the inference workload and the narrower role of the coding-agent case study. **Evidence:** [S11](#s11).

#### RW-45-03 — Separate search progress from the scientific runtime comparison

**Locate:** p. 11, “The matched retrospective reaches 1.6024× by iteration 11…”

- [ ] Retain search progress only as development context; do not make it the section's main contribution.
- [ ] Report the selected implementation's 30-checkpoint qualification and geometric-mean speedup with its native reference.
- [ ] Replace “predicts realized speedup” elsewhere with “is associated with native-relative speedup across the evaluated checkpoints.” Keep the fit descriptive, not causal or out-of-sample predictive.
- [ ] State that topology, weights, quality, and implementation overhead differ across checkpoints. Keep the FP16-count/BF16-timing mismatch visible until resolved.

**Done when:** The fit is neither stronger than the design nor allowed to substitute for a matched sparse-path ablation. **Evidence:** [S11](#s11).

#### RW-45-04 — Promote the fusion and sparse-path decomposition

**Locate:** p. 11, “Ablations distinguish fusion from the contribution of sparse paths.”

- [ ] Split this into two paragraphs if needed. The first reports native versus fused versus sparse execution, including the 1.311× search-checkpoint and 1.043× cohort incremental factors and 14/30 count.
- [ ] The second reports attention skipping: the attention-dense ablation improves every checkpoint; substantial eligible MMA skipping still fails to repay overhead in the measured setting.
- [ ] Explain the granularity plainly: an instruction remains active unless the required zero-fragment condition is met; detection and control have a cost.
- [ ] Keep “either operand fragment entirely zero” correct. Do not say both Q and K fragments must be zero.
- [ ] End with the specific implication: scalar zero opportunity is not sufficient to select a profitable execution path.

**Done when:** A reviewer can attribute the native-relative speedup and understand the negative attention result without reading the appendix. **Evidence:** [S11](#s11).

#### RW-45-05 — Optional existing-data reanalysis of the sparsity/runtime association

**Locate:** Replace or supplement the Figure 6 association paragraph, only when records support it.

- [ ] Analyze incremental sparse-path speedup relative to all-skips-off, not only native-relative speedup.
- [ ] Use BF16 counts if M4 has produced them. Otherwise label the existing FP16/BF16 comparison explicitly; do not pretend the mismatch is resolved.
- [ ] Show sensitivity to the searched checkpoint and recipe families. With only 30 checkpoints, prefer simple, interpretable descriptive checks over a heavily fitted predictor.
- [ ] Treat a weaker association as an informative result, not something to hide or repair by selecting points.
- [ ] Distinguish retrospective sensitivity checks from genuinely held-out prospective validation.

**Safe fallback:** Keep the existing descriptive fit with its limitations; give attribution the stronger narrative role. **Dependency:** DATA-01; M4 for precision alignment.

### 5.9 Section 5 — Discussion and conclusion

#### RW-50-01 — State the decision lesson, not a broad winning recipe

**Locate:** p. 13, “Pressure, thresholds and site placement need to be evaluated together.”

- [ ] Keep the conditional pressure result and the local/global ranking distinction.
- [ ] Explain the practical experimental implication: evaluate the marginal addition against the actual base recipe and inspect both quality and affected operations.
- [ ] Do not repeat every maximum or imply that the seven-site recipe is the best choice under all quality budgets.

**Done when:** The paragraph offers a transferable way to evaluate an intervention while remaining tied to the experiment. **Evidence:** [S12](#s12), [S08](#s08).

#### RW-50-02 — Replace self-evaluation with an unresolved but useful distinction

**Locate:** p. 13, “The interpretation is supported by a connected pattern across experiments.”

- [ ] Remove self-assessment such as the statement that the pattern makes the work “more informative.” Show the connection instead.
- [ ] Use this space to distinguish three decisions: which recipe improves on its paired alternative, which endpoint fits a quality budget, and which execution path is profitable.
- [ ] State that the current results answer parts of those questions differently. A high-sparsity endpoint is not automatically a good model-selection endpoint.
- [ ] Do not add an untested claim of conditional computation or a collapse diagnosis.

**Done when:** The paragraph adds an idea the reader did not already get from the results summary. **Evidence:** [S07](#s07), [S10](#s10), [S11](#s11).

#### RW-50-03 — Explain why the accounting and kernel findings belong in one paper

**Locate:** p. 13, “Product accounting and kernel timing answer different questions.”

- [ ] Keep this strong organizing distinction.
- [ ] Connect selected-site reach → measured zero products → actual skipped instructions → elapsed time, without implying equality between them.
- [ ] Use the attention ablation as the concrete counterexample. Avoid claiming an unmeasured break-even threshold.
- [ ] End with a measurement principle: report the controls needed to determine where acceleration came from.

**Done when:** The architecture and systems sections read as one investigation rather than two loosely joined contributions. **Evidence:** [S05](#s05), [S11](#s11).

#### RW-50-04 — Make limitations specific, proportionate, and actionable

**Locate:** p. 13, “These conclusions concern MiniPile pretraining at a fixed token budget…”

- [ ] Preserve the existing scope: dataset/budget, learning-rate difference, complete-recipe larger-model results, 14M diagnostics, one GPU, and full-sequence workload.
- [ ] Add the single-seed limitation and the restricted post-hoc baseline if still applicable. Mention the precision mismatch if it remains unresolved.
- [ ] State at most two high-value open questions in the conclusion, such as whether fixed-weight placement effects replicate and whether the same execution trade-off holds at larger widths.
- [ ] Put the full proposed-experiment inventory in revision notes, not a final paragraph that promises a different paper.

**Done when:** Every limitation is attached to a real inference boundary, not a generic apology. **Evidence:** [S06](#s06), [S11](#s11), [S12](#s12).

### 5.10 Section 2 — Related work, edited after the results story stabilizes

#### RW-21-01 — Post-hoc thresholding paragraph

**Locate:** p. 2, “Post-hoc thresholding removes activations that a trained model can tolerate losing.”

- [ ] Preserve the manuscript's existing CATS/TEAL distinctions and citations.
- [ ] Keep the explicit difference between allocation-aware prior work and the uniform control used here.
- [ ] End with the study's actual question, not a claim to supersede optimized post-hoc methods.

**Done when:** The reader understands both provenance and the comparator's limits. **Evidence:** [S03](#s03).

#### RW-21-02 — Training-time pressure paragraph

**Locate:** p. 2, “Training-time interventions allow the representation to adapt to sparse computation.”

- [ ] Preserve the supported distinctions among ReLU restoration, progressive pressure, threshold shifts, and the existing training studies.
- [ ] Reduce name-by-name recital where possible; organize around the design dimensions the current experiments examine.
- [ ] End with the marginal-pressure and operation-weighting questions. Do not imply OL1 or activation pressure itself is a newly invented principle.

**Done when:** The paragraph identifies the study's contribution without inventing a novelty gap. **Evidence:** [S03](#s03).

#### RW-21-03 — Beyond-FFN placement paragraph

**Locate:** pp. 2–3, “Extending sparsification beyond FFNs makes the choice of sites explicit.”

- [ ] Preserve the distinction between projection-input sparsity, feature-coordinate sparsity inside attention, and token-position selection.
- [ ] Explain the current seven sites in those terms without pretending all forms of “attention sparsity” are interchangeable.
- [ ] Use only claims supported by the existing cited discussion. Any new literature claim requires checking its primary source before insertion.

**Done when:** The reader knows which type of attention work this paper targets. **Evidence:** [S03](#s03).

#### RW-22-01 — Workload-specific execution paragraph

**Locate:** p. 3, “Converting activation sparsity into speedup requires an execution strategy suited to the workload.”

- [ ] Preserve the distinction between autoregressive decoding and full-sequence/batched computation.
- [ ] Keep the explanation that saved work must repay representation and execution overhead.
- [ ] Do not place prior published speedup numbers beside this paper's maximum as though their workloads and baselines were comparable.

**Done when:** Runtime comparisons are framed by workload, not by the largest isolated multiplier. **Evidence:** [S03](#s03).

#### RW-22-02 — Shape, structure, and development tools paragraph

**Locate:** p. 3, “The execution strategy also depends on shape and sparsity structure.”

- [ ] Keep the role of mask structure and tiled computation.
- [ ] Keep agent-assisted development as a tool for the case study, not an independently validated scientific contribution here.
- [ ] Remove repeated implementation details that now belong in Section 4.5 / Appendix D.4.

**Done when:** The paragraph motivates why the measured negative attention result is informative rather than anomalous. **Evidence:** [S03](#s03), [S11](#s11).

### 5.11 Section 1 — Introduction, rewritten near the end

#### RW-10-01 — Opening problem paragraph

**Locate:** p. 1, “Activation sparsity can improve language-model inference efficiency…”

- [ ] Keep the practical question about useful sparsity at acceptable quality cost.
- [ ] Introduce the mismatch between activation zeros, affected multiplication work, and runtime as the central motivation.
- [ ] Avoid a generic efficiency preamble and unqualified promises that zeros improve memory or latency in every workload.

**Done when:** The first paragraph poses the exact question the results can answer. **Evidence:** [S02](#s02).

#### RW-10-02 — Existing-methods and gap paragraph

**Locate:** p. 1, “Existing approaches induce activation sparsity in different ways.”

- [ ] Keep a compact account of pressure, thresholding, and placement in prior recipes.
- [ ] Replace a sweeping claim that effects “have not been isolated” with the narrower motivation for this matched study, unless a verified literature review justifies the stronger statement.
- [ ] Distinguish intervention interaction from method leaderboard comparison.

**Done when:** The novelty claim does not depend on ignoring existing ablations or on reproducing methods not actually reproduced here. **Evidence:** [S02](#s02), [S03](#s03).

#### RW-10-03 — Definitions and study-design paragraph

**Locate:** pp. 1–2, “In this work, we study three sparsification interventions…”

- [ ] Define pressure as encouraging small magnitudes, thresholding as the specified zeroing maps, and placement as the selected sites.
- [ ] Replace “isolates the individual and interaction effects of each intervention” with the exact scope of the matched contrasts.
- [ ] Introduce model-wide sparsity and architectural reach in plain language; reserve equations and full exclusions for Section 3.2.
- [ ] Do not describe reach as the maximum achievable sparsity at useful quality.

**Done when:** This paragraph matches RW-40-02 and the approved notation without exaggerating experimental identification. **Evidence:** [S02](#s02), [S05](#s05), [S06](#s06).

#### RW-10-04 — Empirical findings paragraph

**Locate:** p. 2, “We pretrain Pythia-family models from scratch at multiple scales…”

- [ ] Lead with the two main findings rather than a list of all three maximum sparsity percentages.
- [ ] Include at most one or two quantitative contrasts that the reader can interpret with the named reference and quality cost.
- [ ] Keep the larger-size result as a selected complete-recipe extension and retain its threshold-dependent ordering.
- [ ] Narrow the clipping claim to the uniform comparator and verified region; omit a global superiority statement if it cannot be supported concisely.

**Done when:** The introduction makes the current evidence sound as informative as it is, not stronger than it is. **Evidence:** [S07](#s07), [S08](#s08), [S09](#s09), [S10](#s10).

#### RW-10-05 — Systems findings paragraph

**Locate:** p. 2, “Finally, we test whether specialized kernels can translate these sparsity gains into faster inference…”

- [ ] State the full-sequence 14M workload and the native-relative nature of any headline multiplier.
- [ ] Bring the matched-fusion sparse-path comparison and negative attention ablation into the paragraph.
- [ ] Downgrade “predicts realized speedup” to the supported descriptive association or remove the fit from the introduction to make space for attribution.
- [ ] Remove any implication that the study compares coding agents or verifies the served model of every development request.

**Done when:** A reader who stops after the introduction understands what generated the speedup and what remains untested. **Evidence:** [S02](#s02), [S11](#s11).

### 5.12 Abstract and title — Last substantive rewrite

#### RW-ABS-01 — Rewrite the abstract as a compact evidence-based argument

**Locate:** p. 1, the complete abstract.

Use the following sentence functions; this is a structure, not mandatory wording:

1. **Question:** Why pressure, thresholding, and placement need to be considered together when seeking useful activation sparsity.
2. **Design and scope:** Matched 14M study, with selected larger-model complete recipes and a scoped execution case study.
3. **First result:** The marginal quality–sparsity effect of pressure varies with the base intervention and threshold.
4. **Second result:** Operation accounting explains why local FFN sparsity can mis-rank model-wide opportunity.
5. **Execution result:** Native-relative speedup has a matched-fusion component; sparse paths contribute conditionally, and attention skipping is slower in the measured workload.
6. **Bounded implication:** Evaluate the intervention, affected operations, quality cost, and execution granularity jointly.

- [ ] Include one interpretable quantitative comparison, not an unqualified string of maxima.
- [ ] If reporting maximum sparsity, attach a same-size A0 loss cost. If reporting 1.78×, identify native execution and avoid attributing the whole factor to sparsity.
- [ ] Remove broad dominance, optimality, prediction, and complete-factor-isolation claims unless new verified evidence supports them.
- [ ] Replace “sparsity ceiling” according to the approved terminology decision.
- [ ] Correct the original grammar errors (“interventions reaches,” “recipes … dominates,” “accross”).
- [ ] Use roughly 170–220 words as a drafting target only; honor the actual author/venue constraint after checking it.

**Done when:** Every abstract claim points to a specific result paragraph, and every decisive caveat needed to interpret a headline is visible here or built into its wording. **Evidence:** [S01](#s01) plus the verified main-result ledger.

#### RW-TITLE-01 — Preserve the title unless the final scope makes a change useful

**Locate:** p. 1, title.

- [ ] Default: keep the current title. It already names the actual study dimensions and does not promise a universally faster model.
- [ ] Consider a subtitle only after the argument is stable; do not add “optimal,” “general,” “scaling law,” or “quality-preserving.”
- [ ] Keep terminology identical to the abstract and Section 3.

**Done when:** The title describes the paper the experiments support, not a larger project planned for later.

<a id="6-figure-and-table-surgery"></a>

## 6. Figure and table surgery

### 6.1 Rules for every visual

- [ ] **VIS-00A:** Give each main visual one question to answer. The caption should state its takeaway, what is plotted, the comparison/normalization, and the one limitation needed to interpret it.
- [ ] **VIS-00B:** Regenerate from verified data and source scripts. Do not redraw coordinates from a screenshot when numerical records are available. Do not fabricate error bars or unmeasured operating points.
- [ ] **VIS-00C:** Use consistent recipe colors, marker shapes, names, units, and line conventions across figures. Ensure recipe distinctions remain readable without color alone.
- [ ] **VIS-00D:** Check the compiled figure at normal reading size. Axis labels and zero-mass annotations must be readable without zooming to poster size.
- [ ] **VIS-00E:** Distinguish a line joining evaluated settings from a fitted curve, a Pareto envelope, and a training trajectory. Do not use “frontier” in a title unless a frontier is actually computed and identified.
- [ ] **VIS-00F:** Keep data exclusions, restricted axes, zero omissions, and unavailable measurements visible. Complete appendix views should remain accessible.

### 6.2 Figure-specific tasks

#### FIG-01 — Figure 1: architecture and recipe matrix, p. 5

- [ ] Retain the graph's parallel branches, post-RoPE q/k sites, z before the output projection, and all seven labels.
- [ ] Replace “sparsification ladder” / “Step” with **recipe matrix** / **recipe**, if approved, to avoid suggesting curriculum stages.
- [ ] Make pressure sites explicit. In the current eight rows they follow the selected sites when pressure is enabled; any added historical h-only or crossed control needs a distinct assignment.
- [ ] Use the approved reach terminology in the three rightmost columns and caption. Preserve the existing calculation and T = 2,048.
- [ ] Standardize GELU / ReLU / nonlinearity spelling and explain that h replaces GELU while other selected maps act at the specified sites.

**Acceptance check:** A reader can reproduce each row's thresholding and pressure placement without inferring assignments from the recipe name. **Evidence:** [S04](#s04), [S05](#s05).

#### FIG-02 — Figure 2: quality–sparsity endpoints, p. 6

- [ ] Retain all 30 training endpoints and the stated A0 clipping reference; keep the full-range appendix view.
- [ ] Remove “frontier” from the plot title unless an explicit nondominated set is added and defined.
- [ ] Make A0 and the low-loss local-pressure regime visible. Label a small number of meaningful endpoints, not every sweep value.
- [ ] Add Δloss versus A0 in an annotation, inset, or companion table rather than making the reader infer the high-sparsity cost from a compressed axis.
- [ ] Identify the four-site versus seven-site reach mismatch in the caption when discussing post-hoc comparisons.

**Acceptance check:** The visual communicates a trade-off, not a universal trained-method victory. **Evidence:** [S07](#s07).

#### FIG-03 — Figure 3: paired effects, p. 7

- [ ] Preserve the treatment-minus-reference convention, paired recipe labels, sweep ordering, and units.
- [ ] In the caption, distinguish pressure additions within a fixed recipe from comparisons that change multiple aspects of the intervention.
- [ ] Keep the shared-control / one-seed limitation near the figure or in the directly adjacent setup; do not imply 29 independent estimates.
- [ ] Add replicate points or uncertainty only after M2. Prefer showing individual paired effects over hiding them behind a single aggregate bar.
- [ ] Display any new fixed-weight M1 design separately from the original normalized-objective sweep.

**Acceptance check:** A reader can tell exactly what each delta means and what uncertainty is available. **Evidence:** [S08](#s08).

#### FIG-04 — Figure 4: activation distributions, p. 9

- [ ] Add zero-mass annotations to every plotted recipe/group panel, using the verified counts from Table 7.
- [ ] Retain h:m weighting, q:k:v pooling, post-RoPE measurement, tail accounting, and non-renormalized histogram convention.
- [ ] Explain symlog once; do not invite readers to read the plotted area as probability mass.
- [ ] Reduce visual crowding rather than deleting the zero masses or hiding the pooling definition.
- [ ] Add separate h/z diagnostics only if available and readable; otherwise keep those in a new appendix panel.

**Acceptance check:** The main figure contains the probability mass most relevant to its sparsity claim. **Evidence:** [S09](#s09).

#### FIG-05 — Figure 5: cross-size comparison, p. 10

- [ ] Rename reach guides and the normalized axis consistently with Section 3.2.
- [ ] Prefer **block-only sparsity** for the bottom axis after verifying the existing identity; preserve the common denominator for every curve within a size.
- [ ] Keep same-size A0 references and distinguish the high-threshold pairwise finding from the full curve ordering.
- [ ] State that model sizes share a token budget, not tokens per parameter, and that these are complete-recipe comparisons.
- [ ] Do not hide the high loss of some 410M settings to make the cross-size result look smoother.

**Acceptance check:** A reader does not interpret increased normalized or raw sparsity as proof of better quality scaling. **Evidence:** [S10](#s10).

#### FIG-06 — Figure 6: prioritize execution attribution over search chronology, p. 12

- [ ] Recommended: move the search-progress panel to the appendix and use the main space for matched-fusion versus sparse execution and/or the operation-level negative result.
- [ ] Preserve the original search sequence and its retrospective status wherever it appears. Do not substitute the later faster ablation into the historical incumbent trace.
- [ ] If retaining the association scatter, label the exact reference, precision, selected checkpoint, and descriptive nature of the fit.
- [ ] Use incremental sparse-path ratios for the stronger attribution question when available. Add absolute latency only from actual records.
- [ ] Identify the attention-dense ablation as faster than the selected K050 on this cohort; “best” must name the search set or comparison set.

**Acceptance check:** The main visual answers “what produced the acceleration?” rather than “how large was the best observed multiplier?”. **Evidence:** [S11](#s11).

#### FIG-07 — Promote the operation explanation, currently Figure 7 on p. 22

- [ ] Consider moving a compact 14M operation breakdown into the main distribution/results discussion; retain the full three-size version in the appendix if space is tight.
- [ ] Use the common model denominator and include the dense head's role in the caption even though it receives no zero contribution.
- [ ] Make the attention contribution and the offsetting projection changes visually identifiable.
- [ ] Do not label bar heights as time or FLOPs saved.

**Acceptance check:** The reader can see the source of the local-versus-model-wide ranking reversal rather than having to trust a sentence about it. **Evidence:** [S09](#s09).

#### FIG-08 — Figures 8–10: retain the complete post-hoc record, pp. 23–25

- [ ] Keep all evaluated settings, ties, dominated points, and full loss ranges.
- [ ] Update recipe/metric terminology and references after main-figure changes.
- [ ] Preserve the distinction between actually measured p = 0 evaluations and canonical training endpoints.
- [ ] Do not present connecting paths as attainable interpolation or as training trajectories.

**Acceptance check:** The main-text view is selective for readability, not selective in the evidence it discloses. **Evidence:** [S07](#s07).

### 6.3 Table-specific tasks

- [ ] **TAB-01 — Table 1 / Table 6:** Retain complete-recipe wording, treatment-minus-reference sign, same-κ matching, and all cross-size reversals. Do not relabel these as isolated placement effects.
- [ ] **TAB-02 — Table 2:** Keep exact site shapes and consuming operations. Check against the implementation before any diagram simplification.
- [ ] **TAB-03 — Table 3:** Preserve all size-dependent settings and tokens-per-parameter differences. Add actual compute cost only from logs, not an estimate disguised as a measurement.
- [ ] **TAB-04 — Table 4:** Preserve all 54 endpoints. Apply notation aliases consistently, retain rounded-zero caveats, and consider a supplementary Δloss-vs-A0 column generated from full precision.
- [ ] **TAB-05 — Table 5:** Keep canonical paired deltas. Any new seeds or objective variants require new rows/blocks with explicit identities rather than replacing the original records silently.
- [ ] **TAB-06 — Table 7:** Keep zero, displayed-tail, and native-grid-tail definitions. Grid tails are a subset, not an additional probability mass to sum.
- [ ] **TAB-07 — Table 8:** Preserve common denominator and sum-to-total checks. Do not infer exact per-site percentages from rounded operation entries when exact counts are available.
- [ ] **TAB-08 — Table 9:** Keep qualification counts, cohort weights, and noncomparable P0 subset disclosure. Add the sparse-versus-fusion factor in a separate clearly referenced column/table, not as though it were native-relative speedup.
- [ ] **TAB-09 — New quality-budget table:** Add only after DATA-02. Name the same-size A0 loss reference, eligibility rule, precision, and whether latency is measured or unavailable.

**Space rule:** Do not simply add all proposed visuals. Replace redundant search chronology or repeated maxima before enlarging the paper. Preserve the current scientific organization unless a concrete clarity benefit justifies a move.

---

<a id="7-appendix-and-reproducibility-edits"></a>

## 7. Appendix and reproducibility edits

The appendices contain important safeguards that should be preserved and selectively promoted into the main text. Technical editing must not change the reported algorithm or counter definitions.

### APP-A1 — Thresholding conventions, p. 15

**Anchor:** “For finite κ ≥ 0, both thresholding nonlinearities retain equality…”

- [ ] Preserve retained-equality behavior, detached comparison masks, exact derivative statements, and the distinction from a straight-through estimator.
- [ ] Check every main-text summary against this paragraph after terminology edits.
- [ ] Keep unselected-site behavior explicit. Do not replace it with an assumption that all sites use the same activation.

### APP-A2 — Pressure and OL1, p. 15

**Anchor 1:** “For the J targeted site–layer tensors…”

- [ ] Keep equal-tensor weighting, post-nonlinearity measurement, float32 reduction, and shared accumulation microbatches.
- [ ] Preserve the fact that naive L1 clips the combined gradient, whereas OL1 uses the task-only clipped update.
- [ ] For any new fixed-weight experiment, give it a separate objective definition and a new configuration ID. Never rewrite this paragraph as though the historical runs used that objective.

**Anchor 2:** “For OL1, let gt and g1…” through Equation (1) and the trust-budget update.

- [ ] Preserve task-only optimizer moments, pressure-gradient handling, eligibility rules, projection condition, stabilizers, group learning rates, and weight-decay exclusion.
- [ ] Compare equations and pseudocode to the actual implementation when accessible; mark any discrepancy for author resolution, not silent correction.
- [ ] Do not simplify the update into “PCGrad plus AdamW” if that omits substantive differences.

**Anchor 3:** “This procedure is related to gradient-conflict projection…”

- [ ] Keep approximate orthogonality under stabilization and the absence of a task-loss guarantee.
- [ ] Avoid presenting the related-work attribution as proof of a property this exact update has not established.

### APP-B1 — Exact counters, pp. 15–16

**Anchors:** “Activation sparsity at site s…”; “The atomic unit…”; “Here P is the causal softmax probability tensor…”

- [ ] Preserve count-pooling, the distinction between near-zero and exactly zero, and zero-weight exclusion.
- [ ] Check the logical OR in the QK/PV numerators and exclusion of future masked pairs from both numerator and denominator.
- [ ] Retain actual-consumed-operand measurement after intervening transformations and the causal-position weighting of V.
- [ ] Keep underflow zeros within the declared precision-dependent measurement. Do not substitute threshold-mask membership for actual zeros.

### APP-B2 — Reach and normalization, p. 16

**Anchors:** “For a specified graph and workload…”; “To construct the reach numerator…”

- [ ] Apply approved terminology without changing exclusions, dense-head treatment, or the block-only metric.
- [ ] Retain one credit for overlapping reach, declared zero-preserving propagation, and the caveat about natural zeros outside reach.
- [ ] Keep the warning that the gap to reach does not measure sparsity available without loss degradation.
- [ ] Audit every use of “maximum,” “ceiling,” and “utilization” in the paper against this definition.

### APP-C1 — Sites, p. 16 and continuation on p. 17

**Anchors:** “Pythia uses parallel attention and FFN branches…”; “At h, the selected nonlinearity replaces GELU.”; “Query and key counts use actual post-RoPE operands…”

- [ ] Keep tensor shapes, partial RoPE placement, and exact z definition.
- [ ] Preserve the zero-threshold A4/A7 identity statement only for the unpressured mathematical maps; pressured recipes still differ.
- [ ] Use this equivalence as the basis for the M2 implementation check, not as proof that two independently executed training traces must be bitwise identical.

### APP-C2 — Architecture-specific counts, p. 17

**Anchors:** “The operation inventory comprises…”; “Let O(N) be the union…”; “With Pythia's df = 4d…”

- [ ] Preserve Equation (4), the exact causal count, vocabulary size, and pinned configurations.
- [ ] Keep the restricted propagation rules, including V = 0 implying PV = 0 and biases blocking some other propagation.
- [ ] Rename reach columns/symbols consistently; do not recalculate a different quantity to make a ratio look more favorable.
- [ ] Verify A0 can have zero selected-site reach while still having natural observed zeros.

### APP-C3 — Evaluation, p. 17

**Anchor:** “Quality and logical counts are paired in the same eager-attention evaluation…”

- [ ] Preserve the 338 blocks, 2,048 input positions, 2,047 prediction targets per block, excluded tail, and no-cache scope.
- [ ] Describe any added BF16 evaluation separately from the original FP16 endpoint evaluation.
- [ ] Do not imply a new confirmation split exists unless one has actually been defined and evaluated.

### APP-C4 — Training protocol, pp. 17–18

**Anchors:** “We use MiniPile and the Pythia-14M-deduped tokenizer…”; “All conditions start from random weights…”; “Training uses AdamW…”

- [ ] Preserve tokenization/packing, pinned revisions, seeds, final-checkpoint rule, token counts, and all optimizer settings.
- [ ] Keep the one-seed and reused-validation disclosures. If new independent seeds are added, identify which conditions they cover rather than declaring the entire study replicated.
- [ ] Distinguish any new shortened screening runs from the canonical 712-update protocol and any later long-training confirmation.
- [ ] Record changed learning-rate schedules, repeated data, or token budgets explicitly; do not label an unmatched continuation a controlled replication.

### APP-D0 — Complete-results preamble, p. 18

**Anchor:** “The following tables report every selected trained endpoint…”

- [ ] Preserve the distinction between the 54-condition manuscript cohort and any historical or new cohorts.
- [ ] Keep dominated settings, common normalization, and count-pooling.
- [ ] Update cohort totals only after matching all records and regenerating affected tables/figures.

### APP-D1 — Distribution measurement, p. 18

**Anchors:** “The seven frozen-checkpoint evaluations…”; “Exact zeros are stored separately…”; “The diagnostic reruns use FP32 parameters…”

- [ ] Preserve pooling weights, bins, tails, precision, no-additional-clipping status, and qualification tolerance.
- [ ] Move zero masses into the main figure without deleting the detailed accounting here.
- [ ] Keep histogram rerun differences separate from independent training-run variability.
- [ ] Add new h/z or context diagnostics as separately described measurements with their own protocol and hashes.

### APP-D2 — Operation contributions, p. 21

**Anchor:** “An operation's contribution to Smodel is its zero-product count divided by the full model product count.”

- [ ] Keep the additivity explanation and common denominator.
- [ ] Update pointers if Figure 7 or a compact version moves to the main text.
- [ ] Retain the distinction between operation contributions and local activation percentages.

### APP-D3 — Post-hoc trajectories, p. 23

**Anchors:** “The complete release contains 540 evaluations…”; “Figures 8–10 retain the full loss range…”

- [ ] Preserve the calibration set, quantile target semantics, threshold inequality, post-trained-nonlinearity placement, and actual p = 0 evaluations.
- [ ] Do not confuse the clipping rule `|x| <= t` with the trained threshold's retained-equality convention.
- [ ] Keep numerical frontier membership separate from drawn lines.
- [ ] Describe any M3 baseline in a separate subsection, including allocation/calibration budget and final transformation at h.

### APP-D4 — Kernel qualification and ablations, pp. 26–27

Treat the following original paragraphs as separate edit units:

- [ ] **APP-D4-01 — “The unchanged Sakana/TwELL control…”:** Preserve upstream supported-workload versus Pythia compatibility-test distinctions and all qualification failures. Do not count an unqualified result as a valid model speedup.
- [ ] **APP-D4-02 — “Historical proposals are replayed…”:** Keep search-versus-sweep separation, 35-versus-30 cohort provenance, and excluded h-only pressure controls. Add DATA-03 results only after verification.
- [ ] **APP-D4-03 — “The coding-agent configuration records…”:** Preserve the distinction between a default configuration and verified per-request served model, plus the single human-guided trajectory. Do not infer agent superiority.
- [ ] **APP-D4-04 — “Every candidate must produce finite outputs…”:** Keep logit, relative-output, loss, process, and full-validation qualification conditions. Bounded numerical agreement is not bitwise equivalence.
- [ ] **APP-D4-05 — “Timing uses the same checkpoint…”:** Preserve timing population, paired-pass protocol, process repeats, geometric means, inclusions/exclusions, no activation cache, and equal-checkpoint weighting.
- [ ] **APP-D4-06 — “K050 combines normalization and RoPE fusion…”:** Preserve the exact differences among K049, K050, all-skips-off, and attention-dense. State that some relative factors derive from separately measured native-normalized timings; do not invent paired confidence intervals from aggregate ratios.
- [ ] **APP-D4-07 — “The attention path tests whether either…”:** Preserve the zero-fragment condition, remaining softmax/masking work, instrumentation percentages, and absence of an identified general break-even threshold.

### APP-D5 — Evidence package, p. 27

**Anchor:** “The supplementary directory supplementary-data/ contains…”

- [ ] Verify which files are actually available to reviewers. Distinguish tables/coordinates from executable training code, kernel code, and checkpoints.
- [ ] Replace unverified release promises with accurate availability statements. Do not fabricate anonymous access links.
- [ ] Provide commands that regenerate the central tables/figures when the actual repository is available; test them in a clean environment or disclose what was not tested.
- [ ] Keep data hashes, checkpoint identities, software versions, evaluation precision, and schema/unit descriptions aligned with the final manuscript.
- [ ] Preserve anonymity and any applicable disclosure requirements; confirm the current requirements rather than assuming them.

### APP-REF — References, pp. 13–14

- [ ] Preserve citations attached to retained scientific claims.
- [ ] Check bibliography metadata against the actual bibliography and, for any new or altered claim about prior work, its primary source.
- [ ] Do not silently update the literature, add an unverified 2026/2027 claim, or import statements from a review into the paper as facts.
- [ ] Do not globally rename words inside publication titles while standardizing manuscript terminology.

<a id="8-optional-experiments-and-exact-insertion-points"></a>

## 8. Optional experiments and exact insertion points

**This section is a proposed research plan, not a statement of completed work.** The rewrite agent may prepare configurations, analysis specifications, and insertion stubs in revision notes. Launching runs or adding new findings to the manuscript requires author approval and verified evidence.

### 8.1 Approval and measurement contract for every new experiment

- [ ] **EXP-00A:** Write the claim being tested, the alternative explanation, the minimum comparison, and the result that would change the paper's wording.
- [ ] **EXP-00B:** Freeze conditions, seeds, checkpoint-selection rule, evaluation, and primary contrasts before looking at new outcomes. Separate short screening from matched-budget confirmation.
- [ ] **EXP-00C:** Specify total cost for the complete package, including controls, replications, evaluation, and implementation work. Reclassify as high effort if necessary.
- [ ] **EXP-00D:** Preserve independent seed pairing within a condition set. Record actual configs and hashes, not only intended settings.
- [ ] **EXP-00E:** Define how favorable, null, mixed, and reversed outcomes change the claim. Do not create a protocol that only has a manuscript destination for favorable results.
- [ ] **EXP-00F:** Keep exploratory selection on existing validation data distinct from confirmation. Do not retrospectively describe the original analysis as preregistered or independently confirmed.

### M1 — Fixed-weight crossed placement/pressure controls

**Priority:** 5/5. **Effort:** Mid for a small short-run screen; a full multiseed, multithreshold matched-budget package may be high. **Question:** Does the placement relationship remain when adding pressure targets does not dilute the existing targets' nominal coefficients?

**Current ambiguity:** Original multisite pressure averages all targeted site–layer means. A4-OL1 versus A7-OL1 changes thresholding sites, pressure sites, and normalization. [S04](#s04), [S08](#s08)

- [ ] **M1-01:** Choose one informative nonzero κ based on the claim being tested. κ = 0.5 targets the headline high-sparsity contrast; it is not necessarily the most useful quality operating point. Record the choice before the run.
- [ ] **M1-02:** Cross thresholding sites `N4 = {a,m,h,z}` / `N7 = N4 ∪ {q,k,v}` with pressure sites `P4` / `P7`.
- [ ] **M1-03:** Define a new fixed-coefficient objective explicitly. For example, let each site–layer mean have coefficient `1/(4L)` in all four cells, where L is the number of layers; adding the three sites then adds terms without reducing the original four terms' coefficients.
- [ ] **M1-04:** Include the corresponding no-pressure N4/N7 controls at the same budget and seed, either by verified reuse or fresh runs. Do not reuse a longer-budget old checkpoint as the control for a new short run.
- [ ] **M1-05:** Keep nonlinearities, optimizer details, data order, and nominal λ fixed across the intended contrast. Label this new objective distinctly from the historical equal-target-mean objective.
- [ ] **M1-06:** Log task-direction and pressure-correction norms, effective correction size, conflict-projection frequency, and trust-budget binding. Fixed nominal coefficients do not hold the total pressure magnitude or capped correction constant.
- [ ] **M1-07:** Analyze the conditional effect of adding threshold sites at fixed pressure sites, and adding pressure targets at fixed threshold sites. Report both loss and sparsity changes, plus operation contributions.
- [ ] **M1-08:** Add a representative multisite naive-L1 condition with a clearly matched objective if the paper continues to suggest OL1 is needed. Do not claim this alone isolates the effect of conflict projection.

**Important boundary:** This design removes dilution of existing objective terms; adding terms still changes total pressure. Stronger claims about “location alone at equal effective strength” need an additional strength control. Do not oversell what the new design isolates.

**Historical reuse rule:** The old A7-OL1 objective has a different normalization from the example above. Do not silently populate the new matrix with that old run. Reuse requires demonstrably identical effective configuration, not a matching recipe name.

**Insert verified results into:** RW-40-02, a new block after RW-42-03, a separate panel/table adjacent to Figure 3, and Appendix A.2/C.4.

**Outcome-to-wording rule:**

| Outcome | Manuscript consequence |
|---|---|
| The relationship persists under fixed coefficients. | State the fixed-weight conditional comparison directly; preserve original full-recipe findings separately. |
| It shrinks or reverses. | Make objective weighting part of the result. Remove the claim that the original gap establishes an attention-placement benefit by itself. |
| It varies strongly with correction norms or budget binding. | Report that dependence; do not attribute the change solely to site choice. |

### M2 — Replicate the contrasts that carry the narrative

**Priority:** 5/5. **Effort:** Mid for a focused short-run package; potentially high in aggregate. **Question:** Which headline effects are stable across independent initializations and data orders?

- [ ] **M2-01:** Choose the conditions from the final claim ledger. A statement about threshold dependence needs at least two operating points, not repetitions of only the best high-threshold checkpoint.
- [ ] **M2-02:** Add at least two independent seed pairs to the selected original-seed conditions when feasible. Each pair shares initialization and data order across treatments, while different pairs use independent seeds.
- [ ] **M2-03:** Example minimal threshold-dependence package: A7 and A7-OL1 at κ = 0.1 and 0.5, repeated for two additional seeds = eight additional training runs. Add seed-matched A0 when making absolute-quality claims. This is an example, not a mandatory package or a guaranteed mid-effort cost.
- [ ] **M2-04:** Reuse M1 conditions only where the objective and protocol genuinely match. A new fixed-coefficient run is not a replicate of an old differently normalized condition.
- [ ] **M2-05:** Add an implementation sanity check for unpressured A4 versus A7 at κ = 0. Verify identical forward/backward mathematics and inspect execution-dependent divergence. Diagnose differences before interpreting effects of a similar scale.
- [ ] **M2-06:** Show per-seed paired deltas, averages, and clearly defined uncertainty when justified. Distinguish training variability from evaluation sampling and repeated timing variability.
- [ ] **M2-07:** Keep checkpoint selection fixed. Do not select the best seed or use different stopping points to make a comparison look stable.

**Do not substitute:** Many validation blocks, repeated inference timing, or many sweep values do not estimate independent training-run variability.

**Insert into:** RW-40-03, RW-42-01/02, Figure 3 or a compact replication panel, Appendix C.4, and the limitations paragraph.

**Outcome-to-wording rule:** Stable large effects support stronger within-protocol conclusions. Mixed or tiny effects become descriptive or inconclusive. Preserve seed-level reversals if they change the design recommendation.

### M3 — Strengthen the post-hoc comparison only if that claim remains important

**Priority:** 5/5 when retaining post-hoc superiority; otherwise defer. **Effort:** Mid for targeted calibration/evaluation; reclassify for expensive allocation searches. **Question:** Does the trained recipe's advantage survive a fairer comparator, and what aspect of the comparison actually explains it?

Treat two questions separately:

1. **Adaptation comparison:** Does training with a particular final sparsifying architecture help relative to applying that same transformation without adaptation?
2. **Competitive baseline comparison:** Does the trained recipe improve the observed quality–sparsity or quality–latency trade-off relative to a reasonably allocated post-hoc alternative?

- [ ] **M3-01:** For the adaptation question, match sites, sign treatment, and the final transformation's position in the graph. Matching the label h is insufficient: the trained map replaces GELU, whereas the current clipping control operates after the trained nonlinearity.
- [ ] **M3-02:** Specify threshold/calibration selection and any remaining architecture or optimization differences. Do not call the result a pure training-adaptation effect if those differences remain.
- [ ] **M3-03:** For the competitive comparison, implement and document a stronger allocation-aware baseline with a bounded calibration budget. Verify its primary-source specification before claiming a faithful reproduction.
- [ ] **M3-04:** Use calibration data separate from final evaluation and give the competing methods comparable selection opportunities. Do not tune allocation on the reported final outcomes.
- [ ] **M3-05:** Report all relevant points, including ties, losses, and unreachable quality budgets. Evaluate latency only with qualified implementations; do not infer speed from matched sparsity.

**Insert into:** RW-40-04, RW-41-02/03, Figure 2/5 as appropriate, and a new Appendix D.3 subsection.

**Outcome-to-wording rule:** If a stronger post-hoc baseline closes the gap, report that the original advantage was baseline-dependent. The empirical-design paper can survive this result; a broad superiority claim cannot.

**Low-effort fallback:** Keep the current uniform control and explicitly narrow every comparison to it.

### M4 — Precision-consistent accounting and functional diagnostics

**Priority:** 4/5, potentially 5/5 if the high-sparsity operating point is central. **Effort:** Mid targeted evaluations, not training by default. **Questions:** Are the counts aligned with the timed precision, and what computation remains active at extreme sparsity?

- [ ] **M4-01:** First inspect DATA-05 outputs. Reuse actual per-site counters rather than rerunning data already recorded.
- [ ] **M4-02:** Evaluate the relevant checkpoints in BF16 with the declared timed graph semantics. Record loss, exact operand-zero counts, and per-operation contributions alongside the FP16 measurements. Do not mix precisions silently in a joined analysis.
- [ ] **M4-03:** Measure h and z nonzero counts per token and layer, all-zero-vector rates, and residual-branch output magnitudes. Separate input sparsity from bias contributions and residual-stream state.
- [ ] **M4-04:** Check whether surviving coordinates vary with inputs when testing a conditional-computation interpretation. A global histogram is insufficient for this question.
- [ ] **M4-05:** Design a small context-sensitivity diagnostic with the same evaluation targets and a clearly specified context perturbation. Compare with A0 and moderate-sparsity recipes. Avoid creating a different task and then interpreting the result as the original validation score.
- [ ] **M4-06:** Instrument zero-fragment and skipped-instruction statistics only for the actual kernel path being timed. Keep those measurements separate from scalar zero-product counts.

**Insert into:** RW-43-04, RW-45-05, a compact diagnostic panel if informative, Appendix D.1/D.4, and the limitations paragraph.

**Outcome-to-wording rule:** Dynamic nonzeros and context-sensitive computation may support a functional sparsity interpretation. Largely inactive branches or strong context loss are limitations to explain, not evidence of useful conditional computation. Neither outcome alone proves global “collapse.”

### M5 — Optional optimizer-mechanism control

**Priority:** 3/5 for the recommended empirical paper; higher only if OL1 is made a central contribution. **Effort:** Mid for a narrow control; high for a broad optimizer study.

- [ ] Compare OL1 against an otherwise identical split task/pressure update with conflict projection disabled.
- [ ] Match moments, clipping, preconditioning, trust budget, parameter eligibility, and update order. Ordinary L1 alone does not isolate projection because those other details differ.
- [ ] Report how often projection and the cap activate, plus training-time cost.

**Insert into:** Section 3.1 / 4.2 and Appendix A.2 only if the evidence supports a mechanistic claim. **Default decision:** Defer and frame OL1 as an implementation choice.

### M6 — Optional sign-treatment control

**Priority:** 4/5 if a pure threshold-placement interpretation remains important. **Effort:** Mid for one controlled operating point.

- [ ] Compare the existing one-sided maps at a, m, and z with symmetric magnitude thresholds, holding the intended FFN nonlinearity at h and the q/k/v maps fixed.
- [ ] Match training protocol and state exactly which sites change sign treatment. Do not call it a placement-only test.
- [ ] Report whether the trade-off depends materially on suppressing all negative coordinates rather than only small magnitudes.

**Insert into:** Section 3.1 and a limited Section 4.2 or appendix comparison. **Default fallback:** Preserve the explicit sign-treatment limitation instead of expanding the study automatically.

### H1 — Longer-training confirmation at 70M

**Priority:** 4/5 after the main comparisons are sound. **Effort:** High. **Question:** Does a selected recipe relationship survive a less restrictive training budget?

- [ ] Select A0 and two recipes that test a specific surviving claim; do not repeat every threshold.
- [ ] Define a matched longer schedule. State whether training continues from a checkpoint or restarts, how learning rate changes, and whether tokens repeat or new data are introduced.
- [ ] Keep these choices matched across recipes. A longer sparse run against the old shorter A0 is not the intended control.
- [ ] Report trajectories and final comparisons. Do not call a favorable longer run proof of convergence or optimal training.

**Insert into:** Section 4.4, Appendix C.4, and a new training-curve appendix. **Decision value:** Distinguishes a persistent recipe relationship from an effect limited to the original budget.

### H2 — One larger-model quality–latency study

**Priority:** 4/5 for the empirical paper; 5/5 if practical efficiency is a headline. **Effort:** High because shape qualification and kernel work can dominate. **Question:** Does the execution finding extend beyond the measured 14M shapes, and is there a useful model-selection operating point?

- [ ] Start with existing 70M checkpoints rather than a new full 410M sweep.
- [ ] Qualify native, matched all-skips-off, sparse, and attention-dense paths in the same workload and precision.
- [ ] Report absolute latency, quality, per-operation time, and sparse-path attribution.
- [ ] Where feasible, compare with a smaller dense model at similar absolute validation loss. Use the same evaluator and workload; same-size Δloss values are not themselves a cross-size quality match.
- [ ] Keep shape-specific failures and negative gains. Do not extrapolate sparse-kernel speedup from the architecture's reach or the scalar zero fraction.

**Insert into:** A bounded extension of Section 4.5 and its appendix. **Decision value:** Supports a broader efficiency story only when the measured comparison actually favors the sparse model at a defensible quality level.

### 8.2 Stopping and escalation rules

| After completing… | Continue when… | Stop or narrow when… |
|---|---|---|
| Low-effort rewrite | The two primary findings remain nontrivial and their evidence gaps are clearly identified. | The narrative still relies on unqualified maximum sparsity, an unfair baseline, or a prediction claim from a descriptive fit. |
| M1/M2 | Controlled and replicated effects support a meaningful comparison, or a reversal itself explains an important dependency. | Only tiny unstable differences remain. Remove those as headlines rather than adding more adjectives or an indiscriminate sweep. |
| M3 | Competitive/adaptation claims remain important and the baseline is now appropriately matched. | The stronger baseline removes the claimed advantage. Report the boundary and keep the empirical contribution. |
| M4 | Functional and precision diagnostics clarify the meaning of the extreme point. | The high-sparsity point cannot support the proposed practical interpretation. Keep it as a stress-test endpoint, not the recommended operating point. |
| Before H1/H2 | A specific remaining generality or practical-value objection is worth the cost. | The unresolved problem is still a basic missing control or unclear claim; larger models will not resolve that ambiguity. |

---

<a id="9-final-verification-and-author-handoff"></a>

## 9. Final verification and author handoff

### 9.1 Claim-strength audit

- [ ] **QA-01:** Every abstract and introduction claim maps to a result, comparator, and explicit scope in the evidence ledger.
- [ ] **QA-02:** All uses of “isolates,” “causes,” “synergy,” “preserves,” “dominates,” “predicts,” “optimal,” “robust,” and “generalizes” have been reviewed manually. Keep only meanings justified by the experiment.
- [ ] **QA-03:** High sparsity is not presented without quality context; same-checkpoint acceleration is not presented as a quality-matched model advantage.
- [ ] **QA-04:** Complete-recipe larger-model comparisons are not relabeled as individual pressure or placement effects.
- [ ] **QA-05:** A post-hoc result names the actual baseline and does not imply a full TEAL reproduction unless one was performed.
- [ ] **QA-06:** Results from optional experiments appear only after their records, analysis, and qualification checks have been verified. No future-tense proposal has become a past-tense finding by accident.

### 9.2 Numerical and methodological audit

- [ ] **QA-07:** Cross-check every headline value against its exact table/record, including reference, precision, units, and rounding.
- [ ] **QA-08:** Verify operation contributions sum to model-wide counts at full precision. Tolerate only explained display-rounding discrepancies.
- [ ] **QA-09:** Verify reach formula, common A7 normalization, and any `U_arch` → `S_block` relabeling. Do not apply this identity outside its declared graph.
- [ ] **QA-10:** Reconcile 30 manuscript checkpoints, 35 historical checkpoints, 54 training endpoints, 540 clipping evaluations, seven histogram checkpoints, and any new cohorts. Do not update one count without its downstream figures and tables.
- [ ] **QA-11:** Preserve 1.7830× search versus 1.7832× final-sweep context; distinguish K050 from the faster attention-dense ablation.
- [ ] **QA-12:** Verify that native-relative, fusion-relative, and absolute-latency comparisons are labeled correctly. Ranges are not confidence intervals; separately measured normalized ratios are not raw paired timing records.
- [ ] **QA-13:** Check the exact threshold inequalities, derivative conventions, pressure averaging, clipping, and OL1 equations against the unchanged method and any explicitly new variants.
- [ ] **QA-14:** Verify the dataset/evaluation split, target-token versus input-token counts, training budget, learning-rate differences, and final-checkpoint rule.
- [ ] **QA-15:** Keep all relevant exclusions and qualification failures. Missing measurements must say unavailable, not zero or equivalent.

### 9.3 Human-readability and insight audit

Ask a reader—or a separate review pass—to answer these questions from the revised paper without consulting this runbook:

1. What are the two principal findings, and which comparison supports each?
2. Why can fewer pooled FFN zeros produce more model-wide zero-product counts?
3. Why is the architectural reach neither a universal observed-sparsity bound nor a quality-preserving target?
4. Which part of the reported acceleration comes from sparse paths rather than matched fusion?
5. What changes across model sizes besides parameter count, and what remains untested?

- [ ] **QA-16:** Each substantive results subsection contains at least one clearly supported implication, not only a sequence of numbers.
- [ ] **QA-17:** No implication outruns its evidence. Proposed mechanisms and future tests are labeled as such.
- [ ] **QA-18:** Recipe IDs are expanded with site or pressure meaning when the reader needs it. A page of unexplained A4/A7 comparisons is not a readable argument.
- [ ] **QA-19:** Remove duplicated caveat paragraphs, repeated maxima, and self-praise. Keep short local caveats that prevent a misleading reading.
- [ ] **QA-20:** Read each section continuously, then the abstract and conclusion together. The ending must not make a broader claim than the opening or vice versa.

### 9.4 Build and visual audit

- [ ] **QA-21:** Compile the source after structural edits and at completion. Resolve undefined citations, references, and missing figures.
- [ ] **QA-22:** Inspect every rendered page at normal reading size. Check legend readability, symbols, zero-mass annotations, table headings, figure order, and page breaks.
- [ ] **QA-23:** Check that the manuscript and supplement use the same names and units. Preserve raw data identifiers or document aliases.
- [ ] **QA-24:** Search the final manuscript for unresolved placeholders, `TODO`, `NEW_DATA`, obsolete symbols, and contradictory old headlines. Place unresolved requests in revision notes, not submission prose.
- [ ] **QA-25:** Verify availability statements and instructions against the actual delivered files. Do not claim the code, checkpoints, or plots were reproduced when they were not.

### 9.5 Required author handoff

Deliver a compact package, not an unexplained rewritten manuscript:

1. **Revised source and compiled PDF**, when source was available; otherwise anchored replacement passages with explicit insertion locations.
2. **Change log by task ID**, including major claims narrowed, removed, or supported by new evidence.
3. **Claim/evidence ledger and numerical checks**, identifying all remaining missing data and unverified items.
4. **Experiment decision note**, distinguishing completed runs from approved-but-pending and proposed-only work.
5. **Readiness assessment using Section 1.3**, including blocking issues and a candid recommendation on whether more experiments are worth the cost.

### 9.6 Final readiness decision

**Low-only completion:** The paper can become clearer, fairer, and more informative. Treat acceptance as uncertain because the single-seed and identification limitations remain. Do not report that a rewrite has “resolved” them.

**Low + targeted mid completion:** This is the recommended route to a defensible empirical paper. A favorable readiness assessment requires informative comparisons that withstand the relevant controls, or a clearly explained reversal that itself provides a useful result. It does not require every recipe to win.

**Low + mid + selected high completion:** Supports a broader claim only to the extent the new data support it. Larger models and longer training are not substitutes for a clear contribution, fair comparator, or correct attribution.

**The final manuscript should leave readers thinking:** “I now know what to measure, which comparison to trust, and what experiment to try next”—not merely “this paper reports a large sparsity percentage.”

### 9.7 Restart prompt for a new agent

```text
You are continuing the rewrite of the supplied activation-sparsification manuscript.
Read the runbook's Start Here section, the current revision state, the glossary,
and the claim/evidence ledger. Work only on the next approved batch of task IDs.
Locate each original paragraph by its heading and opening anchor before editing.
Use verified source data; distinguish measured, derived, inferred, and proposed content.
Do not launch experiments or add findings from the optional experiment plan.
Do not silently change equations, objective normalization, cohort definitions, or baselines.
Make the smallest coherent patch, check neighboring paragraphs and dependent captions,
then update the state with completed tasks, evidence used, and unresolved blockers.
Stop the batch when a required record or author decision is missing; use the documented
safe fallback rather than inventing evidence or expanding the scope.
```

---

<a id="10-portable-source-register"></a>

## 10. Portable source register

All entries below refer to the supplied manuscript, **Activation Sparsification in Transformers: Pressure, Thresholding, and Site Placement**, file `main(20260909-182851).pdf`. These references are intentionally portable: they use printed page numbers, section/table/figure names, and anchors rather than chat-only citation identifiers.

The register records what the manuscript supports. Its descriptions of cited prior work are not an independent verification of those external papers. Proposed editorial changes, experiments, and readiness judgments in this runbook are recommendations, not findings reported by the manuscript.

<a id="s01"></a>
**S01 — Abstract and title.** p. 1. Contains the current isolation claim, reach/“ceiling” terminology, maximum sparsity figures, uniform post-hoc frontier claim, 1.78× headline, and R² association.

<a id="s02"></a>
**S02 — Introduction.** pp. 1–2, Section 1. Five paragraphs, beginning respectively “Activation sparsity can improve…,” “Existing approaches induce…,” “In this work, we study…,” “We pretrain Pythia-family models…,” and “Finally, we test whether…”.

<a id="s03"></a>
**S03 — Related work.** pp. 2–3, Sections 2.1–2.2. Describes post-hoc thresholding, pressure/training adaptation, attention placement, different inference workloads, structured execution, and agent-assisted development as background.

<a id="s04"></a>
**S04 — Intervention definitions and placement.** p. 3, Section 3.1; p. 5, Figure 1; p. 15, Appendix A.1–A.2; p. 16, Appendix C.1 / Table 2. Supports threshold signs and equality conventions, derivative distinctions, pressure normalization, OL1 update semantics, and exact site definitions.

<a id="s05"></a>
**S05 — Sparsity accounting and architectural reach.** p. 4, Section 3.2; pp. 15–16, Appendix B; p. 17, Appendix C.2 / Equation (4). Defines model/block counts, dense-head denominator, causal validity, actual operands, zero-propagation rules, and why selected-site reach is not an upper bound on every observation.

<a id="s06"></a>
**S06 — Experimental setup and training limitations.** p. 4, Section 4 opening; pp. 17–18, Appendix C.3–C.4 / Table 3. Supports dataset and evaluation details, 30/12/12 cohorts, 712 updates, approximately 1.493 billion input tokens, one seed per condition, final-checkpoint evaluation, unequal tokens per parameter, learning-rate differences, and uniform clipping setup.

<a id="s07"></a>
**S07 — Quality–sparsity and post-hoc evidence.** pp. 4 and 6, Section 4.1 / Figure 2; p. 19, Table 4; pp. 23–25, Appendix D.3 / Figures 8–10. Contains all 54 trained endpoints and describes the complete 540 clipping evaluations, actual p = 0 points, high-loss settings, and restricted comparator scope.

<a id="s08"></a>
**S08 — Paired intervention effects.** pp. 6–7, Section 4.2 / Figure 3; p. 20, Table 5; p. 15, Appendix A.2 for normalization. Supports the 29 reported contrasts, conditional pressure results, local L1/OL1 reversal, and complete-objective interpretation of differing target sets.

<a id="s09"></a>
**S09 — Activation distributions and operation explanation.** pp. 7–9, Section 4.3 / Figure 4; p. 18, Appendix D.1; p. 21, Tables 7–8 / Appendix D.2; p. 22, Figure 7. Supports zero-mass pooling, nonlocal distribution changes, h:m weights, Q/K post-RoPE measurement, and operation contributions at κ = 0.5.

<a id="s10"></a>
**S10 — Cross-size complete-recipe comparisons.** p. 8, Section 4.4 / Table 1; p. 10, Figure 5; pp. 19–20, Tables 4 and 6. Supports the high-threshold A7-OL1 versus A4-OL1 ordering, smaller-threshold reversals, common A7 normalization, block-only interpretation, and scale limitations.

<a id="s11"></a>
**S11 — Kernel results, controls, and provenance.** p. 11, Section 4.5; p. 12, Figure 6; pp. 26–27, Appendix D.4 / Table 9. Supports search versus final-sweep maxima, 30-checkpoint qualification, matched fusion controls, sparse-path factors, negative attention skipping, MMA instrumentation, precision mismatch, workload, P0's limited subset, and five excluded historical h-only-pressure checkpoints.

<a id="s12"></a>
**S12 — Discussion and scope.** p. 13, Section 5. Connects conditional pressure, operation accounting, and measured execution, and states dataset/budget, model-size, distribution, workload, and hardware limitations.

<a id="s13"></a>
**S13 — Evidence package.** p. 27, Appendix D.5. Describes `supplementary-data/`, records, coordinates, hashes, protocol, and repository-relative paths; explicitly distinguishes that directory from a standalone checkpoint or kernel release.

---

**End of runbook.** Execute by task ID, preserve the evidence boundary, and optimize for clear scientific understanding rather than stronger-sounding claims.
