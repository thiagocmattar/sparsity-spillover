Rebuild the figures from this folder.

# Shared conventions for all figures

Use these consistently so readers do not have to relearn the encoding.

| Element                 | Convention                                                                                                                           |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Model size**          | Separate panels for 14M and 70M. Do not use marker fill to encode size.                                                              |
| **Threshold scope**     | Four-site: blue diamonds. Seven-site: orange triangles. Dense: gray star. ReLU/one-site: green circles.                              |
| **Pressure scope**      | No pressure: open markers, solid lines. \(h\)-only: filled markers, short-dashed lines. All-site: filled markers, long-dashed lines. |
| **Naming**              | “4-site,” “7-site,” “OL1(h),” and “OL1(all).” Remove `A4*`.                                                                          |
| **Threshold sweeps**    | Connect points in increasing \(\kappa\), not increasing sparsity. Caption: separately trained settings, not training trajectories.   |
| **Paired-effect plots** | Show each \(\kappa\) explicitly. Do not summarize the five thresholds with boxplots.                                                 |
| **Precision**           | Compute from unrounded source records; round only displayed annotations.                                                             |

**Why replace the boxplots?** Your current paired-effect figures summarize variation across five thresholds, not independent replications. Their medians and quartiles hide the threshold dependence that is now the central result. The existing captions correctly identify what the boxes represent, but explicit threshold-response plots communicate the finding better.  

Build every plot from a common checkpoint-indexed table. Join quality, sparsity, and timing by **checkpoint identity**, not rounded sparsity or row order. Keep kernel configuration, timing session, and evaluation precision as separate metadata.

---

# Figure 1 — Overall quality–sparsity trade-offs

**Proposed title:**
**Threshold and pressure choices produce different quality–sparsity trade-offs**

**Placement:** Introduction; referenced again in Section 4.1.

**Goal:** Establish the available operating points and motivate separating threshold scope from pressure scope.

| Specification        | Description                                                                                                                                                                                                                                                       |
| -------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Layout**           | Two side-by-side panels: **(a) 14M**, **(b) 70M**.                                                                                                                                                                                                                |
| **X-axis**           | **Model-wide sparsity \(S_{\mathrm{model}}\) (%)**.                                                                                                                                                                                                               |
| **Y-axis**           | **Validation-loss difference from dense reference (nats/token)**: \(\Delta L=L-L_{\mathrm{dense},m}\), using the corresponding model size’s reference.                                                                                                            |
| **Data**             | All available trained recipe endpoints in the declared comparison set; dense and ReLU controls; post-hoc thresholding trajectories from both dense and ReLU checkpoints. Include pressure-free multisite models at 14M; do not invent corresponding 70M controls. |
| **References**       | Horizontal line at \(\Delta L=0\). Vertical guides for four-/seven-site computational coverage, labeled **coverage**, not attainable sparsity or speedup ceilings.                                                                                                |
| **Main annotations** | Label `.05` and `.5` on the multisite trajectories, with selective offsets to avoid overlap. Highlight measured nondominated points without hiding the others.                                                                                                    |

### Display decisions

Use a common Y range across the two panels that includes **all trained endpoints**. Show the high-loss continuation of post-hoc curves in a full-range appendix figure rather than compressing the main comparison.

This explicitly changes your uploaded `12-...` plot, which combines model sizes on absolute-loss axes and extends to approximately nine nats. Those choices make the within-size trained-recipe differences difficult to inspect. 

Do not connect points from different recipe families into an apparently continuous Pareto curve. An outline around nondominated measured points is sufficient.

### Caption must state

The dense-relative normalization; one final checkpoint per trained setting; which sites the post-hoc procedure thresholds; omitted high-loss post-hoc evaluations and their appendix location; and that connecting lines do not imply attainable interpolated models.

**Intended conclusion:** Useful operating points depend on the intervention combination. Some trained configurations extend the evaluated frontier, but no recipe dominates every quality budget.

---

# Figure 2 — Intervention sites and independent scopes

**Proposed title:**
**Thresholding and pressure act on independently selected activation sites**

**Placement:** Methodology.

**Goal:** Give every subsequent recipe label an unambiguous architectural meaning.

**Keep the existing architecture diagram**, with these requirements:

| Specification                | Description                                                                                                      |
| ---------------------------- | ---------------------------------------------------------------------------------------------------------------- |
| **Axes**                     | None; this is a computation graph.                                                                               |
| **Content**                  | Parallel attention and FFN branches, separate LayerNorms, and the residual addition.                             |
| **Projection-input sites**   | Identify \(a,m,h,z\), each beside its consuming activation–weight multiplication.                                |
| **Internal-attention sites** | Identify \(q,k,v\), with \(q,k\) after RoPE; distinguish probabilities \(p\) from these intervention sites.      |
| **Scope explanation**        | \(T_4=\{a,m,h,z\}\); \(T_7=T_4\cup\{q,k,v\}\). Pressure targets are specified independently in the recipe table. |
| **Boundary**                 | State that the language-model head is not targeted and is omitted from the block diagram.                        |

Do not color every marked site as though it always receives pressure. The diagram identifies **candidate locations**; the recipe table specifies the actual assignments.

**Caption must state:** Site counts refer to site types per block, repeated across layers—not the total number of layers or interventions.

---

# Figure 3 — Pressure scope versus threshold strength

**Proposed title:**
**The benefit of broader pressure depends on threshold strength**

**Placement:** Section 4.2.

**Goal:** Show the central new result directly: when does expanding pressure from \(h\) to all thresholded sites help or hurt?

### Data transformation

For size \(m\), threshold scope \(i\in\{4,7\}\), and threshold \(\kappa\):

$$
\Delta L_{m,i,\kappa}
=
L(T_i/P_i)-L(T_i/P_h),
$$

$$
\Delta S_{m,i,\kappa}
=
100\left[S(T_i/P_i)-S(T_i/P_h)\right].
$$

The second expression is in percentage points when \(S\) is stored as a fraction. When source records already store percentages, simply subtract them.

| Specification     | Description                                                                                                |
| ----------------- | ---------------------------------------------------------------------------------------------------------- |
| **Layout**        | \(2\times2\): rows are **14M / 70M**; columns are **loss difference / sparsity difference**.               |
| **X-axis**        | **Trained threshold \(\kappa\)**, with five explicitly labeled categorical positions: \(0,.01,.05,.1,.5\). |
| **Y-axis, left**  | **Loss change: OL1(all) − OL1(h) (nats/token)**.                                                           |
| **Y-axis, right** | **Sparsity change: OL1(all) − OL1(h) (pp)**.                                                               |
| **Curves**        | Four-site and seven-site thresholding, using the same scope colors and shapes as Figure 1.                 |
| **References**    | Horizontal zero line in every panel; shared Y limits between sizes within each column.                     |
| **Data volume**   | Twenty paired comparisons: two sizes × two threshold scopes × five thresholds.                             |

### Required annotations

For seven-site at 70M, annotate these two contrasts from your earlier endpoint table:

* \(\kappa=.05\): **−0.371 pp sparsity, +0.297 nats loss**.
* \(\kappa=.5\): **+8.369 pp sparsity, −0.121 nats loss**.

These make the change in the trade-off visible without requiring readers to inspect every point.

### What not to include

Do not add latency as a third column here. Keep this figure focused on the pressure-scope finding; evaluate deployment consequences in Figure 5.

Your uploaded `06-...` supplies the existing 14M comparison, but replace the boxplots with explicit \(\kappa\) curves and add the 70M endpoints. 

**Caption must state:** All-site minus \(h\)-only at fixed threshold placement; positive loss differences are worse; positive sparsity differences mean more zeros; pressure normalization changes with the target set; no independent-seed uncertainty is represented.

---

# Figure 4 — Operation-level sources of sparsity changes

**Proposed title:**
**Pressure changes sparsity unevenly across operations**

**Placement:** Section 4.4.

**Goal:** Explain where the aggregate changes come from, including whether narrow pressure changes sparsity outside its directly targeted operation.

| Specification | Description                                                                                                                                        |
| ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Layout**    | Two panels: **(a) \(\kappa=.05\)** and **(b) \(\kappa=.5\)**, using 14M.                                                                           |
| **X-axis**    | Four labeled contrasts: \(T_4:P_h-P_0\), \(T_4:P_4-P_h\), \(T_7:P_h-P_0\), \(T_7:P_7-P_h\).                                                        |
| **Y-axis**    | **Change in contribution to model-wide sparsity (pp)**.                                                                                            |
| **Data**      | Per-operation zero-product counts for the exact checkpoints used in the comparisons.                                                               |
| **Segments**  | QKV projection, FFN-up, FFN-down, attention-output projection, QK, and PV.                                                                         |
| **Geometry**  | Diverging stacked bars: positive contributions above zero, negative contributions below zero. Place a separate black marker at the **net change**. |

For each operation \(o\),

$$
C_o=100\frac{N_{0,o}}{N_{\mathrm{model}}},
\qquad
\sum_o\Delta C_o=\Delta S_{\mathrm{model}}\text{ in pp}.
$$

Use a consistent operation palette within this figure. Here color represents **operation**, not recipe; the X labels identify recipes.

### Interpretation to support

For \(P_h-P_0\), inspect whether contributions outside FFN-down increase. That is the relevant evidence for a nonlocal sparsity response.

For \(P_{\mathrm{all}}-P_h\), identify whether broad pressure increases internal-attention sparsity while reducing sparsity in some projection inputs.

### Data dependency

**The newly uploaded PDFs contain aggregate sparsity, not this operation decomposition.** This figure requires the retained operation counters. Do not infer its segments from aggregate \(S_{\mathrm{model}}\), pooled FFN sparsity, or timing.

**Caption must state:** Every segment shares the full-model denominator; negative segments mean reduced sparsity under the treatment; the net marker is the signed sum; these are logical zero-product opportunities, not measured runtime savings.

---

# Figure 5 — Computational sparsity versus useful execution

**Proposed title:**
**More computational sparsity does not necessarily yield a better quality–latency trade-off**

**Placement:** Section 4.5.

**Goal:** Use the new 70M test to connect representation-level choices to actual deployment outcomes.

This should now be a **four-panel figure**, not a single sparsity–speedup regression.

|         | **Column 1: sparsity and execution**  | **Column 2: deployment trade-off** |
| ------- | ------------------------------------- | ---------------------------------- |
| **14M** | \(S_{\mathrm{model}}\) versus latency | Dense-relative loss versus latency |
| **70M** | \(S_{\mathrm{model}}\) versus latency | Dense-relative loss versus latency |

### Axes and data

| Specification          | Description                                                                                                                                                                                |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **X-axis, left**       | **Model-wide sparsity \(S_{\mathrm{model}}\) (%)**.                                                                                                                                        |
| **X-axis, right**      | **Validation-loss difference from dense reference (nats/token)**.                                                                                                                          |
| **Y-axis, all panels** | **Full-model latency (ms)**.                                                                                                                                                               |
| **Y scaling**          | Linear; identical limits within each size’s row, different limits between 14M and 70M.                                                                                                     |
| **Data**               | All checkpoints in the declared, qualified runtime cohort, including the newly timed \(h\)-only models. Join their ordinary final-checkpoint losses and sparsities by checkpoint identity. |
| **Reference**          | Dense model point and horizontal dense-reference latency line within each size.                                                                                                            |

**Use the actual full-model kernel configuration measured in the new experiment.** Do not label these times “projection-only” unless that is what the timing metadata establishes. The uploaded plots identify full-model latency, not a 70M projection-only ablation. 

### Left column: what to emphasize

Reuse the structure of your uploaded `08-...` plot. It already separates sizes and shows absolute latency.

Label `.05` and `.5` for the main four-/seven-site pressure families. Let readers see examples where more sparsity produces similar or worse latency. Do not force a pooled regression through both model sizes.

### Right column: what to emphasize

Show the same checkpoints, but replace sparsity with quality cost.

Highlight the **observed nondominated configurations**: another measured configuration must not have both lower/equal loss and lower/equal latency, with at least one strict improvement.

Do not draw a continuous Pareto envelope implying interpolation between models. Label selected frontier points with recipe and \(\kappa\).

This is the panel that determines whether a configuration is practically attractive; the largest speedup alone does not.

### Caption and comparability requirements

State hardware, precision, batch size, sequence length, logits workload, kernel configuration, aggregation method, and exact checkpoint coverage.

Your existing paired plots disclose that some \(h\)-only comparisons span GPU sessions. Preserve that qualification; small latency differences should not be presented as established winners without comparable timing evidence.  

Only show runtime uncertainty intervals supported by repeated timing records. They represent measurement variation conditional on a checkpoint—not training-seed uncertainty.

**Important boundary:** The new 70M results support an end-to-end runtime comparison. They do not, by themselves, establish that 70M attention skipping is profitable.

---

# Keep threshold-placement effects as Table 2

Do not create another main figure for the comparison already represented by your `05-...` plot.

Use a table of

$$
Y(T_7/P_h)-Y(T_4/P_h)
$$

with five \(\kappa\) rows and columns for \(\Delta S\) and \(\Delta L\) at both sizes.

This preserves threshold identity and keeps pressure scope fixed. The `OL1(all)` row in the uploaded topology plot changes pressure scope as well as threshold scope, so it should not be labeled a pure threshold-placement effect. The plot itself acknowledges that distinction. 

---

# Appendix figures

These preserve useful evidence without crowding the main argument.

| Figure                                      | Title, axes, data, and goal                                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **A1 — Complete quality–sparsity results**  | **Title:** “Complete trained and post-hoc quality–sparsity trajectories.” Two size panels. X: \(S_{\mathrm{model}}\) (%). Y: ordinary validation loss (nats/token), full range. Include every evaluated post-hoc point and trained endpoint in the declared cohort. **Goal:** document the full comparison, including points outside Figure 1’s range.                                                                             |
| **A2 — Pressure versus no pressure at 14M** | **Title:** “Effects of adding narrow and broad pressure.” Rows: \(P_h-P_0\) and \(P_{\mathrm{all}}-P_0\). Columns: \(\Delta L\), \(\Delta S\), and \(\Delta t\). X in every panel: explicit \(\kappa\) settings. Curves: four-site and seven-site. **Goal:** retain the comparisons in `07-...` without pooling thresholds. Caption the cross-session timing limitation.                                                           |
| **A3 — Dense-reference speedup**            | **Title:** “Full-model speedup relative to the dense reference.” Two size panels. X: \(S_{\mathrm{model}}\) (%). Y: \(t_{\mathrm{A0},m}/t_{\mathrm{recipe},m}\). Show a \(1\times\) line and identify the exact A0 execution implementation. **Goal:** retain the normalized view from `10-...` while keeping its denominator explicit.                                                                                            |
| **A4 — Instruction-level explanation**      | **Title:** “Scalar zeros, instruction bypass, and projection-skipping benefit.” Two panels: X is projection scalar zero-product fraction or projection MMA bypass; Y is \(t_0/t_P\). Use only checkpoints with matching counters and skipping ablations. **Goal:** explain execution structure. Retain the established 14M cohort unless equivalent 70M data exist; do not extend the old regression statistics to the new cohort. |

For **A3**, distinguish:

$$
\underbrace{\frac{t_{\mathrm{dense}}}{t_{\mathrm{recipe}}}}_{\text{dense-reference deployment comparison}}
\quad\neq\quad
\underbrace{\frac{t_{\mathrm{native,\ same\ recipe}}}
{t_{\mathrm{optimized,\ same\ recipe}}}}_{\text{implementation speedup}}.
$$

The new plot uses the former. Its caption must identify whether the dense reference is native or optimized and whether reference and candidate measurements are session-matched. A0 normalization alone does not eliminate session differences. 

# Exact reuse plan for your uploaded files

| Current file                            | Action                                                                                                               |
| --------------------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| `12-...quality-sparsity-clipping.pdf`   | Split by size and normalize loss for **Figure 1**; preserve full-range values in **A1**.                             |
| `06-...paired-pressure-effects.pdf`     | Replace boxes with threshold-response curves; add 70M for **Figure 3**.                                              |
| `05-...paired-topology-effects.pdf`     | Convert the fixed-\(P_h\) quality/sparsity comparison into **Table 2**.                                              |
| `07-...pressure-vs-none-effects.pdf`    | Replace boxes with explicit threshold curves for **A2**.                                                             |
| `08-...final-sparsity-latency.pdf`      | Use as **Figure 5’s left column**; join quality to build its right column.                                           |
| `10-...matched-sparsity-a0-speedup.pdf` | Retain as **A3**, with explicit reference and timing-session definitions.                                            |
| `09-...latency-log-y.pdf`               | Omit from the manuscript: redundant with the size-separated absolute-latency figure and normalized-speedup appendix. |

**The main missing visual is now the quality–latency frontier, not another sparsity–speedup plot.** The new 70M timings make that comparison possible; the operation-level figure remains the separate diagnostic needed to explain where the zeros come from.
