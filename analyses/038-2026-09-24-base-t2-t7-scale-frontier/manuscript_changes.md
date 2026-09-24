Make a **focused revision of the scaling experiment**, while preserving the 14M study as the paper’s mechanistic core.

Below is a manuscript-order change plan. **Page, figure, and table numbers refer to the current PDF.** Proposed manuscript text appears in block quotes; bracketed fields require values from your new run logs.

# Before editing: establish exactly what the 31M experiment contains

Use **31M**, matching the figure, provisionally. Confirm the exact parameter count and architecture before changing the manuscript’s model descriptions.

The image supplies loss–latency relationships, but not the following details:

| Required information                                                                       | Why it matters for the revision                                                   |
| ------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- |
| Exact parameter count, layers, hidden width, FFN width, heads, vocabulary size             | Needed for the architecture description and sparsity ceilings.                    |
| Actual \(\kappa\) values and checkpoint-to-point mapping                                   | Needed before making comparisons “at matched \(\kappa\).”                         |
| Training budget, initialization/data-order seeds, optimizer settings, pressure parameters  | Needed before describing the new run as part of the matched protocol.             |
| Exact validation losses, measured \(S_{\mathrm{model}}\), native and specialized latencies | Needed for numerical claims and the appendix results table.                       |
| Kernel version, numerical-validation results, timing protocol and session identifiers      | Needed to interpret the new runtime evidence alongside the existing measurements. |

**Do not recover publication numbers from the image.** The replacements below that mention matched thresholds or a common protocol are contingent on those details being confirmed.

---

# Main-text changes

## 1. Abstract, page 1: scope the quality improvement and add the intermediate-scale result

### Find this passage

> “These observations motivate a targeted intervention that achieves lower validation loss than the dense control while reducing latency from 0.655 to 0.573 ms.”

The abstract currently gives the favorable loss–latency result without explicitly attaching “14M” to that sentence. The preceding context discusses 14M, but the new multi-scale presentation makes explicit scoping more important. 

### Replace it with

> These observations motivate thresholding only \(h,z\) while applying pressure only at \(h\). At 14M, this intervention achieves lower validation loss than the dense control while reducing latency from 0.655 to 0.573 ms. Comparisons at 31M and 70M retain the lower quality cost of targeted versus seven-site thresholding, but not the validation-loss improvement over the dense control.

Keep the final co-design sentence.

For greater precision, once the threshold pairing is confirmed, replace “retain the lower quality cost” with:

> show lower validation loss for targeted than seven-site thresholding at matched thresholds

### Also make this small scope edit

Change:

> “Conditional kernel ablations identify…”

to:

> “At 14M, conditional kernel ablations identify…”

That prevents readers from assuming that the new 31M experiment includes the same site-by-site execution ablations.

**Do not add a claim that latency benefits grow monotonically with model size to the abstract.** The result is affected by the size-specific implementations, and the new figure does not disentangle that dependence.

---

## 2. Introduction, page 2: add one sentence explaining what the new comparison tests

### In the final introduction paragraph, locate

> “At 14M and \(\kappa \leq 0.1\), \(T_2/P_h\) achieves both lower validation loss than the dense control and lower full-model latency…”

Immediately afterward, the manuscript moves to the general co-design principle. 

### Insert between those two statements

> We additionally compare \(T_2/P_h\) and \(T_7/P_h\) at 31M and 70M, testing whether the benefit of narrowing threshold coverage persists when pressure placement remains fixed at \(h\).

This is a particularly useful addition because it tells the reader **what is controlled** in the new experiment.

You do not need to repeat the detailed outcomes here; the abstract and revised §4.4 will cover them.

### Leave the preceding 14M narrative intact

Keep the introduction’s sequence of global-sparsity decoupling, profitable kernel sites, spillover, and optimization conflict. Those claims remain supported by the existing 14M experiments, not by the new loss–latency plot. 

---

## 3. Table 1, page 4: add the architecture, but distinguish analytic coverage from executed experiments

Table 1 currently has sparsity-ceiling columns for **14M, 70M, and 410M**. Its caption also states that the matched 70M \(T_2/P_h\) latency sweep stops at \(\kappa=0.1\), with the \(\kappa=0.5\) endpoint reported separately. 

### Edit the column structure

Change:

> Ceiling (%) — 14M / 70M / 410M

to:

> Ceiling (%) — 14M / 31M / 70M / 410M

Compute the 31M column from its actual architecture. Do not interpolate between the existing ceilings.

It is acceptable to show analytic ceilings for recipes not trained at 31M, because these are architectural quantities. Make that distinction explicit.

### Add to the caption

> Ceilings describe architectural reach and do not imply that every recipe was trained at every size; the executed conditions are listed in Appendix C.3.

### Resolve the 70M endpoint note

Your new figure includes the low-latency \(T_2/P_h\) endpoint consistent with the existing \(\kappa=0.5\) result. Before revising the caption, determine whether that endpoint has now been included in a matched timing sweep.

**Do not silently delete the existing qualification.** Either retain its separate-measurement status or update the statement to describe the new measurement procedure accurately.

This is a bookkeeping change, but it protects the cross-scale comparison from mixing apparently identical points with different timing provenance.

---

## 4. Opening of §4, page 4: distinguish the exhaustive study from the scale-extension study

### Find

> “We pretrain Pythia-14M, 70M, and 410M from random initialization…”

followed by the common training-budget description. 

### Replace the opening with

> We pretrain Pythia-family models at 14M, 31M, 70M, and 410M from random initialization on MiniPile. The extensive intervention-placement study uses 14M; the cross-scale analysis compares targeted and broad thresholding at 14M, 31M, and 70M. Additional 70M recipes and the 410M fixed-token stress test are reported in Appendix D.

Then retain the existing sentences about 712 updates, 1.493 billion input tokens, matched initialization/data order, and validation **only where they also describe the new run accurately**.

Using “Pythia-family models” avoids implying that the exact new 31M configuration was already one of the configurations documented by the original citation. Specify that configuration in Appendix C.

---

## 5. Figure 5, page 8: replace the figure, and rewrite—not merely extend—the caption

This is the most consequential visual edit.

The current Figure 5 highlights **\(T_2/P_h\) versus \(T_7/P_{\mathrm{all}}\)** and includes other recipes in gray. Your new figure instead highlights **\(T_2/P_h\) versus \(T_7/P_h\)** and adds 31M. The scientific comparison has changed, not just the number of marker shapes. 

### Replace the current caption with

> **Quality–latency trade-offs for targeted and broad thresholding across model sizes.** Circles, triangles, and squares denote 14M, 31M, and 70M, respectively. Blue dashed curves show \(T_2/P_h\), which thresholds \(h,z\); orange solid curves show \(T_7/P_h\), which thresholds all seven sites. Both recipes apply pressure only at \(h\). Lines connect measured threshold settings within each recipe and model size. Open and filled gray markers show the same Base checkpoint under specialized-kernel and native PyTorch execution, respectively; vertical guides mark each size’s Base validation loss. Timings use RTX 5090, BF16, batch one, 2,048 tokens, and full vocabulary logits. Complete numerical results are reported in Appendix D.1.

Retain the hardware/protocol sentence only after confirming it covers the new measurements.

The existing caption says both axes are logarithmic. The supplied image appears consistent with that, but check the plotting code before carrying that sentence over. 

### Make four plotting edits

1. **Label the vertical guides** with the corresponding model size.
2. **Expose the threshold ordering**, either through selected \(\kappa\) annotations or a clearly cross-referenced results table.
3. **Connect points in threshold order**, not sorted validation-loss order.
4. **Remove the old “64 intervention points” and “other recipes gray” descriptions.** They no longer describe this figure.

Do not describe the line segments as measured frontiers. They connect measured checkpoints; intermediate points on the segments have not necessarily been evaluated.

### Preserve the broader evidence elsewhere

Keep the existing \(T_7/P_{\mathrm{all}}\) results in Tables 2–4, Figure 1, and Appendix D. The new figure should clarify the threshold-placement comparison, not erase the pressure-placement results that motivate the paper.

---

## 6. §4.4, pages 7–8: replace the section’s framing and reorganize its evidence

### Change the heading

From:

> **4.4 RESULTS AT 70M**

to:

> **4.4 TARGETED AND BROAD THRESHOLDING ACROSS MODEL SIZES**

This names the experiment more accurately than a generic “scaling results” heading.

### Replace the opening claim

The section currently begins:

> “Scaling improves quality and amplifies the latency payoff of targeted sparsification.”

It later concludes that targeted sparsity remains effective across scales and its latency benefit grows substantially with size. 

That framing blends three different observations: changes in dense-model quality, changes along a threshold sweep, and speedups relative to native execution.

### Replace the section with four short paragraphs

The following is a proposed integrated replacement. The matched-\(\kappa\) statement requires confirmation from the new checkpoint mapping.

> **Holding pressure placement fixed across scales.** Figure 5 compares \(T_2/P_h\) and \(T_7/P_h\) at 14M, 31M, and 70M. Both recipes apply OL1 pressure only at \(h\), while threshold coverage changes from \(h,z\) to all seven sites. The intermediate 31M experiment therefore extends a controlled comparison of threshold placement rather than replicating the full 14M intervention study.
>
> **Targeted thresholding preserves quality, but not uniformly relative to Base.** At matched \(\kappa\), \(T_2/P_h\) has lower validation loss than \(T_7/P_h\) at all three evaluated sizes. Broader thresholding nevertheless reaches lower minimum latency in these sweeps, so neither recipe is uniformly preferable across both metrics. Relative to the dense control, moderate \(T_2/P_h\) improves validation loss at 14M, but the evaluated 31M and 70M settings incur a positive loss penalty. At 31M, \(T_2/P_h\) with \(\kappa=[[\mathrm{THRESHOLD}]]\) obtains loss \([[\mathrm{LOSS}]]\) and latency \([[\mathrm{LATENCY}]]\) ms, compared with Base loss \([[\mathrm{BASE\_LOSS}]]\) and native latency \([[\mathrm{BASE\_LATENCY}]]\) ms.
>
> **Within-recipe latency reductions differ from speedups over native execution.** Increasing \(\kappa\) from 0 to 0.1 within \(T_2/P_h\) reduces specialized-kernel latency by 0.039 ms at 14M, \([[\mathrm{31M\_REDUCTION}]]\) ms at 31M, and 0.549 ms at 70M. These reductions measure the response to thresholding within each implementation; they do not by themselves establish a speedup over the dense PyTorch Base. At 70M, for example, \(T_2/P_h\) at \(\kappa=0.1\) remains slower than native Base, whereas \(\kappa=0.5\) is faster but incurs a larger validation-loss increase.
>
> **Kernel efficiency remains size-dependent.** The specialized implementation’s Base latency is \([[\mathrm{31M\_KERNEL\_BASE}]]\) ms at 31M, compared with \([[\mathrm{31M\_NATIVE\_BASE}]]\) ms for native PyTorch. At 70M, the partially optimized implementation takes 2.732 ms on Base versus 1.611 ms natively. These differences show why the training-side trade-off must be evaluated together with the implementation used at each size. The intermediate-scale result broadens the evidence for targeted thresholding, but does not establish automatic kernel portability or quality-free speedups.

The existing manuscript supports the quoted 14M/70M sweep reductions and the 70M implementation gap. The appendix also shows the distinction between moderate 70M settings that remain slower than native Base and the aggressive endpoint that becomes faster.   

### Two implementation notes for this replacement

First, choose the illustrative 31M threshold by a stated comparison rule—such as the same moderate threshold used in the main 14M comparison—not simply whichever point looks best after inspection.

Second, retain the current 14M/70M numbers only if the revised figure uses their existing measurement series. If you rebenchmark those points, update the paragraph and appendix together.

### Relocate two existing details rather than discard them

The current §4.4 discusses the aggressive \(T_7/P_{\mathrm{all}}\) loss penalties and the initial 70M kernel latency of 3.325 ms. 

Move the former to the commentary accompanying the complete results in Appendix D.1. Move the detailed initial-versus-reoptimized 70M kernel history to Appendix E.1. This keeps the revised main section focused on the comparison actually shown in Figure 5.

---

## 7. §5, page 8: replace the first discussion paragraph, not the co-design conclusion

### Find the paragraph beginning

> “Our results suggest that effective train-time sparsification should be targeted rather than broad.”

It currently joins site susceptibility, gradient conflict, and the best quality–latency trade-offs into a single general statement. 

### Replace it with

> Our results distinguish the benefits of targeted thresholding from the costs of broader pressure. At 14M, execution ablations and optimization diagnostics identify \(h,z\) as useful sparsification sites under the studied implementation. Comparisons of \(T_2/P_h\) and \(T_7/P_h\) across 14M, 31M, and 70M further show that narrowing threshold coverage preserves validation quality at matched thresholds, while broader thresholding can reach lower latency at a larger quality cost. The validation-loss improvement over the dense control is confined to the evaluated 14M conditions. The broader result is therefore a favorable quality–latency trade-off from targeting useful sites, rather than a general improvement in quality from sparsification.

Keep the subsequent paragraph beginning:

> “More broadly, sparsification is one instance of a training–inference co-design problem.”

That paragraph already expresses the right generalization: identify removable expensive computation, then train for the structure that makes it removable. 

**Do not introduce “sparsification acts as a regularizer” as an established mechanism.** Lower validation loss in the 14M measurements does not, by itself, identify why the improvement occurs.

---

## 8. §5.1, pages 8–9: update coverage without implying that the limitations disappeared

### Replace the opening

Current:

> “Our scaling analysis focused on 14M and 70M. We also trained 410M models.”

with:

> Our main analysis combines extensive 14M intervention ablations with targeted-versus-broad thresholding comparisons at 14M, 31M, and 70M. We also trained 410M models, but the fixed-token budget places that size in a different optimization regime.

This preserves the existing rationale for treating 410M separately.  

### Add a scope sentence

After the sentence about the partially explored intervention space, insert:

> The 31M comparison holds pressure placement fixed at \(h\); it does not independently establish the spillover mechanism, site-wise optimization conflict, or conditional execution savings measured at 14M.

### Keep the remaining limitations

Retain the one-seed limitation, narrow runtime scope, cross-site averaging confound, and shared-threshold limitation. An additional model size is not an additional seed or a replication of the full intervention matrix. 

---

# Appendix and reproducibility changes

## 9. Appendix C.2, page 13: document the architecture and complete the \(T_2\) ceiling accounting

### Find

> “Table 1 uses \(V_{\mathrm{vocab}}=50{,}304\) and \((L,d)=(6,128),(6,512),(24,1{,}024)\) for 14M, 70M and 410M…”

Add the actual 31M configuration in the corresponding position. Update the “Pinned architecture configurations” footnote to include its configuration artifact or identifier. 

Do not assume its layer count or width from the nominal parameter count.

### Add the missing \(T_2\) numerator explicitly

The current ceiling-accounting paragraph lists \(T_1\), \(T_4\), and \(T_7\), but does not explicitly give the \(T_2\) numerator. 

Insert:

> For \(T_2\), the per-layer numerator is \(Tdd_f + Td^2\), covering the FFN-down and attention-output projections. When \(d_f=4d\), this becomes \(5Td^2\).

This is a derivation from the paper’s existing accounting, not an additional empirical result. It makes the ceiling for the now-central cross-scale recipe transparent.

---

## 10. Table 7, page 13: add a complete 31M training-settings column

The current table reports parameter counts, learning rates, batch settings, sequence length, and tokens per parameter for the original three sizes. 

### Add a 31M column between 14M and 70M

Populate every row from the actual configuration:

| Table 7 row                | Required 31M entry                                         |
| -------------------------- | ---------------------------------------------------------- |
| Parameters                 | Exact count, not the rounded model name.                   |
| Peak/minimum learning rate | Actual schedule values.                                    |
| Microbatch                 | Actual sequences per microbatch.                           |
| Gradient accumulation      | Actual accumulation count.                                 |
| Effective batch            | Verified product of the preceding quantities.              |
| Sequence length            | Actual training sequence length.                           |
| Tokens per parameter       | Actual input-token count divided by exact parameter count. |

Do not simply copy the 14M or 70M column because the experiment was intended to follow the same protocol.

This table also helps readers understand that a common token budget does not create a common tokens-per-parameter regime.

---

## 11. Appendix C.3, page 13: make the executed-condition coverage explicit

Table 1 says executed recipes are listed in Appendix C.3, but the current C.3 text primarily describes shared training settings rather than enumerating that coverage.  

### Add a short paragraph after the optimizer/schedule description

> **Executed conditions.** The 31M comparison includes Base and the \(T_2/P_h\) and \(T_7/P_h\) recipes at \(\kappa \in \{[[\mathrm{ACTUAL\_THRESHOLDS}]]\}\). Both sparse recipes use OL1 with \(\lambda=[[\mathrm{LAMBDA}]]\) and \(b=[[\mathrm{BUDGET}]]\). The comparisons share \([[\mathrm{CONFIRMED\_MATCHED\_SETTINGS}]]\). Their complete results are reported in Appendix D.1.

Add any additional 31M conditions only if they were actually run.

Also check the existing statement that all sizes use initialization and data-order seeds of 1234. Do not let that become a false description of the new experiments merely by adding “31M” elsewhere.

---

## 12. Figure 6, page 14: add the 31M training trajectory when the logs support it

Figure 6 currently shows Base training loss and pre-clipping task-gradient norm for 14M, 70M, and 410M. Its caption notes that gradient norms are not normalized by parameter count. 

### Preferred edit

Add the 31M Base trajectory to both panels, using the same axes, smoothing convention, and underlying quantities.

This provides useful context for the new model’s optimization regime.

### When matching logs are unavailable

Keep the figure, but change the caption opening to:

> Base-model training at 14M, 70M, and 410M.

That avoids suggesting the figure now covers every evaluated size. Do not manufacture a gradient-norm trajectory from another logged quantity.

---

## 13. Appendix D.1, pages 14–15: add a dedicated 31M results table

### Keep the existing Table 8 layout

Table 8 already spans two pages and reports the existing 14M/70M comparisons. Adding a third full model block would make it substantially wider.  

### Insert a new table immediately after Table 8’s continuation and before §D.2

Use a stable label such as `tab:31m-results`; let LaTeX renumber later tables automatically.

Suggested columns:

| Recipe | \(\kappa\) | \(S_{\mathrm{model}}\) (%) | Loss | \(\Delta\) loss | Latency (ms) | \(\Delta\) latency |
| ------ | ---------- | -------------------------- | ---- | --------------- | ------------ | ------------------ |

Include Base and every 31M checkpoint used in Figure 5. Report the specialized-kernel Base latency separately in the caption or as a clearly distinguished execution row.

### Suggested caption

> **31M targeted-versus-broad thresholding comparison.** Results for the conditions shown in Figure 5. Both sparse recipes apply pressure only at \(h\). Loss and latency differences are relative to the 31M native PyTorch Base, using the fixed references reported in Appendix E.2. Validation metrics and timing follow the protocols in Appendices C.3 and E.2.

Add the pressure parameters and seed count after confirming them.

If \(S_{\mathrm{model}}\) has not yet been measured, mark it as not measured or omit that column temporarily. **Do not infer it from the loss–latency plot or from the analytic ceiling.**

### Extend the D.1 opening paragraph

Add:

> The additional 31M comparison is reported in Table \(\ref{tab:31m-results}\).

The main figure can then emphasize the scientific contrast without becoming the only record of the new results.

---

## 14. Appendix D.2, page 15: update the scale list, not the interpretation of 410M

The existing paragraph lists tokens per parameter at 14M, 70M, and 410M and explains why 410M is treated as a limited stress test. 

### Edit the tokens-per-parameter sentence

Include the verified 31M value in the sequence.

### Replace

> “We therefore treat 410M as a limited stress test and use 14M/70M for detailed intervention and runtime analysis.”

with:

> We therefore treat 410M as a limited stress test. The main runtime comparisons cover 14M, 31M, and 70M, with the extensive intervention-placement diagnostics concentrated at 14M.

Leave the 410M numerical table unchanged unless the underlying results themselves have changed.

---

## 15. Appendix E.1 and Table 10, pages 15–16: document the 31M implementation rather than assuming portability

Table 10 explicitly describes the **14M and initial 70M implementations**. It distinguishes pre-weight-load checks at \(h,z\) from checks that retain operand reads at other sites.  

### Add a paragraph identifying the 31M implementation

Suggested structure:

> **31M implementation.** The 31M measurements use \([[\mathrm{IMPLEMENTATION/VERSION}]]\), derived from \([[\mathrm{PROVENANCE}]]\). Relative to the implementations summarized in Table 10, the changes are \([[\mathrm{CHANGES}]]\). The enabled sparse execution paths are \([[\mathrm{PATHS}]]\).

Specify enough to determine whether the new result reflects simple dimension adaptation, additional tuning, different sparse paths, or other implementation changes.

**Do not merely add “31M” to the Table 10 caption unless all its execution rules actually apply.**

This matters because the paper’s central argument is about the interaction between training and execution, not training alone.

---

## 16. Appendix E.2, page 16: extend validation and baseline accounting

### First, check the existing numerical-validation claim

The manuscript currently states that final checkpoints were checked on all 338 validation sequences in three processes, subject to finite-output, per-logit, relative-error, and validation-loss tolerances. 

Ensure that statement genuinely includes the new checkpoints before presenting them under the same protocol. The fact that a point appears in a latency figure does not establish numerical equivalence.

### Second, update the baseline-reference sentence

Current:

> “Table 8 uses fixed references of 0.65544 ms at 14M and 1.61105 ms at 70M.”

Add a separate sentence:

> Table \(\ref{tab:31m-results}\) uses the fixed 31M native Base reference of \([[\mathrm{31M\_BASE\_REFERENCE}]]\) ms.

The surrounding text already distinguishes cross-recipe comparisons from same-checkpoint execution controls; preserve that distinction. 

### Third, keep the denominators consistent

In the current manuscript, **0.652 ms is the specialized-kernel 14M Base**, while **0.65544 ms is the fixed native 14M reference**. Those are different baselines, not interchangeable rounding choices.  

Apply the same discipline at 31M: distinguish native Base, specialized Base, and a sparse checkpoint’s dense-path control.

---

## 17. Appendices E.3–E.4, pages 16–17: preserve the existing execution controls and their scope

The paper already contains a controlled 14M \(T_2/P_h\) experiment and a controlled 70M \(T_7/P_h\) experiment. These hold the checkpoint and other components fixed while changing sparse execution, unlike the cross-recipe comparisons in Figure 5.  

### Required manuscript action

Keep these tables and their size-specific labels. Do not generalize them to “all evaluated scales.”

### Optional addition with a clear purpose

A 31M sparse-\(h,z\)-on/off control would support attributing part of the 31M speedup specifically to those execution paths. It does not require another training sweep.

Without that control, the appropriate manuscript claim remains **an end-to-end quality–latency comparison at 31M**, not a new causal decomposition of its runtime savings.

---

# What should remain unchanged

The new experiment should not trigger a rewrite of every empirical section.

| Existing material        | Recommended action                                                                               |
| ------------------------ | ------------------------------------------------------------------------------------------------ |
| Figure 1 and Tables 2–3  | Keep the 14M sparsity–latency and quality–latency evidence.                                      |
| §4.2 and Figure 3        | Keep spillover explicitly grounded in the measured 14M site/layer sparsities.                    |
| Figure 4                 | Keep the OL1 geometry evidence scoped to the measured conditions.                                |
| §4.3 and Table 4         | Keep the conditional site-saving results and their implementation-specific interpretation.       |
| Appendix A.1 and Table 5 | Keep the OL1 definition and limited L1 comparison; the new figure does not test OL1 superiority. |

Those sections already provide the mechanisms that motivate the targeted recipe. The 31M experiment extends the recipe comparison; it does not replace that evidence.    

# Final consistency pass

After making the substantive edits, search the LaTeX for **“14M and 70M,” “14M/70M,” “both scales,” “Results at 70M,” “64 intervention points,” and “executed recipes.”** Update each occurrence according to its actual scope rather than performing a global replacement.

Also check every Figure 5 reference for the comparator change: **\(T_7/P_{\mathrm{all}}\) in the old figure versus \(T_7/P_h\) in the new figure.** Conversely, do not change Figure 1’s checkpoint count merely because the paper now includes another scale.

The revised paper should leave the reader with a precise hierarchy of evidence:

> **The 14M experiments identify useful sites and diagnose why broad sparsification can be inefficient. The 31M and 70M comparisons test the quality–latency consequences of narrowing threshold coverage while holding pressure placement fixed. Together, they support targeted training–inference co-design—not universally better quality, universal frontier dominance, or automatic kernel scaling.**
