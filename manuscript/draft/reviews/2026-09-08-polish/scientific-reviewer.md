# Scientific reviewer and evidence audit

Review date: 8 September 2026. This is an independent review of the draft as
read during the subsection-by-subsection polish, not a review of the eventual
revised PDF. No draft prose, figure, experiment, or source measurement was
changed by this reviewer. The first two introduction paragraphs are protected
by the user's instruction.

## Assessment

The strongest contribution is a controlled account of when activation
pressure adds useful sparsity, and how that response depends on thresholding
and site placement. The paired comparisons, actual product accounting, and
kernel ablations support that contribution. The evidence is substantially
stronger than a collection of sparsity percentages: it connects the location
of zeros to a declared workload and then separately tests an implementation.

The manuscript can be scientifically presentable as a bounded empirical
study after the editorial changes below. The evidence does not establish a
general superiority claim for OL1, a scaling law, a causal mechanism of
spillover, or a broadly transferable sparse kernel. Single-seed, single-data,
fixed-budget pretraining and one-device inference remain consequential limits;
prose cannot remove those limits. No new experiment is needed to make the
existing claims accurate, and this review does not authorize one.

## Severity-ranked actions

### High: finish the paper-level argument

The reviewed `main.tex` has no abstract or closing discussion/conclusion.
Add a short abstract stating the controlled choices, 54 trained conditions,
conditional pressure result, restricted cross-size result, and separate
kernel realization. End with the practical conclusion and the principal
scope limits. Do not end the paper directly on the kernel figure: that makes
the systems demonstration appear to replace the central training result.

The introduction should distinguish comparing design choices from reproducing
TEAL, ProSparse, or Q-Sparse end to end. The protected second paragraph
motivates a common comparison; the experimental setup correctly specifies a
uniform TEAL-style control. Preserve that qualification and state the actual
training recipes plainly. The study does not reproduce Q-Sparse's top-k/STE
method or ProSparse's full progressive schedule.

### High: restrict what “best kernel” means

Figure 07 calls K050 the best kernel, while its attention-dense ablation is
faster on every included checkpoint. The frozen-source audit explicitly
restricts “best” to searched historical proposals, excluding diagnostic
ablations. Add this restriction in the caption or immediate prose. The
existing figure artwork can remain unchanged. Report K050 as the selected
searched implementation; do not call it the best measured implementation.

Prefer “1.311 times as fast as the same fused implementation with skips
disabled” to “adds 31.1% acceleration.” The latter can be mistaken for a
31.1% latency reduction; the corresponding reduction is approximately 23.7%.
The 1.0432 cohort ratio helps 14/30 checkpoints and must stay next to this
claim. A large favorable endpoint must not stand in for the cohort.

### Medium: make reproducibility explicit in the appendix

The training appendix gives the optimizer, budget, precision, initialization
and schedule matching, but the reviewed text omits the dataset/tokenizer
revisions and EOS/packing construction. Add the precise pinned tokenizer and
data preparation, or provide a compact protocol manifest accompanying the
paper with an explicit reference. `research/DATA.md` gives the authoritative
identities. The released figure-data summary contains cache and schedule
hashes, but is not by itself a complete replacement for those protocol
details. Add software/code identities for the specialized inference result
where already retained; do not claim an unavailable public release.

The paper should state that all reported quality comparisons use the same
validation split and are descriptive; there is no independent confirmation
split or seed-level uncertainty estimate. This is best stated once in a
concise limitations paragraph rather than repeated after every result.

### Medium: preserve the causal and mathematical boundaries while shortening

- A4-OL1 versus A7-OL1 changes both gate and pressure sites and the relative
  weights of existing sites in the equal-tensor objective. It isolates neither
  placement nor pressure alone.
- A4-OL1 versus A0 changes more than pressure. Changed q/k/v marginals are a
  nonlocal response under that recipe, not proof of pressure causing a
  particular route through the network.
- A7 directly gates and pressures q/k/v. Its attention reshaping should not
  be described as untargeted spillover.
- The density curves omit a very large exact-zero atom. Their disappearance
  near the origin does not represent the full distribution becoming empty.
  Keep the omitted-zero explanation and the exact-zero table reference in
  the caption. Filled area on a symlog axis is not a probability readout.
- The all-zero reach ceiling is selected-site reach, not a quality-preserving
  bound and not an upper bound on every measured value. Natural zeros outside
  the selected reach can add to the numerator. The current text handles this
  correctly and should retain it.
- At kappa zero, the added A7 q/k/v gates are identities. The tiny unpressured
  A7-minus-A4 residual is not a mechanism of additional gating. Appendix
  wording may identify it as a numerical training residual if discussed.
- A1-H to A4 also changes h: its boundary derivative at zero and its positive
  threshold for nonzero kappa. Do not describe this contrast as affecting only
  a/m/z.

### Low: reduce repetition and disclose metric precision once

Captions can share one common protocol reference rather than repeating 500
documents, 338 blocks, one seed, and the excluded tail in each. Retain facts
needed to decode each plot: point count, varying parameter, denominator,
marker meaning, excluded zero mass, and visible-range limitations.

The kernel scatter uses canonical FP16 sparsity and BF16 timing; this is
correctly disclosed in its caption. Keep the distinction. The descriptive
OLS fit is not a causal regression or a quality-matched comparison, and no
confidence interval over training seeds can be derived from these 30
different treatments.

## Claim/evidence audit

| Claim or definition | Authoritative evidence inspected | Outcome |
| --- | --- | --- |
| Random pretraining; matched same-size initialization/order | All 54 endpoint source `diagnostics/logical_products.json` files and retained initialization/schedule hashes | Confirmed; 30/12/12 conditions and one initial/schedule hash per size |
| Full validation and training budget | Operational data contract; retained coverage; training identities and source records | Confirmed: 338 complete 2048-token blocks, 692224 inputs, 691886 prediction targets, 1444 excluded tail; 712 updates and 1493172224 training inputs |
| Gate semantics and h replacement | `research/METHODS.md`, methodology appendix and site definitions | Correct: equality survives, masks detached; G+ at zero has ReLU values but a different derivative at zero; symmetric zero-threshold gate is identity |
| OL1 definition | `src/sparsity_research/pressure.py:apply_ol1_correction` versus equations in the appendix | Exact match in scope: eligible parameters, updated task-only moments, unclipped pressure, conflict condition and stabilizer, global pre-learning-rate budget, post-AdamW correction; no task-loss guarantee |
| Model-wide accounting | Retained integer operation counts, `metrics.py`, `ceilings.py`, methodology equations | All 54 integer numerators and denominators reconcile; activation zeros only, causal pairs only, dense head denominator with no zero credit |
| Reach derivation | `ceilings.py` and the per-operation formula | Correct: per-block T(4d²+2dd_f)+dT(T+1); A4 reach 12Td² for d_f=4d; A7 reaches every counted block operation |
| Common A7 normalization | Direct recomputation for all 54 endpoints | Confirmed: A7 ceilings 29.95237697%, 49.42390059%, 87.24517739%; S_model/Smax(A7) equals S_block exactly |
| Local OL1 and A0 clipping comparison | Retained endpoint and clipping coordinates | Correct examples: local OL1 lambda .5 loss 5.11023594 at 3.76799756%; A0 clipping p .3 at 3.832% and loss 5.3619. Near-matched sparsity does not mean identical reachable operations |
| Paired effects | Independently recomputed treatment-minus-reference values from all 29 source-linked pairs | Confirmed. OL1 has lower loss than naive L1 at 3/4 local weights, not all four |
| Cross-size boundary result | All 15 scale pairs and the six-row compact table | Confirmed: A7-OL1 beats A4-OL1 on both axes at kappa .5 in all sizes; zero-threshold ordering reverses at 410M |
| Reshaped signed distributions | Seven copied raw compressed histograms; all 14 pooled groups; density plot data and O011 | Count partitions and plotted zero fractions confirmed; 99.31/93.64% FFN and .22/95.60% attention at kappa .5 are supported |
| Attention contribution | Six-operation table reconstructed from retained count fractions | Correct: at 14M kappa .5, QK+PV contribute 16.77 pp for A7-OL1 versus .06 pp for A4-OL1; larger S_model can coexist with less pooled FFN sparsity |
| Complete clipping release | All 540 normalized rows and 54 source checkpoint groups | Confirmed: ten targets 0,.1,...,.9 for every checkpoint; integer sums and every loss-minus-measured-p0 value recomputed successfully |
| K050 cohort summary | Independent calculation from the 30 retained qualified final points | GM 1.2339917066; range 1.0059384947–1.7831745640; all 30 qualified |
| Kernel association | Independent OLS with intercept using the 30 canonical points | R² .8167176698, slope 3.7703908645 per fraction, intercept .9512951280; descriptive association only |
| Sparse-path ablation | Matched checkpoint ratios of independently measured native-normalized timings | K050/all-skips-off GM 1.0432289680; 14/30 wins; c30 ratio 1.3111529878 |
| Attention skip overhead | Matched attention-dense control and source implementation audit | Attention-dense faster on all 30; GM 1.2505587529. This must qualify “best” |
| Search progress | Retained progress array and Run 029 audit | 43 rows including native iteration zero; 42 proposals; maximum 1.7830291241. Separate final measurement explains 1.7831745640 |
| Code correctness scope | Run 029 implementation/claim audit and reviewed pressure/accounting source | Cohort-level qualification supported; arbitrary thresholds are not. A retained BF16 threshold-rounding defect at kappa .7 does not affect the actual cohort thresholds |

I verified all 231 hashes recorded in the Analysis 018 source list against
the current worktree. All source files exist. The 226 non-draft files match;
the five mismatches are the intentionally revised manuscript TeX/PDF. This
source list records historical manuscript context, not current-draft integrity.
Do not rewrite historical hashes to make a new version appear identical.

The local `figures/SOURCES.json` has 11 entries and every copied file matches.
The local supplementary-data manifest has 14 entries and every copied file
matches. These checks establish byte identity and the audits above establish
meaning; a hash check alone would not validate the science.

## Figure and table contribution scores

Scores use 1 = weak/unnecessary, 3 = useful support, 5 = central evidence.
They concern scientific contribution; visual-layout scoring is a separate
review.

| Artifact | Score | Role and short insight |
| --- | ---: | --- |
| Main 01 overview | 4/5 | Establishes different quality costs at similar sparsity and the broader reachable regime; it motivates the decomposition |
| Main 02 paired effects | 5/5 | Central evidence: pressure's extra value depends on the gate and target set |
| Main 05-v3 density | 4/5 | Shows that recipes reshape signed distributions differently; the zero-mass table is necessary to interpret it |
| Main 03 cross-size | 4/5 | Specific high-threshold ordering survives, while the zero-threshold ordering changes |
| Main 07 kernel | 4/5 | Separates measured execution from logical opportunity; ablations prevent over-attributing fusion gains |
| Main compact scale table | 4/5 | Gives the exact reversal and high-threshold comparison without reading tiny plot differences |
| Appendix site table/architecture ladder | 4/5 | Defines the exact ports and reachable operations required to understand every contrast |
| Appendix training settings | 4/5 | Makes the fixed-token and differing-learning-rate interpretation checkable |
| Appendix full 54 endpoints | 4/5 | Preserves dominated settings and both local pressure variants |
| Appendix full 29 pairs | 4/5 | Supports every component comparison and makes unfavorable effects visible |
| Appendix all 15 scale pairs | 4/5 | Prevents the boundary comparison from hiding intermediate thresholds |
| Appendix density mass | 5/5 | Supplies the discrete zero probability removed from every continuous overlay |
| Appendix operation table/04 | 4/5 | Explains why less FFN sparsity can yield more model-wide sparsity |
| Appendix three full clipping plots | 3/5 | Important completeness record; machine-readable coordinates carry their detailed value |
| Appendix kernel ablations | 5/5 | Essential evidence separating sparse paths, fusion, and detrimental attention skipping |

The all-variant 01-v2 is a useful retained alternative, but the current
manuscript intentionally embeds the original overview. Both L1 and no-pressure
families are already visible in the paired figure and complete tables. No
scientific requirement forces an artwork switch during this prose polish.

## Reviewer rubric

Rubric: 1 = serious failure, 2 = weak, 3 = adequate, 4 = strong, 5 = unusually
strong. Scores describe the reviewed draft, not a promise of acceptance.

| Criterion | Score | Reason |
| --- | ---: | --- |
| Research question and relevance | 4/5 | Practical choice of combined sparsification interventions, with workload-aware consequences |
| Distinct scientific contribution | 3/5 | Controlled decomposition and accounting are useful; individual techniques are established and OL1 is not uniformly superior |
| Mathematical/implementation precision | 5/5 | Gates, OL1, count pooling, causal denominators and reach closure agree with source |
| Evidence supporting scoped claims | 4/5 | Complete endpoints and diagnostics, matched comparisons and informative runtime ablations |
| Generalizability/statistical strength | 2/5 | One seed, one dataset, three small architectures at unequal tokens/parameter, one runtime device/workload |
| Reproducibility and provenance | 4/5 | Strong retained source identities and data; appendix needs the compact preparation/environment identities |
| Claim calibration | 4/5 | Most distinctions are explicit; “best kernel” still needs restriction |
| Narrative economy | 3/5 | Repeated qualifications and very long captions obscure several simple insights |
| Paper completeness before polish | 3/5 | Results are complete, but abstract and concluding synthesis are absent |

## Primary-literature spot checks

The current descriptions of top-k/STE in
[Q-Sparse](https://arxiv.org/abs/2407.10969), smoothed gradient projection in
[Bloop](https://arxiv.org/abs/2402.02998), L1-induced FFN sparsity plus
specialized CUDA kernels in
[Sparser, Faster, Lighter Transformer Language Models](https://arxiv.org/abs/2603.23198),
and execution-feedback kernel development in
[Agentic Kernel Optimization](https://arxiv.org/abs/2608.14560) are consistent
with their primary records. This spot check does not claim a new exhaustive
related-work search or priority proof. It supports retaining modest
attribution rather than asserting novelty over every prior method.

## Acceptance conditions for this review

Before the root agent's final handoff, verify that the new abstract and closing
paragraphs use the scoped claims above; “best” is restricted; the omitted-zero
and common-A7-denominator explanations survive caption edits; source identities
are available; no new significance, causality, or state-of-the-art claim has
appeared; and every changed caption still matches the unchanged artwork.
Rebuild and inspect the complete document after those edits. No new data,
larger claim, or figure redesign is necessary to satisfy these conditions.


## Follow-up review of the revised prose

The revised abstract, introduction after the protected paragraphs, methods,
experimental setup, results, kernel subsection, conclusion, and appendices
were reread. The abstract and conclusion now close the argument; K049, P0,
and MMA are defined; the kernel caption restricts best to searched proposals;
and data preparation/environment identifiers are supplied. The retained
numbers and scope qualifications remain scientifically accurate. Narrative
economy and paper completeness improve to 4/5 on this follow-up reading.

Two small scientific/provenance edits remain for the root agent: describe
protocol.json as a locally composed record rather than a byte-identical
measurement copy in the supplementary README, and avoid attributing the
A0-clipping versus A7-OL1 difference to separate learning/reach contributions.
That comparison changes both the trained representation and site coverage.

The user's requested terminology change is scientifically compatible with
the work: use activation sparsity, model-wide sparsity, sparsity ceiling,
and thresholds/nonlinearities. The ceiling definition must retain its exact
estimand: the fraction of counted products guaranteed to have a zero
activation operand when selected sites have 100% activation sparsity. It
must not become the total observed sparsity of such a hypothetical checkpoint,
which could include natural zeros elsewhere, or a quality-preserving bound.

Protocol provenance was independently rechecked: all 30 repository-relative
source entries resolve and match their SHA256 and byte counts. The new
kernel.known_limits field records the kappa .7 BF16 threshold-rounding
counterexample, its affected inherited h/z and Q/K/V paths, unaffected K050
a/m threshold rounding, the exhaustive agreement at actual cohort thresholds,
and the frozen implementation's broader execution/qualification limits.
The bug does not affect the reported cohort; this is not a claim that the
implementation supports arbitrary thresholds or inputs. No new experiment,
TeX edit, or figure edit was performed by this reviewer.
