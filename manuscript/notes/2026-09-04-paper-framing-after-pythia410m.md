# Paper framing after the Pythia-410M promotion

- Date: 2026-09-04
- Status: paper-planning note approved for retention
- Scope: argument, result selection, and next-work priorities

This note records a paper strategy for later drafting. It does not promote a
finding, revise result-bearing TeX, approve a new experiment design, or approve
a compute launch.

## Executive decision

Do not try to rescue, replace, or hide the Pythia-410M result for the current
paper. Freeze the verified Run 019/021 evidence and report 410M as the
preplanned fixed-token stress test that partially refuted the original
scale-persistence expectation.

The paper's main empirical argument should rest on the Pythia-14M and
Pythia-70M results. Its central contribution should be topology-aware,
model-wide logical-product accounting and the resulting
validation-loss--`R_model` frontiers. The 410M result supplies an important
boundary condition: logical opportunity remains large, but the quality and
dose-response ordering are not portable under a common one-pass token budget.

Do not make uniform-spillover, general OL1-superiority, monotonic model-scaling,
or runtime-speedup claims.

## Evidence motivating the decision

Every scale received the same 712 optimizer boundaries and 1,493,172,224
scheduled input tokens. Because parameter count changes sharply, this is not a
capacity-scaled training budget.

| Model | Total parameters | Training tokens per parameter | A0 paired validation loss | Mean A0 train loss, boundaries 649--712 |
| --- | ---: | ---: | ---: | ---: |
| Pythia-14M | 14,067,712 | 106.142 | 5.208594 | 5.249282 |
| Pythia-70M | 70,426,624 | 21.202 | **4.099766** | 4.009096 |
| Pythia-410M | 405,334,016 | 3.684 | 4.547456 | 4.493717 |

The 410M A0 baseline is therefore worse than the 70M baseline despite having
more parameters. Its late training-loss windows continue to decline more
strongly than those of 14M and 70M, which is consistent with an insufficient
optimization horizon but does not prove that explanation.

Run 021 tested the narrow alternative that the one-pass 410M run merely needed
a larger peak learning rate:

| Peak LR | Mean train loss, boundaries 649--712 | Paired validation loss | Decision |
| ---: | ---: | ---: | --- |
| `3e-4` | **4.493717** | **4.547456** | selected Run 019 baseline |
| `6e-4` | 4.513309 | 4.552068 | not selected |
| `1e-3` | 6.851160 | 6.786663 | not selected |

This refutes a simple upward peak-LR repair within the tested grid. It does not
test a longer horizon, a different schedule shape, a lower LR, a second data
pass, new training tokens, or another seed.

The intervention result is not uniformly negative. At `kappa=0.5`, A7-OL1
dominates A4-OL1 on both plotted axes at all three scales:

| Model | A4-OL1 loss / `R_model` | A7-OL1 loss / `R_model` |
| --- | --- | --- |
| Pythia-14M | 6.037982 / 12.7134% | **5.829407 / 27.4827%** |
| Pythia-70M | 5.389480 / 35.5962% | **5.215925 / 40.6019%** |
| Pythia-410M | 5.190966 / 71.5914% | **5.120692 / 80.6155%** |

However, the complete frontier does not persist. A4-OL1 dominates A7-OL1 at
`kappa=0` for 14M and 70M, while A7-OL1 dominates at 410M. More importantly,
increasing `kappa` from 0 to 0.5 increases loss for the 14M/70M trained
families but decreases it for both 410M families. The 410M run is therefore not
just a vertically displaced version of the same dose-response curve.

## Recommended paper argument

Suggested title:

> **Beyond Local Activation Sparsity: Model-Wide Zero-Product Frontiers in
> Transformers**

Suggested thesis:

> Local activation zero rates do not determine model-wide sparse-compute
> opportunity. Topology-aware exact operand counting exposes where zeros occur
> and how much model-wide matrix-multiplication work is logically removable.
> Under a matched one-pass MiniPile protocol, architecture-wide interventions
> produce informative validation-loss--logical-opportunity frontiers through
> Pythia-70M, while a preplanned Pythia-410M promotion exposes the dependence of
> those frontiers on the training-token regime.

The argument should proceed as follows:

1. Local activation sparsity percentages are insufficient because different
   sites feed operations with different reuse and output widths.
2. `R_model` provides exact, count-first model-wide zero-product opportunity;
   `R_model_max` describes topology- and architecture-conditioned analytic
   reach. Neither is measured speedup.
3. A topology-aware intervention ladder tests whether expanding from FFN sites
   to branch, attention-output, and post-RoPE attention-operand sites expands
   the attainable quality--logical-opportunity frontier.
4. The clean 14M A4-to-A7 comparison and the selected 14M/70M trained frontiers
   provide the main empirical evidence.
5. The 410M promotion is a preplanned boundary test: high-dose A7-over-A4
   ordering persists, but baseline quality and the complete dose-response do
   not. The fixed one-pass budget is a plausible explanation, not an identified
   cause.

`Sparsity spillover` may remain as a secondary motivation, but it should mean a
site-specific nonlocal activation redistribution. The current evidence does
not support a general statement that unpressured attention distributions all
broaden or all lose near-zero mass.

## Claim audit

| Candidate claim | Evidence status | Paper disposition |
| --- | --- | --- |
| Local FFN pressure produces a uniform opposing attention response | Not supported as written. `v`, Q/K, and post-`W_o` responses are mixed and non-monotonic. | Replace with a scoped, descriptive site-specific redistribution statement; do not headline it. |
| `R_model` converts actual zero operands into a model-wide logical-product fraction | Defined, implemented, and count-reconciled across the executed runs. | Keep as a central contribution, with the logical-opportunity and no-speedup caveats. |
| `R_model_max` exposes topology-, architecture-, and sequence-dependent reach | Analytic and verified against the declared topology/workload contract. | Keep as a central contribution. |
| Clean A7 attention-site expansion extends A4 opportunity | Tentative one-seed Pythia-14M Finding F002; dose-dependent quality cost. | Keep, precisely scoped; prioritize replication. |
| Four-site OL1 improves A4 for all `kappa <= 0.1` | False for corrected Run 015. Dual-axis improvement occurs only at `kappa=0` and `0.01`; `0.05` and `0.1` incur loss costs. | Correct the introduction and any derived prose. |
| OL1 generally improves endpoint quality | Not supported consistently across lambda, threshold, and topology. | Present its operational non-opposition geometry and trust bound; do not claim general endpoint superiority. |
| The complete trained frontier persists monotonically with model size | Refuted by the 410M baseline inversion and dose-response reversal. | Claim persistence through 70M and partial high-dose persistence at 410M only. |
| Larger `R_model` is measured runtime acceleration | Not tested. | Explicitly reject this interpretation throughout the paper. |
| Pythia-410M is undertrained | Plausible from exposure and trajectory evidence, but not established by Run 021. | Describe a fixed-token regime mismatch and an unresolved horizon hypothesis. |

## Results to place in the main paper

### 1. Architecture and accounting

Present the Pythia forward map, eligible sparsification sites, exact
zero-product accounting, and `R_model_max` by topology and architecture. Use
integer-count provenance and state that the metric is a logical opportunity,
not eliminated FLOPs or wall-clock speedup.

### 2. Site-specific response to local pressure

Show the directly pressured `h` response beside a restrained selection of
unpressured sites. Phrase the outcome as nonlocal, site-specific redistribution.
Do not merge the GeLU-to-ReLU change with the within-ReLU lambda trajectory.

### 3. Clean Pythia-14M intervention ladder

Present the A0/A1-H controls, post-hoc TEAL reference frontiers, and the clean
matched A4-to-A7 comparison. Clearly distinguish trained gates from
evaluation-only clipping and distinguish A4/A7 without OL1 from the selected
A4-OL1/A7-OL1 recipes.

### 4. Pythia-14M to Pythia-70M promotion

Use this as the main scale-persistence result. Report all selected thresholds,
not only favorable endpoints. State that it is a one-seed, two-scale
observation rather than a scaling law.

### 5. Pythia-410M fixed-token stress test

Give 410M a dedicated main-text subsection rather than hiding it or pooling it
uncritically with the other scales. Report:

- the A0 baseline inversion relative to 70M;
- equal total tokens and the 3.684 tokens-per-parameter exposure;
- the Run 021 LR-screen result;
- persistence of high-dose A7-OL1 dominance over A4-OL1;
- reversal of the zero-threshold ordering and trained dose-response;
- the unresolved longer-horizon hypothesis.

## Figure and table plan

Do not use one undifferentiated three-scale overlay as the sole headline
figure. Prefer:

1. Three aligned absolute frontier panels with common visual grammar. Label the
   410M panel explicitly as a fixed-token stress test and place each model's
   tokens-per-parameter value in its panel header or caption.
2. A companion within-size delta figure: TEAL points relative to their own
   `p=0` control and trained dose points relative to their own `kappa=0`
   endpoint. This shows persistence and reversal without concealing absolute
   baseline quality.
3. A compact baseline/recipe table containing parameter count, total tokens,
   tokens per parameter, A0 train loss, A0 validation loss, selected LR, and
   clipping incidence.
4. A compact cross-scale endpoint table for the matched A4-OL1/A7-OL1
   thresholds that the prose discusses.

Place the following in the appendix or supplement:

- every TEAL target and all sitewise activation tables;
- the complete 410M trained grid;
- A0 train-loss, global-gradient-norm, and learning-rate trajectories;
- OL1 conflict, projection, and trust-budget diagnostics;
- detailed counter reconciliation and analytic ceilings;
- any implementation-correction history needed for evidence provenance.

## Prioritized next work

### Phase 1: paper synthesis without new training

1. Reconcile `research/INDEX.md`, `research/MANUSCRIPT.md`,
   `manuscript/README.md`, and `manuscript/experiment-control/README.md` with
   completed Runs 019/021 and Analysis 011.
2. Create a new paper-facing analysis containing the baseline/exposure table,
   separate scale panels, within-size delta figure, compact cross-scale table,
   and corresponding observations.
3. Correct the stale result-bearing statements in `manuscript/introduction.tex`.
4. Draft Experimental Setup, Results, Limitations, and Discussion around the
   claim hierarchy above.
5. Freeze the primary empirical claim before selecting any confirmatory run.

### Phase 2: highest-value confirmatory compute

Prioritize seed replication over another model size. If the primary empirical
claim remains the clean A4-to-A7 topology expansion, the preferred design is:

```text
2 additional seeds x 2 topologies (A4, A7) x 5 kappas = 20 conditions
```

Use Pythia-14M, one full matched MiniPile pass, complete validation, and the
existing diagnostic contract. This would raise the clean result from one to
three seeds at comparatively low cost and avoid choosing thresholds after
seeing the first seed. This is a proposed direction only; it still requires
the repository's design and launch confirmations.

### Phase 3: optional model-scale work

Do not substitute Pythia-160M merely because Pythia-410M was unfavorable. If a
third well-behaved trained scale becomes a hard paper requirement, make the
post-result decision explicit:

1. run A0-160M alone under a predeclared health criterion;
2. proceed to the other eleven selected conditions only if A0 passes;
3. report the 410M promotion and the staged 160M decision in the paper.

Do not launch a longer 410M ladder directly. If resolving 410M becomes central,
start with one A0-only experiment. First choose which scientifically distinct
hypothesis is being tested:

- a second MiniPile pass tests additional optimization on the same corpus;
- genuinely new training tokens test data-budget scaling;
- a new longer schedule from initialization tests a different optimization
  recipe.

Only a predeclared successful A0 result should trigger consideration of the
remaining eleven longer-budget interventions. Simple upward LR retuning has
already been tested and should not be repeated.

## Manuscript discrepancies to correct during drafting

- `manuscript/introduction.tex` currently says that unregularized attention
  activation distributions broadly broaden and later says there are fewer
  near-zero attention values. The Run 004 observations support only a mixed,
  site-specific response.
- The introduction says corrected four-site OL1 improves both axes through
  `kappa=0.1`. Corrected evidence supports dual-axis improvement only at
  `kappa=0` and `0.01`.
- Manuscript control documents still say Pythia-410M is unobserved. Runs 019
  and 021 and Analysis 011 supersede that status.
- The selected cross-scale A4-OL1/A7-OL1 comparison changes both gate topology
  and OL1 pressure set. It is a comparison of complete recipes, not an isolated
  attention-gate effect. The clean no-pressure A4/A7 contrast presently exists
  only at 14M.

## Evidence references

- `analyses/011-2026-09-03-pythia14m-70m-410m-selected-ladder/README.md`
- `analyses/011-2026-09-03-pythia14m-70m-410m-selected-ladder/tables.md`
- `runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/RESULTS.md`
- `runs/021-2026-09-03-pythia410m-a0-learning-rate-screen-resolved-lr/RESULTS.md`
- `analyses/010-2026-09-01-pythia14m-vs-70m-selected-ladder/README.md`
- `analyses/009-2026-08-31-run012-vs-run015-a4-ol1-pressure-sites/observations/O001-h-only-vs-four-site-a4-ol1.md`
- `analyses/003-2026-08-30-run004-vs-run009-full-pass-l1-ol1/README.md`
- `runs/004-2026-08-29-pythia14m-full-pass-l1n/observations/O001-h-vs-site-near-zero-grid.md`
- `runs/004-2026-08-29-pythia14m-full-pass-l1n/observations/O002-h-vs-attention-output-near-zero.md`
- `research/findings/F002-a7-extends-a4-logical-opportunity.md`
- `research/METHODS.md`, `research/METRICS.md`, and `research/DATA.md`
