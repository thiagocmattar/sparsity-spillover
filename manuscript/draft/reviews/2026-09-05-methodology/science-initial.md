# Scientific review — first methodology draft

5 September 2026. Reviewed `methodology.tex`, `methodology-appendix.tex`,
`main.tex`, and `methodology-notes.md` against the source audit and current
implementation. This is a review of methodology prose and mathematics, not a
claim that the deferred experimental-setup or results sections are complete.

## Rubric

Each criterion is scored 1–5: **1 = blocking**, **2 = major revision**,
**3 = competent but uncompetitive**, **4 = strong**, **5 = exceptional**.
Equal weights, maximum 25. No score is an acceptance probability.

| Criterion | Score | Assessment |
| --- | ---: | --- |
| Operational fidelity | 4/5 | The text correctly specifies post-gate equal-tensor pressure, fixed-gate equality and gradients, the ReLU-at-zero distinction, task clipping, task-only AdamW moments, updated-moment preconditioning, unclipped pressure, conflict-only stabilized projection, mandatory budget, and update order. It does not copy the old unsupported AMSGrad or optional-budget statements. One small scope qualification to matching would improve precision. |
| Metric and ceiling precision | 4/5 | Scalar multiplications, six block families, dense-head denominator-only treatment, valid causal counts, natural zeros, and fractional versus percentage units are correct. The ceiling caveat is explicit. Two wording/indexing details should be tightened: the first architectural-reach sentence and whether attention count formulas are per block or global. |
| Intervention identifiability | 5/5 | Gate and pressure sets are independently declared, tensor weighting changes with the target set, the A4/A7 objective confound is named, and the ladder is explicitly not a fully crossed design. These are unusually clear safeguards against misreading a recipe comparison as isolated gate placement. |
| Reproducibility split between main and appendix | 4/5 | The main section is compact but preserves the choices and estimands needed to read later results. The appendix has the exact operational rules and validation coverage. Condition-specific settings are transparently deferred rather than fabricated. Explicit scope on the attention formulas would make the formal appendix easier to reproduce directly. |
| Claim rigor for later results | 5/5 | The draft distinguishes logical opportunity, selected-site reach, acceptable-quality sparsity, and hardware timing. It explicitly rejects an all-observed-zero upper bound and a task-loss guarantee. Candidate conditions and larger-size analytic ceilings are not presented as executed experiments. |
| **Total** | **22/25** | **Strong methodology draft; no blocking scientific error.** |

## Ranked fixes

### 1. Correct the absolute reach sentence

**Quote:** “The selected sites determine which operations can receive zero
operands.” [`methodology.tex`, line 68](../../methodology.tex)

Taken literally, this excludes natural zeros in operations outside selected
reach. The surrounding paragraph later gives the correct qualification, so
this is a local contradiction rather than a wrong metric definition.
[`metrics.py`](../../../../src/sparsity_research/metrics.py) counts zeros
in all six block families; [`ceilings.py`](../../../../src/sparsity_research/ceilings.py)
separately determines selected-site reach.

**Fix:** “The selected sites determine which operations their all-zero limit
reaches.” Alternatively, begin directly with the next sentence defining the
all-zero reach ceiling. This should be corrected before handoff.

### 2. Give the attention counters an explicit block/batch scope

**Quote:** “For query position $t$, key position $u\leq t$, and within-head
coordinate $j$,” followed by formulas summing over b, h, t, u, and j but no
layer index. [`methodology-appendix.tex`](../../methodology-appendix.tex)

The formulas match `qk_zero_product_counts` and `pv_zero_product_counts` for a
single block invocation. The global N0 also sums across layers and evaluation
batches. This can be inferred from the surrounding pooling rule, but the
formal scope should be explicit to prevent treating the displayed count as
the already-global numerator.
[`metrics.py`](../../../../src/sparsity_research/metrics.py),
[`logical_capture.py`](../../../../src/sparsity_research/logical_capture.py).

**Fix:** Add “Within one block and evaluation batch” before the formula setup,
and clarify that the resulting operation counts are summed over all blocks and
evaluation batches in N0. No longer indexed equation is necessary.

### 3. Scope initialization matching to same-size contrasts

**Quote:** “Matched contrasts share initialization, data order, and training
budget.” [`methodology.tex`, line 43](../../methodology.tex)

This is correct for the matched comparisons, but the nearby Pythia-family
framing could invite the impression that initialization is matched across
different parameter shapes. The study uses selected cross-size extensions,
which are a different comparison role. Analysis 013's README explicitly
separates paired contrasts from the cross-size evidence.

**Fix:** “Within each model size, matched contrasts share initialization, data
order, and training budget.” This is a useful precision improvement, not a
reason to add condition-specific experimental detail now.

## Formula verification

- The two fixed gates use >= and therefore preserve equality; detached masks
  produce derivative one for retained inputs. Standard ReLU differs only at
  zero when compared with the one-sided zero-threshold gate. Symmetric zero
  threshold is exact identity. This agrees with
  [`sites.py`](../../../../src/sparsity_research/sites.py).
- The L1 objective gives each targeted site/layer tensor equal weight, even
  when its width differs. Captures are post-gate, as verified in
  [`capture.py`](../../../../src/sparsity_research/capture.py).
- The OL1 formula includes the implementation's c<0 and q>epsilon trigger,
  c/(q+epsilon) coefficient, updated bias-corrected Adam moments, and correct
  r=0 branch of the budget. It specifies eligible parameters and excludes
  decoupled decay and group learning-rate multiplication from the geometry.
  It states approximate orthogonality rather than a nonexistent strict
  nonconflict guarantee. These agree with
  [`pressure.py`](../../../../src/sparsity_research/pressure.py) and
  [`optimization.py`](../../../../src/sparsity_research/optimization.py).
- Linear N=npq and N0=q times the number of zero inputs are correct. QK/PV
  logical OR counts zero-zero products only once. Future causal positions are
  excluded, while valid probability underflow zeros are counted.
- The one-sequence, one-block denominator
  T(4d²+2dd_f)+dT(T+1) is correct. The model denominator is L times this
  quantity plus Td times vocabulary size; multiplying by the number of
  sequences is correct. The dense head receives no numerator credit.
- Reach unions correctly include QK once for either Q or K, PV plus attention
  output for all-zero V, and attention output once for either V or z. The text
  explains why QKV biases and softmax prevent unsupported further propagation.
- A1-H, A4, and A7 reachable counts are correct for d_f=4d. A0 has zero
  selected reach while allowing observed natural zeros. The ceiling's gap is
  not equated with quality-preserving slack.
- Input positions versus next-token targets are correctly separated:
  338 × 2,048 inputs are counted, while each block contributes 2,047 causal
  prediction targets. The 1,444-token tail and 500-document coverage are stated.
- The S-model/S-block/S-max macros do not collide with gate/pressure sets.
  The appendix distinguishes V as values from the vocabulary symbol. The
  scalar q for a projection's output width and for an optimizer squared norm
  is locally scoped and does not create a substantive ambiguity. The overloaded
  B/b notation is conventional, though explicit counter scope helps.

## Does the brief main text preserve understanding?

Yes. A reader can identify the independently controlled intervention fields,
understand what naive L1 versus OL1 changes, see the critical A4/A7 comparison
limit, interpret the sparsity numerator/denominator, and distinguish analytic
reach from measured behavior. The appendix supplies the mathematical details
at the point where they become necessary. Reintroducing the entire Pythia
forward graph into the main text would not materially improve that understanding.

Condition-specific learning rates, budgets, seeds, training precision, and
checkpoint-selection rules remain required for the eventual complete paper,
but are correctly deferred to its experimental setup/results cohort. This
review does not require choosing or inventing those settings now.

Only this review file was modified by this reviewer.

