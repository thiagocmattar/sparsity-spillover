# Methodology provenance and editorial scope

5 September 2026. The compact main methodology defines interventions,
model-wide sparsity, and selected-site reach. Its appendix contains general
gate, pressure, counter, and reach definitions. The separate experimental
section and appendix now contain Pythia recipes, architecture specialization,
configuration sources, and common evaluation coverage. This is
manuscript work; no experiment is designed or launched, and no result is added.

## Sources and operational fidelity

- Historical style/formal source: [../methodology.tex](../methodology.tex).
- Current definitions: [DEFINITIONS](../../research/DEFINITIONS.md),
  [METHODS](../../research/METHODS.md), [METRICS](../../research/METRICS.md), and
  [DATA](../../research/DATA.md).
- Gate and capture semantics: `src/sparsity_research/pythia.py`, `sites.py`,
  and `capture.py`; pressure geometry: `pressure.py` and `optimization.py`;
  integer counting: `metrics.py` and `logical_capture.py`; evaluation:
  `evaluation.py`; analytic reach: `ceilings.py`.
- The common packing coverage comes from the operational data contract.
  Training schedules, precision, seeds, and optimizer values remain tied to
  executed conditions and will be stated alongside the later results.
- [AdamW](https://arxiv.org/abs/1711.05101) was checked against its primary
  record for title, authors, and ICLR 2019 venue; its bibliography entry was
  added. Architecture, data, PCGrad, and Bloop records were checked in the
  introduction/related-work pass. The projection citations give context,
  not guarantees inherited by this implementation.

The new appendix corrects old-source omissions: task clipping before OL1's
task-only AdamW step, a required positive pressure budget, and the absence of
an implemented AMSGrad variant. It also states the post-gate, equal-tensor
pressure average and the stabilized projection's approximate orthogonality.
The old source and immutable scientific artifacts remain intact.

## Paper notation and units

| Paper notation | Existing artifact field / definition |
| --- | --- |
| `\mathcal{S}_{\mathrm{model}}` | `R_model`: pooled block zero products divided by block products plus dense LM-head products |
| `\mathcal{S}_{\mathrm{block}}` | `R_block`: same numerator divided by block products |
| `\mathcal{S}_{\mathrm{model}}^{\max}` | `R_model_max_fraction`: all-zero selected-site reach divided by the same model denominator |

All three are fractions; printed percentages multiply by 100. The metric's
atomic unit is a scalar multiplication, not a dot product or total FLOPs.
The observed numerator includes natural zeros in all six counted block
families. Selected-site reach does not bound every possible observed zero;
in particular, A0 can have positive observed sparsity and zero selected reach.

The paper uses `\mathcal{G}` and `\mathcal{P}` for gate and pressure site sets,
reserving calligraphic S for sparsity. Display notation is updated in the
introduction, current wrapper/caption, and ladder; the historical source,
review snapshots, operational keys, and result files are not renamed.

## Figure choice and deferred specifics

The existing ladder already compares ceilings for the pinned 14M, 70M, and
410M configurations. It is retained with the new notation rather than adding
a second display of the same values. A cross-family plot would require a
separate choice of architectures, workloads, and gate correspondences and is
not necessary for this compact section. The ladder is a map of alternatives,
not an execution-coverage table or a cumulative sequence of pressure additions.

A4-OL1 versus A7-OL1 changes both the gate set and pressure objective. The new
main text preserves this interpretive boundary. The appendix does not invent
a universal training schedule before the results cohort is selected.

Independent source audits and scored reviews are retained under
[reviews/2026-09-05-methodology/](reviews/2026-09-05-methodology/README.md).

## Architecture pin cross-check

The appendix configuration hyperlinks use the identities recorded in:

- [Run 004 config](../../runs/004-2026-08-29-pythia14m-full-pass-l1n/config.yaml):
  `EleutherAI/pythia-14m-deduped`, revision
  `7386d9a4ae45aef494a6e704910394def3037fc5`.
- [Run 018 config](../../runs/018-2026-09-01-pythia70m-selected-ladder-canonical-init/config.yaml):
  `EleutherAI/pythia-70m-deduped`, revision
  `e93a9faa9c77e5d09219f6c868bfc7a1bd65593c`.
- [Run 019 config](../../runs/019-2026-09-01-pythia410m-selected-ladder-canonical-init/config.yaml):
  `EleutherAI/pythia-410m-deduped`, revision
  `b5e8535141902c0e985cea61fd02afe7fe86af32`.

The source audit independently matched the pins to Run 013/018/019. The
410M immutable configuration was also fetched from Hugging Face and its
architecture dimensions verified. Web-tool retrieval of the 14M and 70M
immutable pages failed; those identities are verified from local run records.
The configuration links specify architectures, not a released training recipe.

All twelve architecture/topology numerator and denominator pairs were checked
against `architecture_ceiling`; the recorded exact integers and fractions are
in [ceiling-verification.json](reviews/2026-09-05-methodology/ceiling-verification.json).
No measured results were recomputed.

The final visual follow-up also enlarged architecture labels and compacted
the ladder to preserve legibility at manuscript width. This changed typography
and spacing only; the exact twelve ceiling pairs remain unchanged.

## Separation from the experimental study

The user-authorized structural revision moves all A* recipe names, concrete
ports, model/data context, matched-comparison logic, and the combined setup
figure out of main Methodology into [Experimental Study](experimental-study.tex).
Pythia-specific counts and configuration links now appear in
[experimental-appendix.tex](experimental-appendix.tex); general equations retain
their labels. [Experimental notes](experimental-notes.md) record coverage and
comparison provenance. The four existing numbered equations, figure artwork,
ceiling values, introduction, related work, and bibliography are unchanged.
