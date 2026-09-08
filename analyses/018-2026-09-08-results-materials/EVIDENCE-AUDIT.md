# Evidence scope and discrepancies

## What was admitted

| Evidence | Source | Role |
| --- | --- | --- |
| 14M A0, A1-H, four L1 doses | Run 004 | Controls and local pressure |
| Four 14M OL1 doses at h | Run 009 | Matched pressure-method comparison |
| Five 14M A4 doses | Run 011 | One-sided gate topology |
| Five historical A4 + OL1@h doses | Run 012, realization audit Analysis 009 | Explicit pressure-placement control |
| Five 14M A7 doses | Run 013 | No-pressure topology contrast |
| Five 14M A7-OL1 doses | Run 014 | Pressure at seven sites |
| Five corrected 14M A4-OL1 doses | Run 015 | Pressure at four sites |
| Twelve trained conditions each at 70M/410M | Runs 018/019 | Selected-recipe transfer |
| 150 14M post-hoc clipping evaluations | Analysis 006 raw combined artifact | Complete retained A0/A1/A4 clipping comparison |
| Twenty clipping evaluations each at 70M/410M | Runs 018/019 raw TEAL artifacts | Full controls and complete numeric appendix |
| Kernel progress and final ablations | Run 029 matched report and final r03 PDF | Existing systems subsection |

Analysis 013 supplies the exact endpoint identities and prior provenance;
the current reduction reopens each of the 59 raw logical diagnostics,
activation diagnostics, manifests, and configs. It checks complete coverage,
integer pooling, site-layer reconstruction, RMS, same-size initialization and
schedule hashes, random initialization, seed 1234, 712 boundaries, and total
tokens. All 190 clipping points are read from raw evaluation artifacts and
their counts and coverage are checked. All direct numerical inputs are hashed.
The larger clipping values are also reconciled against Analysis 011.

Training is random Pythia pretraining, using FP16 dynamic loss scaling and
FP32 parameters. The repository's general BF16 reference recipe is not the
executed full-pass recipe. The generated protocol table comes from actual
per-attempt configs. Model heads receive no numerator credit. The analytic
ceilings are recomputed with integer per-sequence counts and compared against
each source diagnostic, then multiplied by 338 to match measured denominators.

## Differences that affect presentation

1. **Historical pressure identity.** Run 012's manifest declares four sites,
   but the executed capture was h only. This package labels it A4-OL1[h],
   retains the declared metadata as historical provenance, and cites the
   already-established Analysis 009 realization audit. Run 015 supplies
   corrected four-site A4-OL1. Discarded F001 stays discarded.
2. **Loss pairing.** Some run README tables pair a terminal SDPA loss with a
   separate eager logical pass. This package uses loss and R_model from the
   same eager pass; differences in the final decimals are expected. The
   terminal loss remains in JSON. In particular, A7's kappa=.1 pressure
   difference is +.000819 here versus +.000816 in the older terminal table.
3. **Old p=0 display anchors.** Analysis 006's plotting reduction substitutes
   some trained anchors at p=0. This package instead reads the actual raw
   sweep evaluation. Tiny numerical differences can place both p=0 and its
   unchanged trained checkpoint on an exact mathematical frontier. They are
   not claimed as meaningful improvements.
4. **Source-document chronology.** Run 018's README opening is a preserved
   preflight status, while its later completion records and terminal artifacts
   establish completion. Parts of the general manuscript integration notes
   still describe the old spillover framing or 410M as unobserved. The dated
   current draft and completed Runs 018/019 are authoritative for this task.
5. **Notation.** Current draft S_model / S_model^max correspond to operational
   R_model / R_model_max. Fractions in JSON become percentages in figures.
   Figure 07 retains the old R notation with an explicit caption crosswalk.
6. **Diagnostic ports and available distributions.** Source activation passes
   retain exact counts and abs(x)<=.001/.01 counts, not dense histograms.
   The 14M A0 source has six diagnostic sites, lacking a and pre-Wo z; a
   separate clipping pass cannot fill their missing near-zero distributions.
   The six-site grid uses only common coverage. Post-Wo attention_output is
   explicitly distinct from z. Activation and logical measurements are
   separate full-validation passes and are not forced to be bitwise equal.
7. **Ceiling semantics.** U_arch has all observed zeros in its numerator;
   U_reach restricts it to selected reachable operation families. The latter
   is a supplementary sensitivity measure, not an operational redefinition.
   A0 normalization is null/undefined; no values are clipped at 100%.
8. **Zero-threshold controls.** No-pressure A4 and A7 have identical
   mathematical gates at kappa=0 but non-identical trained numerical outcomes.
   Their small residual is disclosed, not interpreted as an identity-gate
   treatment benefit. A1-H ReLU and G+,0 also differ in derivative at zero.

## Excluded and non-identifying evidence

- Runs 001-003 and 005-008/010 are calibrations or shorter/different-budget
  pilots. They are not pooled with the seed-1234 712-boundary study.
- Runs 016/017 failed initialization portability checks before valid science;
  Run 018 is their completed replacement. Run 020's LR implementation issue
  was corrected by Run 021; neither replaces the frozen Run 019 cohort.
- The 70M/410M cohorts lack unpressured A4/A7, local L1/OL1 dose grids, and
  historical h-only-pressure comparisons. They cannot replicate the full
  14M factorial/pressure story. No A7+OL1@four-sites control exists.
- Runs 022-024 document the original exact-ELL path's limitations: 14M raw
  correctness failure and no Pythia 70M/410M break-even on the qualified H100
  path. Runs 025-028 document adaptive systems development with differing
  hardware/baselines. Run 029 is the matched 14M retrospective used here.
  These histories are not independent agent trials or a common scaling curve.
- No raw activation resampling, histogram reconstruction, training, new
  inference, cloud action, seed bootstrap, or model-size scaling fit was run.
- Every scientific endpoint covers all 500 validation documents packed into
  338 blocks, 692,224 inputs and 691,886 prediction targets; 1,444 tail tokens
  are excluded. Thresholds and blocks are not independent training seeds.
- No confidence intervals or p-values are inferred from these one-seed grids.
  Frontier selection and threshold inspection reuse this validation set;
  there is no untouched confirmatory holdout in the retained protocol.

The current analysis is descriptive manuscript material authorized by the user;
the finding registry and the manuscript sources are unchanged.
