# Recovered h-only control measurements: incomplete cohort

## Question and method

Run the matched Pythia-14M A7 gate topology with orthogonal L1 pressure only
at h, at five thresholds. The scientific protocol and source identities are
unchanged from launch. This observation reports recovery status and measured
endpoints; it does not answer the high-threshold comparison while kappa=0.5
is inaccessible.

## Coverage and checks

Four recovered manifests report 712 completed updates and 1,493,172,224 input
tokens each, with no overflow-skipped updates. Initialization, data order,
source identity, all six h captures, OL1 geometry, full validation coverage,
and diagnostic integer counts pass the original checks. Each final validation
uses all 500 documents through 338 complete 2,048-token blocks, excluding the
1,444-token tail. Local transfer inventories remain incomplete except at 0.01.

## Results and legend

Loss is final-checkpoint ordinary validation cross-entropy in nats. R_model
is a logical scalar-product opportunity, expressed as a percentage; it is not
a measured speedup. Missing bytes below count required inventory files, not
partially downloaded archive bytes.

| Kappa | Validation loss | R_model (%) | Local checkpoint state | Missing inventory bytes |
|---:|---:|---:|---|---:|
| 0 | 5.219827 | 8.399380 | Final and recovery states verified; steps 2 and 4 absent | 112558688 |
| 0.01 | 5.198102 | 8.856062 | Entire inventory passes original local verifier | 0 |
| 0.05 | 5.194959 | 10.126069 | Final and most intermediate states absent | 900635221 |
| 0.1 | 5.237395 | 10.903597 | Final and later recovery states absent | 450400469 |
| 0.5 | pending | pending | Host cannot restart; data retained on stopped volume | unknown |

## Caveats and provenance

The full cohort verifier and three-way reduction have not run. There is no
high-threshold sufficiency/necessity conclusion, manuscript update, or promoted
finding. One seed cannot establish variability or a general effect. Failure
of h-only pressure would not isolate direct Q/K/V necessity because the all-site
arm also adds pressure at a, m, and z.

Sources: the original per-attempt manifests, metrics, events and diagnostic
files under `artifacts/attempts/`; `prelaunch/salvage-receipt.json`; and
`prelaunch/recovered-metadata-audit.json`. Generating recovery/audit scripts:
`16_recovery_metadata.py`, `17_salvage_downloads.py`, and
`18_audit_recovered_metadata.py`. Infrastructure and spending limitations are
recorded in `prelaunch/infrastructure/004-controller-interruption-recovery/`.
