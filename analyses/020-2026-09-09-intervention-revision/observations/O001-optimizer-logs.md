# O001: optimizer geometry and retained boundaries

Question: when does the executed OL1 cap saturate, and what do training norms mean?
Method: CPU reduction of original manifests/configs/events; source identities verified.
Coverage: 54 primary conditions; 38,447 readable train records. The first 14M A1-H
record contains 4,055 NUL characters and is not parsed; it is outside all OL1 and
selected dynamics cohorts. All 34 OL1 and all nine dynamics curves are complete.
Legend/caption: each condition has all/early/late cap counts; ratios precede LR on
eligible parameters. Detailed values and proof: manuscript/draft/revision-v2/ol1-audit.md.
Result: 7,283/24,208 OL1 boundaries bind, with strong recipe and size dependence.
Caveats: global rescaling invariance is conditional at fixed state; no new weight
sweep, training or checkpoint evaluation. No unknown training record is imputed.
Source script: ../01_audit_logs.py; data/log-audit.json and ol1-steps.csv.
Dynamics interpretation and figures are completed in the next batch.
