# O004: late training dynamics

Question: do retained loss and task-gradient trajectories support inspecting a finite budget?
Method: raw nine-curve plot with learning rates; fixed final20% window; descriptive OLS.
Coverage: nine complete712-boundary logs, 6,408 records, no skipped steps.
Legend/caption: columns A0/A4-OL1/A7-OL1(kappa=.5), colors sizes, rows raw task
loss/pre-clip AMP-unscaled global task-gradient norm/LR. Shading=steps570..712.
Result:410M A0 remains improving and has a larger late norm than smaller A0s;
410M high-threshold sparse recipes also keep improving. Detailed numbers/semantics:
manuscript/draft/revision-v2/training-dynamics.md and data/log-audit.json.
Caveats: two retained validation events only; no late validation trend or
causal proof of undertraining. No continuation or new measurement.
Source scripts: ../01_audit_logs.py and ../03_figures.py.
