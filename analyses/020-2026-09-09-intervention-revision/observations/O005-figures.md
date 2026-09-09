# O005: intervention and execution displays

Question: which cross-size trade-offs and execution increments are visible together?
Method:03_figures.py reads the verified FP16 CSV, raw nine-curve training log
reduction and Analysis019 runtime audit. No new evaluation/timing.
Coverage:96 cross-size evaluations repeated in two rows,30 runtime checkpoints,
9x712 raw training records. PDF sources and omission IDs:figures/SOURCES.json.
Legend/caption:existing family colors/markers; open post-hoc versus filled
trained points. Runtime pairs show absolute implementation latency against
the same checkpoint's native BF16 reference loss, checked equal across source
replicates; this does not assert exact candidate-loss equality. Sparse factor
is a ratio of separately native-normalized speedups.
Result:restored main cross-size display,full-range appendix,raw dynamics grid,
and quality/latency plus sparse-factor panels replace cohort means. At unpressured
A4/A7 kappa=.5,latencies .478973/.478926ms correspond to reference losses
5.662305/5.707563. Their native baselines differ (.739422/.844909ms).
Caveats:no equivalence test,paired implementation timing experiment,or
quality-matched cross-model speedup is inferred. Full ranges preserve tails.
Source script:../03_figures.py; original runtime verification:Analysis019/02_audit_runtime.py.
