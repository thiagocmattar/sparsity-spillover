# First assessment of the 70M kernel gap

Requested on 22 September 2026: assess the repository and existing results for
efficient dense execution and useful sparse speedups at lower-loss 70M recipes,
especially T2/Ph kappa=.05/.1, with T7/Pall also in scope.

The [assessment](observations/001-assessment.md) identifies the measured h/z
fallback bottleneck, the existing efficient dense control, the mismatch between
the previous candidate-selection objective and this goal, and the missing
comparisons. Recommendations are proposals, not an approved experiment design.
No training, GPU execution, kernel modification or manuscript edit was performed.

Reproduce the CPU-only retained-evidence audit from the repository root:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/035-2026-09-22-70m-kernel-assessment/01_assess.py
```

[Machine-readable evidence](data/assessment.json) retains all 26 matched Run045
conditions, pooled and per-layer h/z occupancy, actual instruction/scalar counts,
the three available optimized component profiles and separate Run042 controls.
The audit checks 78 complete qualified process records, recomputes all 26
three-implementation timing aggregates, reconciles all 26 full-validation
occupancy records to integer element counts, and records 321 source hashes.
This verifies retained measurements; it does not rerun numerical qualification.

The later Run047 T2/Ph kappa=.5 result is cited separately from its source
observation and does not enter this matched Run045 reduction. Publication figures
are not needed for this assessment; tables and their interpretations are in the
observation. Existing manuscript edits and unrelated Run032 prelaunch files are
outside this change. The research index's stale next-run/analysis numbers are
updated from the directories actually present.
