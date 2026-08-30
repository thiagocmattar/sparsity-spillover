# Observation Index

| ID | Statement | Status | Figure/source |
| --- | --- | --- | --- |
| O001 | Run 004 naive L1 and Run 009 OL1 trace closely matched `R_model` endpoints, while OL1 has lower final validation loss at lambda 0.05, 0.1, and 0.5 but higher loss at lambda 1.0. | descriptive; no finding promoted | `O001-r-model-vs-final-validation-loss.md`; `../figures/01-r-model-vs-final-validation-loss.pdf` |
| O002 | Higher-lambda naive-L1 trajectories have more frequent and more negative raw task-pressure interaction, but the combined raw gradient remains task-aligned; OL1 enforces adaptive-direction non-opposition rather than loss safety. | descriptive; no finding promoted | `O002-gradient-interference-and-ol1-geometry.md`; `../gradient_tables.md` |
