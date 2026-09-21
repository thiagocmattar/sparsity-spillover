# Paper-to-code and evidence map

This map follows section titles and figure filenames, so it remains useful if
the paper's float numbers change. The source is the 21 September 2026 working
draft under manuscript/draft in the author repository.

| Paper location / result | Runnable implementation | Retained evidence |
|---|---|---|
| Methodology: pressure P and threshold topology T; Table 1 | `training/config.py`, `src/sparsity_research/sites.py`, `pythia.py`, `pressure.py` | `configs/original/`, `configs/paper-grid.json` |
| Sparsity accounting and architectural reach; Appendix B/C | `ceilings.py`, `logical_capture.py`, `metrics.py`; `reproduce.py results` | Integer numerator/denominator in `results/endpoints.json` |
| Introduction: quality/sites/quality figure (24-*) | `reproduce.py figures` | `endpoints.json`, `operation-latency.json` |
| Extensive 14M ablations; quality/latency (22-*) | `training.train` for 14M conditions; `benchmark.py` K050 | `14m-figure.json`: 41 displayed points; 45 endpoints include four L1-only appendix settings |
| Spillover: local pressure and layer sparsity | `training.evaluate`; `capture.py` | `pressure-placement.json`: all 30 T4/T7 rows, seven sites, six layers |
| Conditional execution savings; Table 2 | final 14M kernels plus retained ablation results | `operation-latency.json`: h ≈150.1 μs, z ≈29.0 μs, PV ≈−6.2 μs; process spans, not confidence intervals |
| Narrow T2/Ph; loss 5.151 and 0.573 ms at κ=.1 | `14M-T2-Ph-0.1`, HZ gate registration | `endpoints.json`, `14m-figure.json` |
| 70M quality/latency (23-*); original port versus optimized | 70M conditions; `benchmark.py --backend port/optimized` | 26 Run045 points plus separate Run047 T2/Ph κ=.5; `70m-sessions.json` |
| 70M specialized h,z conditional benefit, 17.4% | `benchmark.py --control native-hz` versus optimized on the same checkpoint | `70m-controls.json`; separate session, not pooled with the grid |
| Appendix: L1 versus OL1 and budget saturation | `training/optimizer.py`, `training/ol1.py`; per-boundary events | `ol1-geometry.json`: 712-step traces; T7/Pall cap active 96.99% versus Ph 0.25% |
| Base training trajectories (19-*) | `training.train`, events.jsonl | `base-training.json` |
| Complete endpoint tables | `reproduce.py results`, all IDs from `training.train --list` | `endpoints.json`: 84 rows with ordinary final loss, count-pooled sparsity, identities |
| Post-hoc clipping, 14M appendix (21-*) | `scripts/clip.py`; ten training-block calibration | `clipping.json`: 30 checkpoints × ten targets; no fabricated HZ/Ph sweeps |
| 410M fixed-token stress test (A5-*) | 410M training configs; full diagnostics | 12 endpoints and Base/ReLU clipping in `endpoints.json` |
| Appendix: scalar sparsity, MMA bypass, speedup (20-*) | exact logical counters; final CUDA counters | `kernel-structure.json`; instruction padding and scalar substitution remain distinct from logical counts |

## Naming crosswalk

| Paper | Code topology | Gate sites | Pressure |
|---|---|---|---|
| Base / T0/P0 | A0 | none (stock GELU) | none |
| ReLU / T1/P0 | A1-H | h, ReLU replacing GELU | none |
| T1/Ph | A1-H | h, ReLU | h, OL1; L1 has a separate ID |
| T2/Ph | HZ | h,z, one-sided threshold | h only |
| T4 | A4-Z | a,m,h,z, one-sided threshold | P0, Ph, or all four |
| T7 | A7-Z-POST | T4 plus symmetric q_post,k_post,v | P0, Ph, or all seven |

Training provenance: 14M Base/ReLU/L1 Run004, OL1 T1 Run009; T4/P0 Run011;
T4/Ph Run012 (realized h-only, despite its old declaration); T4/Pall corrective
Run015; T7/P0 Run013, T7/Pall Run014, T7/Ph Run032; HZ Runs041/044.
70M controls/Pall Run018, Ph Run034, HZ Runs043/046. 410M Run019.
The compact release does not include these historical folders.

## Scope of reproduction

Central numerical results and plots can be reconstructed from compact evidence.
New training/evaluation and final-kernel measurements have runnable entry points.
The adaptive kernel search trajectory, rejected candidates, and old cloud
orchestration are deliberately excluded. The six-path Table 2 controls live in
`kernels/ablation14m/`; `benchmark.py --operation-mode` reruns them with complete
qualification. `--diagnostics` retains full-validation actual-operand and MMA
counters for 14M. The 70M native-h,z replacement control is also included.
