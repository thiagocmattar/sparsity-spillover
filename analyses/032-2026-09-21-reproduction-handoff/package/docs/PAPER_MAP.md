# Paper-to-code map

Read [main.pdf](../main.pdf) for the complete manuscript and appendices. This map
uses section titles and result descriptions so it survives float renumbering.

| Paper topic | Code / command | Evidence |
|---|---|---|
| Methodology: pressure and threshold topology; Table 1 | `training/config.py`, `src/sparsity_research/sites.py`, `pressure.py` | `configs/paper-grid.json` |
| Logical sparsity and architectural reach | `ceilings.py`, `logical_capture.py`, `metrics.py`; `reproduce.py results` | Integer counts in `results/endpoints.json` |
| 14M quality and latency | `training.train`; `scripts/benchmark.py` | `14m-figure.json`: 41 displayed points; four additional L1 settings in `endpoints.json` |
| Local pressure and sparsity spillover | `training.evaluate`, `capture.py` | `pressure-placement.json`: 30 T4/T7 settings, seven sites, six layers |
| Conditional execution savings; Table 2 | `kernels/ablation/`; `benchmark.py --operation-mode` | `operation-latency.json`: all six paths and process ranges |
| Narrow T2/Ph; loss 5.151 and 0.573 ms at κ=.1 | `14M-T2-Ph-0.1` | `endpoints.json`, `14m-figure.json` |
| 70M quality and final specialized latency | 70M conditions; `kernels/model_70m/` | 26 main-session settings plus the separate T2/Ph κ=.5 endpoint |
| 70M h,z conditional benefit, 17.4% | `benchmark.py --control native-hz` versus final execution on the same checkpoint | `70m-controls.json`; separate session |
| L1 versus OL1; budget saturation | `training/optimizer.py`, `training/ol1.py`; boundary events | `ol1-geometry.json`: 712-update traces |
| Base training trajectories | `training.train`; boundary events | `base-training.json` |
| Complete endpoint tables | `reproduce.py results`; `training.train --list` | `endpoints.json`: 84 settings at 14M/70M/410M |
| Post-hoc clipping | `scripts/clip.py` | `clipping.json`: 30 × 10 14M settings; Base/ReLU scale controls in `endpoints.json` |
| 410M fixed-token stress test | 410M training conditions; full diagnostics | 12 trained endpoints; no specialized 410M kernel |
| Scalar zeros, MMA bypass and speed | `benchmark.py --diagnostics`; `kernels/ablation/diagnostics.py` | `kernel-structure.json` |

## Condition names

`14M-T2-Ph-0.1` means 14M parameters, two gated sites, h-only pressure, κ=.1.
For T1, the final number is the pressure weight instead of a gate threshold.

| Paper topology | Gate sites | Internal topology key |
|---|---|---|
| T0 | none; stock GELU | A0 |
| T1 | h; ReLU replaces GELU | A1-H |
| T2 | h,z; one-sided threshold | HZ |
| T4 | a,m,h,z; one-sided threshold | A4-Z |
| T7 | T4 plus symmetric post-RoPE q,k and v | A7-Z-POST |

P0 means no pressure, Ph means h-only OL1, Pall means OL1 at every active gate,
and L1 identifies ordinary L1. Internal keys preserve checkpoint compatibility.

## Release boundary

The code contains the final 14M and 70M implementations and published execution
ablations. It excludes discarded candidates, optimization history and the
superseded 70M port. The port's recorded comparison values remain in the results;
the supported GPU rerun path uses the final implementations and native baselines.
No private run folders or manuscript TeX sources are needed.
