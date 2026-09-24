# Reviewer guide

This supplement supports **Activation Sparsity as Training–Inference Co-Design**.
The paper is submitted separately; this archive contains its supporting code,
evidence and figure assets. Its prose may evolve independently. No account,
GPU, weights or dataset download is needed to inspect or reconstruct the results.
Ready-to-read CSV tables are also included in `results/` and can be opened
without Python. Python dependencies must already be installed for offline
execution; see README.md.

## A short review route

1. Read the separately submitted paper and `docs/PAPER_MAP.md`.
2. Run `python scripts/reproduce.py verify` to check all distributed SHA-256 hashes.
3. Run `python scripts/reproduce.py results` to reconstruct CSV tables, logical
   ceilings and the current paper's numerical checks.
4. Run `python scripts/reproduce.py figures` to reconstruct ten PDF views.
   `reproduced/scale-frontier.pdf` uses the paper's Figure 5 drawing code and
   exact 36 coordinates. The six original figure assets are in `figures/`.
5. Run `python -m pytest -q` to check pressure mathematics, hook placement,
   validation coverage, serialization, and portable kernel assembly on CPU.

## What the evidence establishes

| Claim or comparison | Inspect | Check |
|---|---|---|
| 14M quality, logical opportunity and runtime differ | Figure 1; `14m-paper-figure.json` | Same 40 checkpoint keys in the quality and latency panels; pooled integer counts |
| Pressure can change zeros outside its target | Figure 3; `pressure-placement.json` | Per-site/per-layer zero and total counts; fixed color scale |
| OL1 projects and caps a separate pressure update | Figure 4; `ol1-geometry.json`; `training/ol1.py` | Boundary traces and mathematical tests |
| T2/Ph and T7/Ph occupy different regimes at 14M/31M/70M | Figure 5; `scale-frontier.json`; `scale-evidence.json` | 33 checkpoints, 36 executions, two Base backends, five thresholds per recipe and size |
| New 31M points passed the numerical gate | `31m-timing-processes.json.gz` | All 33 processes, 338-block qualification rows, and 29,568 raw host/event timing samples |
| 14M T2 h,z execution saves 12.9% versus switches off | `14m-t2-execution-controls.json` | Same checkpoint and unchanged zeros/logits; three process summaries; 57.9% of its native-to-enabled gap |
| 70M h,z implementation comparisons have different denominators | `70m-controls.json` | 58.3% versus custom skips-off; 17.4% versus native h,z substitution |
| The fixed token budget has different optimization implications by size | Figure 6; `base-training.json` | 712 updates at each of four sizes; 2,848 retained update records |

File names in this table are relative to `results/`. CSV outputs retain explicit
units and backend names. In `31m-results.csv`, the Base latency is PyTorch Base,
matching the manuscript's delta table. In `endpoints.csv`, latency is the
specialized kernel. `scale-frontier.csv` lists both Base executions explicitly.

## Boundaries of the claims

All training comparisons use one seed and one fixed token budget. The 410M
conditions support an optimization stress test, not a fourth measured latency
scale. Cross-scale and cross-recipe timing points come from distinct recorded
sessions; they are not paired causal skip interventions. Process ranges are not
confidence intervals over training seeds. Only the named same-checkpoint
controls isolate execution changes. There is no 31M same-checkpoint skip-control
experiment in this release and no fitted scaling law.

Full checkpoints, recovery state and token caches are excluded from this ZIP.
Hashes identify the historical artifacts, but no public download URL is provided.
Thus the supplement supports immediate numerical review and recipe replication;
an exact historical weight replay additionally requires those external artifacts.
The source is refactored for portability and CPU checked. Its CUDA arithmetic
is retained, but the portable assembly has not been requalified on a GPU.
