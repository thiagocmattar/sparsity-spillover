# Exact paper figure assets

These PDFs retain the figure assets for this evidence snapshot. The paper is
submitted separately; its captions and numbering may evolve during editing.
`docs/PAPER_MAP.md` links each topic to its data and executable reconstruction.

| Asset | Paper use | Evidence / source |
|---|---|---|
| 01-quality-savings-sparsity.pdf | Figure 1, 14M quality/opportunity/latency and six conditional effects | `results/14m-paper-figure.json`, `operation-latency.json`; `scripts/reproduce.py` |
| 02-architecture.pdf | Figure 2, site and execution-path diagram | Site definitions in `docs/METHODS.md`, `training/config.py`, `src/sparsity_research/sites.py`; explanatory artwork |
| 03-layer-sparsity.pdf | Figure 3 uses page 1; page 2 is the retained higher-threshold companion | `results/pressure-placement.json`; `scripts/reproduce.py` |
| 04-ol1-geometry.pdf | Figure 4, 14M OL1 geometry | `results/ol1-geometry.json`; `scripts/reproduce.py` |
| 05-scale-frontier.pdf | Figure 5, Base/T2/Ph/T7/Ph at 14M/31M/70M | `results/scale-frontier.json`; `scripts/plot_scale.py` |
| 06-base-training.pdf | Figure 6, four Base training trajectories | `results/base-training.json`; `scripts/reproduce.py` |

Figure 5 includes 36 execution points from 33 checkpoints. Colors distinguish
T2/Ph and T7/Ph, shapes distinguish sizes, and filled versus open Base markers
distinguish PyTorch versus kernel execution. Vertical guides mark each size's
Base loss. Connected threshold points describe the measured recipes; they are
not a scaling-law fit. The second absolute/relative companion panel considered
during analysis is not part of the paper and is not distributed here.

Ten regenerated views are written to `reproduced/`, never over these assets.
Except for Figure 5's retained drawing code, the reconstructions are numerical
views rather than replicas of the final typography. Source and release hashes
are in PROVENANCE.json and MANIFEST.json.
