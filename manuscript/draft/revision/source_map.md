# Authoring and evidence inventory

All paths are repository-relative. AVAILABLE means the file exists and its
structure has been inspected; detailed numerical/source-hash verification is
recorded separately in the numerical ledger. The untouched baseline compilation matched all 27 source-PDF pages in text and
rendered pixels. The revised 30-page verification is in verification.json.

## Authoring sources

| Role | Actual files | Status |
|---|---|---|
| Entry point/template | `manuscript/draft/main.tex`, `iclr2027_conference.sty`, `.bst`, `natbib.sty`, `fancyhdr.sty` | AVAILABLE; baseline compiled |
| Main prose | `abstract.tex`, `introduction.tex`, `related-work.tex`, `methodology.tex`, `experimental-study.tex`, `training-results.tex`, `kernel-autoresearch.tex`, `conclusion.tex` under `manuscript/draft/` | AVAILABLE; paragraph map records 45 RW tasks |
| Appendices | `methodology-appendix.tex`, `experimental-appendix.tex`, `results-appendix.tex` under the draft | AVAILABLE |
| Bibliography | `manuscript/draft/references.bib` | AVAILABLE; 15 resolved citations in baseline |
| Figure copies/provenance | `manuscript/draft/figures/`, `figures/SOURCES.json` | AVAILABLE; 15 retained PDFs, 11 embedded in the revised draft |
| Table fragments | `manuscript/draft/tables/`, `tables/SOURCES.json` | AVAILABLE; twelve generated fragments in the revised draft |
| Supporting records | `manuscript/draft/supplementary-data/` and `SOURCES.json` | AVAILABLE; 14 original measurement copies, three audit summaries and protocol |

## Figure and table generators

| Manuscript visual | Original generator / source | Revised owner |
|---|---|---|
| Figure 1 architecture/recipe map | `manuscript/artifacts/pythia-architecture-map.tex`, `sparsification-ladder.tex`, `pythia-architecture-sparsification-ladder.tex` | Analysis 019; preserve graph and analytic counts |
| Figure 2 14M overview | `analyses/018-2026-09-08-results-materials/04_overview_v2.py` | Analysis 019 |
| Figure 3 paired effects | Analysis 018 `plots.py:effects` | Retain original points; review caption |
| Figure 4 distributions | Analysis 018 `02_activation_density_v3.py` | Analysis 019; add measured zero masses |
| Figure 5 scale comparison | Analysis 018 `plots.py:scaling` | Analysis 019; unchanged common A7 denominator |
| Figure 6 runtime | Analysis 018 `plots.py:kernels`; Run 029 `18_paper_summary.py` | Analysis 019; promote attribution |
| Appendix operation breakdown | Analysis 018 `plots.py:operations` | Retain full three-size view; compact main-text comparison |
| Appendix complete clipping | `runs/030-2026-09-08-all-models-posthoc-clipping/` figure scripts and PDFs | Retain all 540 coordinates |
| Existing numerical tables | Analysis 018 `03_manuscript_tables.py` | Preserve numerical rows; add separate derived summaries |

## Scientific records and implementation

| Required evidence | Actual location | Access / limit |
|---|---|---|
| 54 training endpoints, 29 contrasts, 15 scale pairs, integer counts, identities | Analysis 018 `figure_data.json`, `evidence.py` and recorded original sources | AVAILABLE; full precision and source paths |
| 540 clipping evaluations with actual p=0 | `runs/030-2026-09-08-all-models-posthoc-clipping/results/clipping-points.json`; copied in supplement | AVAILABLE; canonical training endpoints remain distinct |
| Seven signed histogram checkpoints | `runs/031-2026-09-08-signed-activation-density/results/`; supplement `histograms/*.json.gz`; Analysis 018 `activation-density-v3-data.json` | AVAILABLE; aggregate histograms alone do not show token vectors |
| Absolute runtime and normalized ratios | `runs/029-2026-09-07-pythia14m-matched-kernel-retrospective/results/matched-retrospective-001.json` | AVAILABLE; native/candidate milliseconds, 3-process replicates |
| Raw timing/qualification | Run 029 `artifacts/attempts/scientific-final-*/timing.json`, `quality.json`, `result.json`, `manifest.json` | AVAILABLE; align by candidate/checkpoint/protocol before inference |
| Per-token row-NNZ distributions and site/layer counts | Run 029 `artifacts/attempts/scientific-final-k050-*-r1-a01/diagnostics.json` | AVAILABLE; actual BF16 candidate operands, untimed pass |
| Complete BF16 counterpart to canonical FP16 counts | Run 029 only supplies `bf16_scalar_opportunity_lower_bound` | MISSING as an equivalent counter: excludes probability underflow; do not substitute silently |
| Context perturbation/residual-branch-function tests | No verified source identified | UNVERIFIED; no functional claim permitted |
| Five historical h-only controls | Run 029 c31-c35; Run 012 and Analysis 009 comparison audit | AVAILABLE; keep outside the main 30-checkpoint cohort |
| Gates/sites/pressure/counters | `src/sparsity_research/sites.py`, `pressure.py`, `optimization.py`, `ceilings.py`, `logical_capture.py`, `clipping.py`; frozen run-local implementations | AVAILABLE; audit exact implementation before modifying equations |

## Presentation decisions

The author approved architectural reach and R_arch, with unchanged mathematics.
Framing/headline and figure-budget decisions are pending. New experiments remain
unapproved; optional M/H tasks are deferred rather than represented as findings.
