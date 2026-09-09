# Revision v2 source map

The editable baseline is commit `33be0d1`, with 28-page `manuscript/draft/main.pdf`.
Its SHA-256 matches the task's supplied base exactly. A baseline rebuild reproduces
all pages in text and pixels; preserved PDF and logs are under ignored
`tmp/revision-v2-baseline/`. Only the author's edited `task.md` was initially dirty.
`baseline.json` records the exact task, PDF and TeX identities.

| Target | Editable source | Numerical / implementation authority |
|---|---|---|
| Root and template | `manuscript/draft/main.tex`, existing ICLR 2027 styles | Official styles unchanged; draft README build command |
| Method rationale and formulas | `methodology.tex`, `methodology-appendix.tex` | `src/sparsity_research/pressure.py`, `optimization.py`, `sites.py`; executed run-local implementations and configs |
| Reach | `methodology.tex`, `experimental-appendix.tex` | `src/sparsity_research/ceilings.py`; architecture dimensions in Analysis 018 `figure_data.json:ceilings` |
| Setup and site diagram | `experimental-study.tex`; `figures/pythia-architecture-map.pdf` | `experimental-notes.md`; Analysis 019 `figure-source/`; primary endpoint identities |
| Quality, pairs, distribution, size | `training-results.tex`; full material in `results-appendix.tex` | Analysis 018 `figure_data.json`; Analysis 019 training audit; Run 030 clipping points; Run 031 signed histograms |
| Cross-size visual | `figures/03-scale-transfer-and-reach.pdf` | Analysis 019 `05_make_figures.py`, Analysis 018 `plots.py:scaling`; expanded view will be owned by Analysis 020 |
| Runtime visual and prose | `kernel-autoresearch.tex`; `figures/07-kernel-attribution.pdf` | Analysis 019 `02_audit_runtime.py`, `runtime-audit.json`, `05_make_figures.py`; Run 029 raw timings/qualification |
| Quality-budget table | `training-results.tex`, `tables/quality-budgets.tex` | Analysis 019 `04_make_tables.py` and training audit; relocate, retain numerical rows |
| Boundary-threshold table | `tables/scale-contrasts-compact.tex` | Analysis 019 table copies; all 15 pairs remain in full appendix table |
| Historical pressure targets | Scientific paragraph currently in `results-appendix.tex`, kernel subsection | Analysis 019 `03_audit_history.py`, `historical-audit.json`; Runs 012/015 |
| Framing | `related-work.tex`, `conclusion.tex`, `introduction.tex`, `abstract.tex` | Verified analyses first; Q-Sparse, Spark and Pythia primary sources named in task |
| References and disclosure | `references.bib`, `ai-use.tex` | Existing literature audit and source papers; retained coding-agent records |

All paths without a prefix in the second column are under `manuscript/draft/`.
The earlier visual can be recovered from retained Analysis 018/019 PDFs and Git;
its conversation-specific exported filename is not required to regenerate it.

Primary training/log paths are the `source` attempt directories in Analysis 018's
54 `trained` records. The initial A0 examples are Run 004 attempt
`001-20260829-221007-bb5288c8`, Run 018 attempt `001-20260901-133016-4e43b254`,
and Run 019 attempt `001-20260902-141527-bcb97fb1`. Their `events.jsonl`,
`config.yaml`, manifests and frozen source inventories identify the actual log
semantics; the v2 log audit will record coverage before interpreting gradients.

Analysis owner: `analyses/020-2026-09-09-intervention-revision/`. New reductions
and PDF figures stay there. Required author-facing data are exact copies under
`revision-v2/data/`, with provenance. No original run record is edited.

Scientific invariants: 54 primary trained conditions (30/12/12), one seed,
29 14M contrasts, 15 size contrasts and 540 clipping evaluations; 338 complete
validation blocks and the reported 1,444-token tail. FP16 endpoint/count and
BF16 timing/quality measurements remain separate. Pressure/gate sites, maps,
normalization, checkpoint identities and raw metric keys are preserved.
