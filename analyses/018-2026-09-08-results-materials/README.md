# Analysis 018 - Results materials for the revised argument

Prepared for the user's 8 September 2026 request after reading the ten-page
`manuscript/draft/main.pdf`, dated 8 September. This folder reprocesses existing
local evidence. It does not alter prior runs or the manuscript, execute a model,
or promote a research finding. The repository convention is `analyses/`, rather
than a separate `/analysis` tree.

Start with [RESULTS.md](RESULTS.md): the recommended argument, numerical claims,
and decisions about normalization and the activation case study. The maintained
[checklist](CHECKLIST.md) records every work item. [EVIDENCE-AUDIT.md](EVIDENCE-AUDIT.md)
documents coverage, exclusions, and discrepancies.

## Recommended results order

| Role | Material | Question answered |
| --- | --- | --- |
| Main, opening | [01: 14M overview](figures/01-14m-overview.pdf) | Which recipes occupy the quality-sparsity frontier? |
| Main, explanation | [02: blocked effects](figures/02-blocked-intervention-effects.pdf) | What does each matched intervention add, and how does the effect change with dose? |
| Main, transfer | [03: scale and ceilings](figures/03-scale-transfer-and-ceilings.pdf) | Which recipe orderings persist, and what does normalization change? |
| Supporting explanation / appendix | [04: operation accounting](figures/04-operation-accounting.pdf) | How much is learned zero rate versus architecture-dependent weighting? |
| Main case study or appendix if space is tight | [05: activation mass grid](figures/05-activation-mass-grid.pdf) | Where does mass become exactly zero, and where does it merely become small? |
| Appendix, full comparison | [06: all post-hoc points](figures/06-complete-posthoc-comparison.pdf) | Does the overview survive the available trained-plus-clipped controls? |
| Existing systems subsection | [07: kernel realization](figures/07-kernel-realization.pdf) | Can a specialized implementation exploit the opportunity? |

Figures 01-06 use the draft's calligraphic S notation. Figure 07 is a byte-identical
copy of Run 029's final revision, already used in the draft; its R_model is the
same operational field. Do not insert a duplicate of the existing kernel figure.
Every figure's caption, interpretation, and source is in [observations/INDEX.md](observations/INDEX.md).

## Tables and insertion material

- Main table candidates: [pressure and gate effects](tables/pressure-and-gate-effects.md)
  and [scale endpoints](tables/scale-endpoints.md), each with a `.tex` counterpart.
- Complete tables: [all 59 trained conditions](tables/all-trained-endpoints.md),
  [all 190 clipping points](tables/all-clipping-points.md), [30 matched contrasts](tables/blocked-effects.md),
  [all 15 cross-scale recipe pairs](tables/scale-paired-recipes.md), and [frontier membership](tables/frontiers.md).
- Audit/appendix tables: [training protocol](tables/training-protocol.md),
  [analytic counts](tables/architecture-counts.md), [operation counts](tables/operation-counts.md),
  [activation statistics](tables/activation-statistics.md), [normalization audit](tables/normalization-audit.md),
  and [runtime summary](tables/runtime-summary.md).
- [results.tex](results.tex) is an optional, self-contained results fragment for
  insertion before the existing kernel subsection. It has source-observation
  comments and uses analysis-owned assets. The draft itself is untouched.

The main ladder has 30 trained 14M conditions. The overview also explicitly
identifies five historical A4-OL1[h] conditions, for 35 total. Each larger size
has 12 trained conditions. Clipping covers 150 14M evaluations and 20 at each
larger size. These are evaluated conditions, not 249 independent random seeds.

## Reproduction and provenance

```powershell
.venv/Scripts/python.exe analyses/018-2026-09-08-results-materials/01_build.py
.venv/Scripts/python.exe -m pytest -p no:cacheprovider analyses/018-2026-09-08-results-materials/test_evidence.py tests/test_ceilings.py tests/test_metrics.py -q
```

`evidence.py` reads raw retained counts, manifests, configs, and clipping
evaluations; `plots.py` owns only this folder's figures; `01_build.py` writes
the reproducible data, tables and PDFs. `figure_data.json` includes source
SHA-256 identities, integer counts, normalization numerators, contrast IDs,
and both endpoint and terminal losses. `artifact_inventory.json` identifies
the generated figures/tables. Read [VERIFICATION.md](VERIFICATION.md) for the
actual verification scope. No dataset or checkpoint bytes are copied here.
