# Analysis 013 - Matched interventions for the manuscript rewrite

Authorized by the user's 2026-09-04 request to rewrite the complete draft and
refactor figures, tables, and evidence using the completed experiments and
adversarial-review-v3. This analysis reprocesses existing numerical artifacts.

The main question is how gate placement and activation pressure change
validation quality and model-wide activation sparsity opportunity under the
matched 14M protocol, and which selected recipes persist at 70M. The 410M
equal-token extension and completed Run 023 runtime sentinels delimit scope.

## Evidence and reduction

- Analysis 008 supplies the 30 main 14M checkpoint identities.
- Analysis 009 supplies five historically realized A4+OL1@h conditions.
- Analyses 011/012 supply 70M/410M identities, all 60 uniform clipping
  evaluations, and architecture/exposure context.
- Run 023 supplies the six verified runtime sentinels, including slowdowns.
- Run 021 supplies the three-rate selection record for the 410M appendix.

Every trained loss is read from the same logical-product pass as its plotted
R_model. Terminal training-evaluation loss is retained separately. Raw counts
are checked before division, with 338 validation blocks and a 1,444-token
excluded tail. Source SHA-256 identities accompany the output. Original runs
and analyses remain immutable.

The paired contrasts hold initialization, data order, and training budget
fixed. Thresholds are treatment levels, not statistical replicates. Differences
between A4 and A7 pressure responses refer to topology-conditioned objectives;
the absent A7+OL1@4 condition prevents fixed-objective interaction attribution.

## Outputs and reproduction

`evidence.py` reduces sources; `plots.py` draws publication PDFs;
`01_build.py` writes tables and invokes those figures. Each figure has an
observation under `observations/`. The draft imports this analysis's assets.

```powershell
.venv/Scripts/python.exe analyses/013-2026-09-04-matched-intervention-manuscript/01_build.py
.venv/Scripts/python.exe -m pytest analyses/013-2026-09-04-matched-intervention-manuscript/test_evidence.py -q
powershell -File manuscript/draft/01_build.ps1
```

No new checkpoint forward pass or training run is part of this reduction.
Read `observations/INDEX.md` for interpretation and `figure_data.json` for the
complete selected evidence, including every 410M point and clipping target.

Verification: ten focused tests pass; all copied asset hashes match; the
17-page compiled draft was rendered and inspected, with main text ending on
page 8. See [VERIFICATION.md](VERIFICATION.md) for the exact scope and PDF hash.
