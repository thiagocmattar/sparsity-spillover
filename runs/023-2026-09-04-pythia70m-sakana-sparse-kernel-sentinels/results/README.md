# Run 023 sentinel results

`sentinel-summary.csv` is the compact condition-level calibration table.
`linear-primitive-summary.csv` contains all 108 layer/operation measurements,
and `attention-summary.csv` contains the 12 standalone A7 layer compositions.
`association-summary.json` records descriptive correlations and their explicit
non-inferential warning.

Speedup is dense latency divided by sparse latency, so values above one are
faster and values below one are slowdowns. Full-model speedup is paired only
with `R_covered_linear`, because the timed sparse full model retains dense SDPA
attention. `R_covered_linear_plus_attention` belongs only to the standalone A7
composition.

`raw/` contains the six condition JSON files, passed verification, cohort,
preflights, environment inventories, and a SHA-256 manifest. The complete
hash-verified transfer archives remain local under `../artifacts/` and are
intentionally not versioned.

Regenerate the tables with:

```powershell
python ..\07_summarize.py --attempt-dir <extracted-attempt-directory> --output-dir .
```
