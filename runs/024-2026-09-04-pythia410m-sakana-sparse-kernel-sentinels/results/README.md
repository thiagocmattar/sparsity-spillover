# Run 024 sentinel results

`sentinel-summary.csv` is the six-condition result table.
`linear-primitive-summary.csv` contains all 432 layer/operation measurements,
and `attention-summary.csv` contains the 48 standalone A7 layer compositions.
`association-summary.json` is descriptive only; the six selected conditions
are topology-confounded and are not an inferential regression sample.

Speedup is native-dense latency divided by sparse-path latency, so values above
one are faster. The full model replaces only the declared linear operations;
attention remains dense SDPA. A7 attention is timed only as a separate,
unfused Q-only/V-only composition.

For A7-OL1 `kappa=0.5`, the H100 eager V-only opportunity exceeds the pinned
Run-019 A100 eager P/V union by 60,013,381 products (about 3.7 ppm of the
component). The exact H100 counts and attention timings remain valid, but they
cannot be combined with that A100 union into a canonical extended
`R_covered`. The corresponding CSV field is therefore blank; counts were not
capped and canonical `R_model` was not changed.

`raw/` contains the six condition JSON files, passed verification, cohort,
preflights, environment inventories, and SHA-256 manifest. The transfer
archive remains local under `../artifacts/` and is intentionally not versioned.

Regenerate with:

```powershell
python ..\07_summarize.py --attempt-dir <extracted-attempt-directory> --output-dir .
```

