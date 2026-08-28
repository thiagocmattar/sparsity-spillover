# Runs

One folder is one approved experiment or diagnostic.

## Naming

`NNN-YYYY-MM-DD-short-description`, using the next number in
`research/INDEX.md`. Numbers are never reused.

Copy `000-template/` only after design confirmation. Rename it, fill the README
and config, then implement numbered scripts. After implementation, tests and
ETC, ask for separate launch confirmation.

Run-local structure may grow only as the run requires:

```text
README.md
config.yaml
01_prepare.py
02_train.py
03_diagnostics.py
04_plot.py
artifacts/attempts/NNN-timestamp/
figures/NN-description.pdf
observations/INDEX.md
observations/O001-description.md
```

Do not add empty scripts or directories speculatively. Completed run folders are
records; cross-run comparisons belong in `analyses/`.

