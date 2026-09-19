# Threshold placement and pressure placement

19 September review: [the restyled Pythia-14M layer maps](figures/appendix-14m-layer-sparsity.pdf)
use black T/P labels and a shared viridis sparsity
scale. Both thresholds and all 504 original values are preserved. Reproduce
with `plot_layer_sparsity.py`; [O017](observations/O017-layer-sparsity-style.md)
records the caption and audit. A copy is saved directly in
`manuscript/draft/figures/` for review; the manuscript still includes the old figure.

This is the targeted revision requested in the updated
`manuscript/draft/feedback-review-task.md` on 17 September 2026. It uses
retained measurements only. It does not restore the rejected 16 September
rewrite, alter training records, launch compute, or anticipate 70M results.

- **Training:** 64 endpoints: 54 original plus ten verified 14M h-only models.
- **Pressure comparison:** all 30 multisite 14M endpoints, six recipes and five thresholds.
- **Geometry:** all 14,240 optimizer steps from the 20 pressured multisite runs.
- **Activation structure:** complete-validation counters for all six recipes,
  including separate m/h and seven sites by six layers.
- **Runtime:** 35 qualified historical checkpoints. The five four-site h-only
  records are restored from Run 029's raw timing pairs and diagnostics, including
  all-skips-off and attention-dense controls. The original 30 records are unchanged.
  The five seven-site h-only timings remain pending in separately prepared Run 033.

[The observation](observations/O016-pressure-placement.md) records coverage,
captions, results and limitations. [The task audit](TASK-AUDIT.md) maps all 30
requests to the implementation. The former alternative remains separately
archived in `../review-alternative/`.

## Reproduce

From the repository root:

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/pressure-scope/evidence.py
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/pressure-scope/figures.py
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/pressure-scope/tables.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/pressure-scope/test_evidence.py -q -p no:cacheprovider
```

`data/evidence.json` preserves integer counts, both loss passes, per-step geometry,
paired contrasts, historical timing reductions and source hashes. The reported
losses retain Analysis 023's author-approved convention: ordinary final loss for
h-only and eager logical-pass loss otherwise. Maximum pass discrepancy is
0.000105806356 nats. Uniform-pass alternatives are retained, not silently substituted.

`figures/` contains six main-figure replacements and two appendix PDFs (the
layer-map PDF has two pages). `tables/` contains the 64-endpoint table,
matched threshold-scope and pressure-scope contrasts, and five restored runtime rows.
Draft copies have source hashes in its figure and supplementary manifests.

Figure 6 retains the existing measured broad-pressure points. Its all-site
triangles/dash-dot curves use the same pressure-style mapping as the overview;
h-only is an open diamond/dashed curve. Future verified 70M points can be
overlaid with that mapping and the existing panel ranges. No placeholder
points, h-only scaling claim, or pending-measurement extrapolation is present.
