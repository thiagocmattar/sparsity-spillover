# Archived review figure alternatives

The manuscript revision in commit `f9351d7` was rolled back at the author's
request. These five alternative PDFs are preserved byte-for-byte from that
commit; they are not used in the restored manuscript (`8af18ca`).

- [Quality–sparsity overview](figures/01-quality-sparsity.pdf)
- [Site distributions and operation accounting](figures/04-site-distributions.pdf)
- [Cross-size comparison](figures/05-cross-size.pdf)
- [Quality, absolute latency and projection bypass](figures/07-kernel-quality-latency.pdf)
- [Site-by-layer zero heatmap](figures/appendix-site-zero-heatmap.pdf)

[The observation](observations/O015-review-corrections.md) retains the methods,
coverage, captions and caveats. The original reduction and tests are included;
only source/output paths were adjusted for this separate directory. The builder
reuses Analysis 021 plotting helpers and the same retained source measurements.
It writes only inside this directory and does not update the manuscript.

From the repository root:

```powershell
.venv/Scripts/python.exe analyses/021-2026-09-10-training-results-figures/review-alternative/10_review_figures.py
.venv/Scripts/python.exe -m pytest analyses/021-2026-09-10-training-results-figures/review-alternative/test_review_figures.py -q -p no:cacheprovider
```
