# 14M quality–latency view with tile and vocabulary-head interventions

User-requested local analysis of retained results. This folder compares the
existing 14M historical cohort, the h/z and T7 corner-tile experiments,
their specialized consumers, and the newer vocabulary-head gate/pressure
experiments. Regenerated for 14M only at the user's request; the primary frontier
follows Analysis041 Figure04 panel (b), with linear axes, connected historical
T2/Ph and T7/Ph sweeps, and a wider latency range to retain all head modes.
It creates no training or timing run and changes no source run.

Quality is canonical final-checkpoint validation over all 500 MiniPile documents,
338 complete 2,048-token blocks and 691,886 predictions; 1,444 tail tokens are
excluded. The horizontal axis is loss and the vertical axis is measured complete
forward latency. Sessions and kernels differ: older optimized paths use BF16;
the head experiments use FP16 with native dense transformer projections.
Their sparse head kernels have not been combined with the earlier sparse h/z
pipeline. The global envelope is descriptive, with compatible execution cohorts
also reported separately. No inferred combined latency enters the frontier.

The latency decomposition keeps Analysis042's nine semantic categories and
colors. Run064 has the retained 14M complete CUPTI stage decomposition. Runs067/068
have measured complete-forward, head gate+cast+projection and projection-only
boundaries. These overlap; they are shown separately and never subtracted to
invent an additive body/head decomposition. A native or no-skip execution keeps
its trained gate; it does not disable thresholding.

For OL1, the existing comparison is Ph versus Ph+head, including the matched
Run066 pairs. There is no head-only pressure run. Geometry is pooled over all
eligible parameter tensors for the joint pressure, not attributed to the head
alone. Ph weights each of six h tensors by 1/6; Ph+head weights six h tensors
and the final head-input tensor by 1/7. The eligible parameter pool also changes.
The manuscript's removed component is normalized by the task direction, not by
the pressure norm. Both normalizations, raw-gradient and adaptive-direction
conflict frequency, pre-projection pressure norm and post-projection pre-cap
ratio are reported with exact definitions.

All 712 optimizer boundaries per condition are retained, including the initial
zero-learning-rate boundary. Interquartile ranges describe training-step
variation, not uncertainty across seeds. This analysis adds no manuscript text
and promotes no finding.

Status: completed local analysis of retained evidence. The declared cohort has
87 retained 14M execution points and 12 observed 14M frontier points. No head execution
extends the observed global envelope; the specialized T2 tile model at κ=0.1
does (loss 5.596746, full forward 0.486441 ms). Execution-cohort panels retain the
precision/body-kernel distinction. Matched head8 κ=0.1 Ph→Ph+head raises median
removed/task norm from 2.227% to 8.503%, while cap activity remains rare. These
are descriptive results, not promoted findings.

Start with the [report and tables](results/REPORT.md),
[all 87 14M coordinates](results/ALL-POINTS.md), and
[measured decomposition](results/DECOMPOSITION.md).

| Figure | Question |
| --- | --- |
| [01 14M quality–latency](figures/01-global-quality-latency.pdf) | Do head interventions extend the existing 14M envelope? |
| [02 Execution-cohort frontiers](figures/02-execution-cohort-frontiers.pdf) | What remains within each execution package? |
| [03 Existing full decomposition](figures/03-existing-full-decomposition.pdf) | Where is CUDA work spent under common colors and labels? |
| [04 Head timing boundaries](figures/04-head-measured-boundaries.pdf) | What changes in projection, head path and full forward? |
| [05 Global orthogonalization comparison](figures/05-orthogonalization-head-comparison.pdf) | How does Ph compare with Ph+head? |
| [06 Manuscript-style OL1 view](figures/06-manuscript-style-geometry.pdf) | How do the new conditions relate to the original geometry view? |

Every PDF has a corresponding [observation](observations/INDEX.md). Source file
identities are retained in [latency provenance](provenance/latency-sources.json)
and [geometry provenance](provenance/geometry-sources.json). Reduced data keep
the complete point coordinates, process ranges, integer counts, geometry
quartiles and traces. [Verification](results/verification.json) records the source
hash audit, mathematical checks and table/figure checks; [generated file hashes](provenance/generated.json)
identify the completed artifacts. Publication outputs are PDF only; temporary
rendered previews are ignored. Source runs and manuscript are unchanged.

Reproduce from the repository root, using the existing Python environment:

```powershell
.venv/Scripts/python.exe -X utf8 analyses/043-2026-10-09-global-tile-head-frontier/01_collect_latency.py
.venv/Scripts/python.exe -X utf8 analyses/043-2026-10-09-global-tile-head-frontier/02_collect_geometry.py
.venv/Scripts/python.exe -X utf8 analyses/043-2026-10-09-global-tile-head-frontier/03_plot_frontiers.py
.venv/Scripts/python.exe -X utf8 analyses/043-2026-10-09-global-tile-head-frontier/04_plot_latency.py
.venv/Scripts/python.exe -X utf8 analyses/043-2026-10-09-global-tile-head-frontier/05_plot_geometry.py
.venv/Scripts/python.exe -X utf8 analyses/043-2026-10-09-global-tile-head-frontier/06_report.py
.venv/Scripts/python.exe -X utf8 analyses/043-2026-10-09-global-tile-head-frontier/07_verify.py
```

The collectors require the locally retained source artifacts. No new model or
benchmark execution is involved. The geometry audit reuses the historical
Analysis021 metric auditor, as a read-only mathematical reference.
The regeneration check compares every retained 14M coordinate and full profile
with the original inventory in commit `19f4cd82`; verification therefore requires
that commit in the local Git history. It also checks that no PDF labels another
model size. The 14M filter changes presentation scope only.
