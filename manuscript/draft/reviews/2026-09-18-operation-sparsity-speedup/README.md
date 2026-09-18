# Operation sparsity and runtime integration

## Requested changes

The author requested removal of "Broader threshold placement under matched
pressure scope," a much shorter local-sparsity discussion using Analysis024
Figure04, and a connection or merge with the runtime subsection. The subsequent
instruction also removes "Pressure placement across model sizes."

Both subsections and their associated main-text table/figure are removed.
Their source artifacts and the complete appendix measurements remain in the
repository. The preceding quality and paired-pressure prose is preserved.
The results now proceed directly to the merged **Section 4.3: When model-wide
sparsity translates to speedup**, starting on page 6 of the rebuilt 32-page
[main.pdf](../../main.pdf).

The requested [operation figure](../../figures/04-operation-sparsity-changes.pdf)
is copied byte-for-byte from Analysis024 and appears as Figure 5 on page 7.
The existing instruction-bypass chart is retained unchanged as Figure 6 on
page 8. The former local-sparsity label aliases the merged section. Appendix
references now distinguish site counters from operation contributions; references
to the removed scale figure point to the all-size overview or ceiling table.

## Main argument and evidence

The merged text has three short paragraphs: where the additional high-threshold
sparsity comes from; why projection skipping can save time; and why the tested
attention implementation can bypass arithmetic without reducing latency.
It preserves the distinction between zero products, bypassed instructions and
time saved. Individual h/z timing effects and transfer of the 14M timing
attribution to 70M are not claimed; native-relative gain also includes fusion.

The [operation observation](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/016-operation-changes-caption.md)
documents twelve 14M checkpoints at kappa=.05/.5, T4/T7 and P0/Ph/Pall.
Each of the six operation contributions is 100 times its pooled integer
zero-product numerator divided by the full-model denominator, including the
dense head. The stacks sum to model-wide sparsity and cover all 338 complete
validation blocks. They are not runtime estimates.

At T7, kappa=.5, QK/PV contribute 6.384830 pp under Ph and 16.773285 pp under
Pall. Their increase of 10.388455 pp accounts for most of the 10.819044 pp
increase in model-wide sparsity. At kappa=.05 the corresponding contributions
are 1.069185 and 1.753028 pp. These values are checked against the existing
`paired-pressure-figure-data.json` export; no derived measurements were changed.

The [bypass observation](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/029-operation-bypass-summary.md)
and [matched timing evidence](../../../../analyses/024-2026-09-17-h-only-kernel-latency/observations/027-operation-bypass.md)
support the runtime discussion. All 35 historical 14M attention skip-toggle
controls are slower with attention skipping enabled and projection skipping
fixed. This identifies a limitation of the tested implementation. It does not
establish that attention sparsity is intrinsically unhelpful or that every
component of the remaining overhead has been separately timed.

## Verification

Seven existing paper-figure tests and four operation-bypass tests pass.
The adopted operation PDF and retained bypass PDF match their source hashes.
The twelve operation records, their integer count sums and the reported
QK/PV increments are checked. Both removed section titles and labels are
absent from active TeX; all references and citations resolve after BibTeX and
three pdflatex passes. There are no overfull boxes. Four underfull vertical
boxes and one underfull horizontal box remain; visual review checks the
changed pages and all 32 pages in contact sheets.

Build artifacts and raster previews remain ignored under `tmp/pdfs/`.
No training, inference, new timing or cloud work ran. The concurrent paired
figure restyle is separate from this integration.
[verification.json](verification.json) records the build, resolved labels,
numerical audit and source/output hashes.
