# Run051/052 executions added to the loss-latency overview

## Question and method

Where do the newer measured execution paths sit relative to the existing 14M/
70M loss-latency points? The user requested additions to the existing PDF.

All 68 original execution points and their 66 checkpoint records remain intact.
Add 18 absolute 70M measurements: Run051's five T2/Ph and five T7/Pall settings,
Run051 PyTorch/optimized-dense Base, Run052's T2/Ph sparse/dense pairs at
kappa=.05/.1, and Run052 PyTorch/optimized-dense Base. Ten added sparse
intervention points qualify; two Run051 T7/Pall points (.0/.1) fail numerical
qualification. Together with six qualified dense/Base additions, this gives
16 qualified additions and two explicitly marked failures, 86 displayed points.

The C policy is the retained grouped h-feature gather/tensor-core path: sparse
h in layers 1--5, dense h in layer 0, and dense z everywhere. Run052's points
repeat this reference policy; none of its 16 experimental component candidates
was promoted to a full-model benchmark. No component timings are converted into
full-model points. Run051's failed sparse paths are crosses and their curve
segments are broken, so they cannot be read as valid frontier points.

Source scripts: `../02_combined.py` and `../recent_runs.py`. Source data are
Run051 `complete-table.json` / `final-summary.json` and Run052
`references-001-summary.json`. The helper verifies 144 raw source records and 11
original training manifests, as well as all 44 model/config/metadata files for
the 11 added checkpoint identities. Older aggregate checkpoint keys have mixed
model-only/full-recovery scopes; actual file hashes establish the match.

Quality uses the existing ordinary FP16 checkpoint loss, exactly preserving
the figure's x coordinates. The newer tables' logical-pass FP16 losses differ
slightly from that ordinary pass; they and the BF16 numerical-qualification
losses are retained separately in the data and are not substituted on the axis.
All validation uses 338 complete 2048-token blocks from 500 MiniPile documents,
with 1444 tail tokens excluded. Timings use RTX 5090, BF16, batch 1, full logits,
64 inputs, seven passes and three fresh processes. Sessions remain separate:
no pooling, rescaling or replacement of older measurements.

## Figure and caption

[Updated PDF](../figures/02-14m-70m-latency-quality.pdf).

**Validation loss and full-model latency across 14M and 70M, with newer 70M
execution paths.** The original filled circles/squares and connecting lines
retain K050/opt073 results. Blue denotes T2/Ph, orange T7/Pall and gray other
recipes. Hollow diamonds and dotted segments add the Run051 C policy; plus
markers show its Run052 repetitions, and open downward triangles show the
matched Run052 dense controls. Upward triangles and filled crosses distinguish
Run051/052 Base measurements, with hollow faces for optimized dense and filled
faces for PyTorch. Orange x markers fail numerical qualification and are not
usable frontier points. The two legend groups distinguish historical and added
measurements. Both axes remain logarithmic; vertical guides retain Base loss.

## Result and limits

The newer policy moves moderate-kappa T2/Ph much lower on the plot at unchanged
checkpoint quality. Run051 latencies at .05/.1 are 1.183922/1.171148 ms; the
Run052 repetitions are 1.217876/1.203205 ms. Run052's same-checkpoint dense
controls are 1.283211/1.282619 ms. Its native/optimized-dense Base markers are
1.681633/1.286191 ms; Run051's are 1.647401/1.257799 ms.

This also exposes the substantially improved dense baseline and the much flatter
kappa response of the newer policy. The visual shift from opt073 includes dense
engineering gains; it is not evidence of broad h/z sparse execution. The plot
does not resolve the cross-size Delta test and does not supply uncertainty bars
or a formal pooled-session Pareto estimate. In particular, Run052's experimental
candidate search still has no new full-model winner.

The original manuscript copy and its caption remain untouched; this user request
updates the analysis artifact only. Numerical data, source hashes and checks are
in `../data/combined-figure.json` and `../data/combined-verification.json`.
