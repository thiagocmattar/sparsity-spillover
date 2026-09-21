# Later 70M endpoint in the quality-sparsity-latency figure

## Question and coverage

Complete the five-threshold T2/Ph curve with the separately measured kappa=.5
checkpoint. Run047 uses the frozen Run045 protocol and opt073; all 9 final
processes across the endpoint and two repeated references qualify. All 217
returned files are hash-verified and local reduction is identical. No new
training or clipping is included.

## Figure method and caption

`figures/23-70m-quality-sparsity-native-latency.pdf` plots canonical validation
loss and full-model latency versus model-wide logical sparsity for 27 trained 70M
checkpoints. The original 26 Run045 rows and historical clipping curves remain
unchanged. Base marker and horizontal latency reference remain the original
native 1.611ms. Other latency markers use opt073. The added T2/Ph kappa=.5
endpoint uses the same circular marker as its recipe in both panels. At the
user's request on 21 September, the extra diamond overlays and later-session
legend entry were removed; session provenance remains in the appendix and data.
Recipe colors, line styles, ceilings and panel ranges retain
Analysis028's conventions. Lines connect measured thresholds, not fitted trends.

## Result and interpretation

The added point is loss 4.838033648, sparsity 15.42696957%, optimized 1.214368251ms.
The same session's original port takes 1.695805837ms and native Base 1.656128879ms:
1.363778x native-base speedup, versus0.976603x for the port. The endpoint's own
native implementation takes 1.720100231ms; that is a different denominator.
Low-threshold T2/Ph points still cost 1.956--2.504ms at losses 4.129--4.193.
The high threshold improves latency with a clear quality cost.

Repeated optimized Base/kappa=.1 change +2.190%/+0.198% versus Run045;
native Base changes +2.798%. The appendix table presents both sessions.
No rescaling, averaging or cross-session speedup denominator is introduced.
The complete table contains all 84 unique trained conditions (45/27/12),
including all 27 at 70M. Earlier Run046 port timing remains in the exported
historical record, separately from Run047's paired port measurement.

## Caveats

One seed and workload; measurements across two physical GPU sessions are not
one matched sweep. Canonical sparsity and quality are unchanged by kernel
measurement. BF16 qualification losses differ from canonical validation losses.
The kernel includes dense improvements; this endpoint does not isolate pure
skipping or establish a scale-dependent speedup law. Process ranges are
repeated-measurement summaries, not confidence intervals.

## Sources and generation

Run047 `results/matched-grid.json`, `results/local-verification.json`,
`provenance/inputs.json`; Analysis028 `data/full-trained-results.json`.
`01_integrate.py` verifies and joins the new endpoint; `02_plot.py` generates
the PDF; `03_install_verify.py` checks all original rows, integer arithmetic,
27-point membership, table numbers, source hashes and installed copies.
`data/verification.json` records the checks; source hashes are carried in
`data/full-trained-results.json`. Manuscript TeX links this observation.
