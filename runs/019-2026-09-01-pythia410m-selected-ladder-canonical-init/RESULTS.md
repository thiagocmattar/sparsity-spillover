# Run 019 terminal results

Run 019 is complete and valid under its approved one-seed, one-pass operational
contract. The canonical endpoint below pairs validation loss and `R_model` from
the same full-validation eager logical-product pass. It is not the separately
retained SDPA execution loss.

| Condition | Paired validation loss | `R_model` | `R_model` (%) | Median train tok/s |
| --- | ---: | ---: | ---: | ---: |
| A0 / GeLU | 4.547456 | 0.000110 | 0.0110 | 72,171 |
| A1-H / ReLU | 4.651294 | 0.210497 | 21.0497 | 47,040 |
| A4-OL1, kappa=0 | 5.692075 | 0.383387 | 38.3387 | 23,756 |
| A4-OL1, kappa=0.01 | 5.691362 | 0.391492 | 39.1492 | 23,733 |
| A4-OL1, kappa=0.05 | 5.678925 | 0.421245 | 42.1245 | 23,755 |
| A4-OL1, kappa=0.1 | 5.678887 | 0.456880 | 45.6880 | 45,092 |
| A4-OL1, kappa=0.5 | 5.190966 | 0.715914 | 71.5914 | 23,769 |
| A7-OL1, kappa=0 | 5.426296 | 0.411215 | 41.1215 | 22,522 |
| A7-OL1, kappa=0.01 | 5.433138 | 0.414781 | 41.4781 | 21,307 |
| A7-OL1, kappa=0.05 | 5.467853 | 0.440532 | 44.0532 | 33,948 |
| A7-OL1, kappa=0.1 | 5.490749 | 0.489976 | 48.9976 | 21,389 |
| A7-OL1, kappa=0.5 | 5.120692 | 0.806155 | 80.6155 | 21,358 |

All four endpoint validation passes per condition covered all 338 complete
2,048-token MiniPile validation blocks (692,224 input tokens), excluding and
reporting the 1,444-token tail. All conditions share initial-parameter SHA-256
`76217bf2ef13de2377c7750515a559cb71f574c274476e08408a5f7749ea9cff`,
training-schedule SHA-256
`d17a6c0c0d4aacff4b477e6d576f511c12c04ebbc37468f08e6fe61ff1c6ad8e`,
and run-code SHA-256
`9fb300223fb57058c42e4030dced0570ce5aadd206db42eebdbeec1d59c4d652`.

## TEAL controls

The consolidated `artifacts/teal/teal_frontiers.json` has status
`complete_verified` and contains all 20 expected points: ten target sparsities
for A0 and ten for A1-H. A0 spans paired loss 4.547456--9.027046 and `R_model`
0.000110--0.700471. A1-H spans paired loss 4.651294--9.029800 and `R_model`
0.210497--0.691629. These are evaluation-only post-hoc clipping frontiers, not
additional training runs or measured runtime speedups.

## Compute and closeout

RunPod billing over the complete lifetime of the twelve scientific Pod IDs
reports $306.5626 of GPU compute and $2.5141 of Pod disk, totaling **$309.0767**.
The balance at teardown was $83.1324. Required balance to completion and
remaining experiment ETC are both zero.

| Condition | Pod | GPU / tier | Billed total (USD) |
| --- | --- | --- | ---: |
| A0 | `fo1h6sbbohjf9w` | RTX PRO 6000 Blackwell / Secure | 18.4931 |
| A1-H | `c2o0tqtsu1c677` | A100 SXM 80 GB / Community | 13.9294 |
| A4 k=0 | `9cayqnss01lq7q` | A100 SXM 80 GB / Community | 26.4129 |
| A4 k=.01 | `3o9ksu1bam0y4m` | A100 SXM 80 GB / Community | 27.0168 |
| A4 k=.05 | `2p3zhy9rh4if1a` | A100 SXM 80 GB / Secure | 30.4658 |
| A4 k=.1 | `sle3lrr9oz2dhf` | H100 80 GB HBM3 / Secure | 33.5442 |
| A4 k=.5 | `tz8ih5yfjhxx6l` | A100 SXM 80 GB / Community | 26.4131 |
| A7 k=0 | `mjpbwge4e257ug` | A100 SXM 80 GB / Community | 27.7137 |
| A7 k=.01 | `dipyx36j0r0zj7` | A100 SXM 80 GB / Community | 28.7386 |
| A7 k=.05 | `ql1f798i6q752o` | RTX PRO 6000 Blackwell / Secure | 29.2314 |
| A7 k=.1 | `yi8idcrkeh9etg` | A100 SXM 80 GB / Secure | 32.2416 |
| A7 k=.5 | `rosnxzyto0ggpp` | A100 SXM 80 GB / Secure | 14.8761 |

The terminal control-plane query at 2026-09-03 10:10 UTC returned no GPU Pods
and no endpoints. The pre-existing 100 GB Standard network volume
`9luykg5yc3` remains intentionally retained at approximately $0.01/hour; it was
not attached to Run 019 and is not GPU spend.

## Verification

- terminal cohort verifier: `verified 12`;
- post-hoc consolidation: `complete_verified`, 20/20 points;
- focused Run 019 tests: 23 passed;
- full bootstrap suite: 183 passed;
- locally retained attempt payload: 58,391,205,180 bytes, including
  58,371,742,618 checkpoint bytes.

The mixed hardware is execution provenance, not a scientific factor, and train
throughput should not be interpreted as an intervention comparison. The
scientific result remains descriptive one-seed evidence; cross-scale analysis
and any manuscript claim require a separate analysis and human approval.
