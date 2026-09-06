# Run 025 frozen-kernel results

All speedups are paired geometric means of native/candidate host-time ratios on the first 16 fixed seed-2500 training-cache blocks. The 95% intervals resample input identities, not fresh processes. Canonical loss and `R_model` come from the source checkpoint's complete 338-block logical pass; Run 025 separately repeated complete candidate/native loss and elementwise-logit checks.

## Pythia-14M

| Family | kappa | partition | canonical loss | R_model | speedup (95% input-cluster CI) | validation delta | status |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| A0 | - | development | 5.208573 | 0.0000% | 0.9791x [0.9605, 1.0023] | -0.000004 | pass |
| A1-H | - | development | 5.269634 | 2.7141% | 0.9920x [0.9877, 0.9960] | +0.000022 | pass |
| A4-OL1 | 0 | development | 5.458276 | 7.8100% | 1.0583x [1.0500, 1.0701] | -0.000001 | pass |
| A4-OL1 | 0.01 | untuned | 5.458309 | 8.5867% | 1.0538x [1.0240, 1.0838] | -0.000001 | pass |
| A4-OL1 | 0.05 | untuned | 5.489760 | 10.4564% | 1.0301x [1.0182, 1.0414] | -0.000000 | pass |
| A4-OL1 | 0.1 | untuned | 5.548254 | 11.3943% | 1.0356x [0.9915, 1.0731] | -0.000001 | pass |
| A4-OL1 | 0.5 | development | 6.037982 | 12.7134% | 1.0486x [1.0375, 1.0568] | +0.000000 | pass |
| A7-OL1 | 0 | development | 5.480181 | 7.0542% | 1.0519x [1.0290, 1.0741] | -0.000036 | pass |
| A7-OL1 | 0.01 | untuned | 5.475801 | 7.7234% | 1.0391x [1.0182, 1.0582] | +0.000032 | pass |
| A7-OL1 | 0.05 | untuned | 5.462800 | 9.8630% | 1.0667x [1.0505, 1.0868] | -0.000043 | pass |
| A7-OL1 | 0.1 | untuned | 5.429488 | 11.7968% | 1.0534x [1.0333, 1.0750] | -0.000015 | FAIL (2 blocks) |
| A7-OL1 | 0.5 | development | 5.829407 | 27.4827% | 1.0436x [1.0391, 1.0481] | -0.000000 | pass |

## Pythia-70M

| Family | kappa | partition | canonical loss | R_model | speedup (95% input-cluster CI) | validation delta | status |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| A0 | - | development | 4.099766 | 0.0005% | 1.0052x [0.9859, 1.0287] | +0.000000 | pass |
| A1-H | - | development | 4.222750 | 10.0658% | 0.9596x [0.9484, 0.9733] | +0.000004 | pass |
| A4-OL1 | 0 | development | 4.805361 | 25.6725% | 0.7661x [0.7637, 0.7686] | -0.000099 | pass |
| A4-OL1 | 0.01 | untuned | 4.822850 | 26.2242% | 0.9699x [0.9485, 0.9877] | +0.000005 | FAIL (1 blocks) |
| A4-OL1 | 0.05 | untuned | 4.874177 | 28.4900% | 0.9423x [0.9340, 0.9503] | -0.000043 | FAIL (2 blocks) |
| A4-OL1 | 0.1 | untuned | 4.943887 | 30.5743% | 0.9067x [0.8994, 0.9140] | -0.000003 | FAIL (1 blocks) |
| A4-OL1 | 0.5 | development | 5.389480 | 35.5962% | 1.0514x [1.0433, 1.0623] | +0.000001 | pass |
| A7-OL1 | 0 | development | 4.941206 | 23.5624% | 0.9993x [0.9878, 1.0169] | +0.000061 | pass |
| A7-OL1 | 0.01 | untuned | 4.936396 | 24.2263% | 0.9911x [0.9727, 1.0120] | -0.000096 | pass |
| A7-OL1 | 0.05 | untuned | 4.963656 | 26.5239% | 0.9904x [0.9800, 1.0073] | +0.000010 | pass |
| A7-OL1 | 0.1 | untuned | 4.950079 | 28.6938% | 0.9896x [0.9849, 0.9940] | +0.000018 | pass |
| A7-OL1 | 0.5 | development | 5.215925 | 40.6019% | 0.9922x [0.9645, 1.0125] | +0.000000 | pass |

## Pythia-410M

| Family | kappa | partition | canonical loss | R_model | speedup (95% input-cluster CI) | validation delta | status |
| --- | ---: | --- | ---: | ---: | ---: | ---: | --- |
| A0 | - | development | 4.547456 | 0.0110% | 0.9450x [0.9424, 0.9496] | +0.000000 | pass |
| A1-H | - | development | 4.651294 | 21.0497% | 0.9436x [0.9422, 0.9459] | +0.000000 | pass |
| A4-OL1 | 0 | development | 5.692075 | 38.3387% | 0.9451x [0.9440, 0.9461] | +0.000000 | pass |
| A4-OL1 | 0.01 | untuned | 5.691362 | 39.1492% | 0.9458x [0.9427, 0.9480] | +0.000000 | pass |
| A4-OL1 | 0.05 | untuned | 5.678925 | 42.1245% | 0.9467x [0.9446, 0.9485] | +0.000000 | pass |
| A4-OL1 | 0.1 | untuned | 5.678887 | 45.6880% | 0.9442x [0.9384, 0.9477] | +0.000000 | pass |
| A4-OL1 | 0.5 | development | 5.190966 | 71.5914% | 1.0176x [1.0143, 1.0197] | +0.000000 | pass |
| A7-OL1 | 0 | development | 5.426296 | 41.1215% | 0.9436x [0.9387, 0.9467] | +0.000000 | pass |
| A7-OL1 | 0.01 | untuned | 5.433138 | 41.4781% | 0.9495x [0.9455, 0.9518] | +0.000000 | pass |
| A7-OL1 | 0.05 | untuned | 5.467853 | 44.0532% | 0.9526x [0.9512, 0.9541] | +0.000000 | pass |
| A7-OL1 | 0.1 | untuned | 5.490749 | 48.9976% | 0.9545x [0.9496, 0.9604] | +0.000000 | pass |
| A7-OL1 | 0.5 | development | 5.120692 | 80.6155% | 1.0118x [1.0046, 1.0170] | +0.000000 | pass |

## Descriptive associations

| Size | stratum | n | excluded failures | slope per +10 pp R_model | Pearson r | R2 | Spearman rho |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 14M | all-final-including-failures | 12 | 0 | +0.0195x | +0.4918 | 0.2419 | +0.2797 |
| 14M | qualified-final | 11 | 1 | +0.0190x | +0.4849 | 0.2351 | +0.2727 |
| 14M | development-qualified | 6 | 0 | +0.0198x | +0.5657 | 0.3201 | +0.4857 |
| 14M | transfer-qualified | 5 | 0 | -0.0252x | -0.2473 | 0.0611 | -0.5000 |
| 14M | A4-qualified | 5 | 0 | -0.0321x | -0.5379 | 0.2894 | -0.6000 |
| 14M | A7-qualified | 4 | 0 | -0.0035x | -0.2779 | 0.0772 | +0.0000 |
| 70M | all-final-including-failures | 12 | 0 | -0.0004x | -0.0065 | 0.0000 | -0.0280 |
| 70M | qualified-final | 9 | 3 | +0.0034x | +0.0515 | 0.0027 | +0.0667 |
| 70M | development-qualified | 6 | 0 | +0.0017x | +0.0252 | 0.0006 | -0.0286 |
| 70M | transfer-qualified | 3 | 0 | -0.0033x | -0.9999 | 0.9999 | -1.0000 |
| 70M | A7-qualified | 5 | 0 | -0.0015x | -0.2581 | 0.0666 | -0.4000 |
| 410M | all-final-including-failures | 12 | 0 | +0.0101x | +0.7814 | 0.6106 | +0.7552 |
| 410M | qualified-final | 12 | 0 | +0.0101x | +0.7814 | 0.6106 | +0.7552 |
| 410M | development-qualified | 6 | 0 | +0.0103x | +0.8609 | 0.7412 | +0.6571 |
| 410M | transfer-qualified | 6 | 0 | +0.0064x | +0.5455 | 0.2976 | +0.3714 |
| 410M | A4-qualified | 5 | 0 | +0.0227x | +0.9754 | 0.9515 | +0.4000 |
| 410M | A7-qualified | 5 | 0 | +0.0166x | +0.9941 | 0.9883 | +1.0000 |

These fits are descriptive across one-seed trained checkpoints. They are not scaling laws; repeated timings are not training replicates. The figure uses only the `qualified-final` fits.

## Same-checkpoint search transitions

| Group | condition | R_model | baseline | optimized | ratio | both correct |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 14M exploratory P0-to-K001 | 14m/a0 | 0.0000% | 0.9757x | 1.0299x | 1.0556x | yes |
| 14M exploratory P0-to-K001 | 14m/a1h | 2.7141% | 0.9681x | 1.0054x | 1.0385x | yes |
| 14M exploratory P0-to-K001 | 14m/a4-0p5 | 12.7134% | 1.0906x | 1.1653x | 1.0685x | yes |
| 14M exploratory P0-to-K001 | 14m/a7-0p5 | 27.4827% | 1.0949x | 1.1514x | 1.0516x | yes |
| 14M final-policy repair K012-to-K013 | 14m/a0 | 0.0000% | 1.0364x | 0.9791x | 0.9446x | no |
| 14M final-policy repair K012-to-K013 | 14m/a1h | 2.7141% | 0.9907x | 0.9920x | 1.0013x | yes |
| 14M final-policy repair K012-to-K013 | 14m/a4-0 | 7.8100% | 1.1096x | 1.0583x | 0.9538x | no |
| 14M final-policy repair K012-to-K013 | 14m/a4-0p5 | 12.7134% | 1.1083x | 1.0486x | 0.9462x | yes |
| 14M final-policy repair K012-to-K013 | 14m/a7-0 | 7.0542% | 1.1057x | 1.0519x | 0.9513x | no |
| 14M final-policy repair K012-to-K013 | 14m/a7-0p5 | 27.4827% | 1.0955x | 1.0436x | 0.9526x | yes |
| 70M complete-validation K009-to-K016 | 70m/a0 | 0.0005% | 0.3214x | 1.0052x | 3.1274x | yes |
| 70M complete-validation K009-to-K016 | 70m/a1h | 10.0658% | 0.9518x | 0.9596x | 1.0082x | yes |
| 70M complete-validation K009-to-K016 | 70m/a4-0 | 25.6725% | 0.8163x | 0.7661x | 0.9386x | yes |
| 70M complete-validation K009-to-K016 | 70m/a4-0p01 | 26.2242% | 0.8241x | 0.9699x | 1.1769x | no |
| 70M complete-validation K009-to-K016 | 70m/a4-0p05 | 28.4900% | 0.9793x | 0.9423x | 0.9623x | no |
| 70M complete-validation K009-to-K016 | 70m/a4-0p1 | 30.5743% | 0.9566x | 0.9067x | 0.9479x | no |
| 70M complete-validation K009-to-K016 | 70m/a4-0p5 | 35.5962% | 1.0017x | 1.0514x | 1.0496x | yes |
| 70M complete-validation K009-to-K016 | 70m/a7-0 | 23.5624% | 0.8543x | 0.9993x | 1.1698x | yes |
| 70M complete-validation K009-to-K016 | 70m/a7-0p01 | 24.2263% | 0.8680x | 0.9911x | 1.1418x | yes |
| 70M complete-validation K009-to-K016 | 70m/a7-0p05 | 26.5239% | 0.8362x | 0.9904x | 1.1843x | no |
| 70M complete-validation K009-to-K016 | 70m/a7-0p1 | 28.6938% | 0.8520x | 0.9896x | 1.1616x | yes |
| 70M complete-validation K009-to-K016 | 70m/a7-0p5 | 40.6019% | 0.9693x | 0.9922x | 1.0236x | yes |
| 410M development K004-to-K010 | 410m/a0 | 0.0110% | 0.4712x | 0.9431x | 2.0016x | yes |
| 410M development K004-to-K010 | 410m/a4-0p5 | 71.5914% | 1.0158x | 1.0205x | 1.0046x | yes |
| 410M development K004-to-K010 | 410m/a7-0p5 | 80.6155% | 1.0034x | 1.0160x | 1.0126x | yes |

The 14M P0-to-K001 and 410M K004-to-K010 comparisons are development evidence. The 14M K012-to-K013 and 70M K009-to-K016 comparisons include complete validation. A dense fallback can improve deployed latency but is not a sparse-kernel win for that condition.
