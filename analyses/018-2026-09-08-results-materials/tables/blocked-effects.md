# Blocked Effects

Same eager full-validation pass for loss and R_model; 338 blocks, one seed. R_model is the draft S_model. Treatment minus reference; treatment levels are not replicates.

| Contrast | Level | Dose type | Δ loss | Δ S_model (pp) | Reference ID | Treatment ID |
| --- | --- | --- | --- | --- | --- | --- |
| GELU to ReLU | None | none | +0.061061 | +2.7141 | 14M:A0:None | 14M:A1-H:None |
| Add L1 at h | 0.05 | lambda | -0.063491 | +0.4275 | 14M:A1-H:None | 14M:A1-H-L1:0.05 |
| L1 to OL1 at h | 0.05 | lambda | -0.008080 | +0.0036 | 14M:A1-H-L1:0.05 | 14M:A1-H-OL1:0.05 |
| Add L1 at h | 0.1 | lambda | -0.104152 | +0.6222 | 14M:A1-H:None | 14M:A1-H-L1:0.1 |
| L1 to OL1 at h | 0.1 | lambda | -0.006085 | +0.0023 | 14M:A1-H-L1:0.1 | 14M:A1-H-OL1:0.1 |
| Add L1 at h | 0.5 | lambda | -0.156947 | +1.0734 | 14M:A1-H:None | 14M:A1-H-L1:0.5 |
| L1 to OL1 at h | 0.5 | lambda | -0.002452 | -0.0196 | 14M:A1-H-L1:0.5 | 14M:A1-H-OL1:0.5 |
| Add L1 at h | 1.0 | lambda | -0.167359 | +1.2352 | 14M:A1-H:None | 14M:A1-H-L1:1.0 |
| L1 to OL1 at h | 1.0 | lambda | +0.018910 | -0.0110 | 14M:A1-H-L1:1.0 | 14M:A1-H-OL1:1.0 |
| A1-H to A4 | 0.0 | kappa | +0.200881 | +4.4979 | 14M:A1-H:None | 14M:A4:0.0 |
| Add OL1 to A4 | 0.0 | kappa | -0.012240 | +0.5980 | 14M:A4:0.0 | 14M:A4-OL1:0.0 |
| A4 to A7 gates | 0.0 | kappa | -0.002123 | +0.0056 | 14M:A4:0.0 | 14M:A7:0.0 |
| Add OL1 to A7 | 0.0 | kappa | +0.011788 | -0.1634 | 14M:A7:0.0 | 14M:A7-OL1:0.0 |
| Add OL1 to A4 | 0.01 | kappa | -0.008179 | +1.1730 | 14M:A4:0.01 | 14M:A4-OL1:0.01 |
| A4 to A7 gates | 0.01 | kappa | -0.007669 | +0.2039 | 14M:A4:0.01 | 14M:A7:0.01 |
| Add OL1 to A7 | 0.01 | kappa | +0.016981 | +0.1057 | 14M:A7:0.01 | 14M:A7-OL1:0.01 |
| Add OL1 to A4 | 0.05 | kappa | +0.055645 | +2.2506 | 14M:A4:0.05 | 14M:A4-OL1:0.05 |
| A4 to A7 gates | 0.05 | kappa | +0.003760 | +0.9210 | 14M:A4:0.05 | 14M:A7:0.05 |
| Add OL1 to A7 | 0.05 | kappa | +0.024925 | +0.7361 | 14M:A7:0.05 | 14M:A7-OL1:0.05 |
| Add OL1 to A4 | 0.1 | kappa | +0.128580 | +2.4413 | 14M:A4:0.1 | 14M:A4-OL1:0.1 |
| A4 to A7 gates | 0.1 | kappa | +0.008995 | +1.4720 | 14M:A4:0.1 | 14M:A7:0.1 |
| Add OL1 to A7 | 0.1 | kappa | +0.000819 | +1.3717 | 14M:A7:0.1 | 14M:A7-OL1:0.1 |
| Add OL1 to A4 | 0.5 | kappa | +0.378304 | +2.4979 | 14M:A4:0.5 | 14M:A4-OL1:0.5 |
| A4 to A7 gates | 0.5 | kappa | +0.043217 | +5.1713 | 14M:A4:0.5 | 14M:A7:0.5 |
| Add OL1 to A7 | 0.5 | kappa | +0.126512 | +12.0959 | 14M:A7:0.5 | 14M:A7-OL1:0.5 |
