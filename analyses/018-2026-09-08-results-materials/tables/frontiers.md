# Frontiers

Minimize paired loss and maximize raw model-wide sparsity among evaluated points only. No interpolation or equal-quality inference.

| Pool | Scale | Recipe / checkpoint | Parameter / target p | Loss | S_model (%) | Evidence ID |
| --- | --- | --- | --- | --- | --- | --- |
| 14M_main_trained | 14M | A1-H-L1 | 1.0 | 5.102275 | 3.9493 | 14M:A1-H-L1:1.0 |
| 14M_main_trained | 14M | A4 | 0.1 | 5.419673 | 8.9530 | 14M:A4:0.1 |
| 14M_main_trained | 14M | A7 | 0.1 | 5.428668 | 10.4250 | 14M:A7:0.1 |
| 14M_main_trained | 14M | A7-OL1 | 0.1 | 5.429488 | 11.7968 | 14M:A7-OL1:0.1 |
| 14M_main_trained | 14M | A7 | 0.5 | 5.702895 | 15.3868 | 14M:A7:0.5 |
| 14M_main_trained | 14M | A7-OL1 | 0.5 | 5.829407 | 27.4827 | 14M:A7-OL1:0.5 |
| 14M_all_trained_and_clipped | 14M | A1-H-L1 | 0.0 | 5.102270 | 3.9493 | 14M:A1-H-L1:1.0:clip:0.0 |
| 14M_all_trained_and_clipped | 14M | A1-H-L1 | 1.0 | 5.102275 | 3.9493 | 14M:A1-H-L1:1.0 |
| 14M_all_trained_and_clipped | 14M | A1-H-L1 | 0.1 | 5.106970 | 4.7982 | 14M:A1-H-L1:1.0:clip:0.1 |
| 14M_all_trained_and_clipped | 14M | A1-H-L1 | 0.2 | 5.144074 | 5.6441 | 14M:A1-H-L1:1.0:clip:0.2 |
| 14M_all_trained_and_clipped | 14M | A1-H-L1 | 0.3 | 5.245573 | 6.4902 | 14M:A1-H-L1:1.0:clip:0.3 |
| 14M_all_trained_and_clipped | 14M | A4 | 0.5 | 5.419605 | 8.9537 | 14M:A4:0.1:clip:0.5 |
| 14M_all_trained_and_clipped | 14M | A7 | 0.1 | 5.428668 | 10.4250 | 14M:A7:0.1 |
| 14M_all_trained_and_clipped | 14M | A7 | 0.5 | 5.428695 | 10.4251 | 14M:A7:0.1:clip:0.5 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.0 | 5.429461 | 11.7968 | 14M:A7-OL1:0.1:clip:0.0 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.1 | 5.429461 | 11.7968 | 14M:A7-OL1:0.1:clip:0.1 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.2 | 5.429461 | 11.7968 | 14M:A7-OL1:0.1:clip:0.2 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.3 | 5.429461 | 11.7968 | 14M:A7-OL1:0.1:clip:0.3 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.4 | 5.429461 | 11.7968 | 14M:A7-OL1:0.1:clip:0.4 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.5 | 5.429461 | 11.7968 | 14M:A7-OL1:0.1:clip:0.5 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.6 | 5.453681 | 12.2029 | 14M:A7-OL1:0.1:clip:0.6 |
| 14M_all_trained_and_clipped | 14M | A7 | 0.5 | 5.702895 | 15.3868 | 14M:A7:0.5 |
| 14M_all_trained_and_clipped | 14M | A7 | 0.6 | 5.704142 | 15.4657 | 14M:A7:0.5:clip:0.6 |
| 14M_all_trained_and_clipped | 14M | A7 | 0.7 | 5.761269 | 16.6082 | 14M:A7:0.5:clip:0.7 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.0 | 5.829378 | 27.4824 | 14M:A7-OL1:0.5:clip:0.0 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.1 | 5.829378 | 27.4824 | 14M:A7-OL1:0.5:clip:0.1 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.2 | 5.829378 | 27.4824 | 14M:A7-OL1:0.5:clip:0.2 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.3 | 5.829378 | 27.4824 | 14M:A7-OL1:0.5:clip:0.3 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.4 | 5.829378 | 27.4824 | 14M:A7-OL1:0.5:clip:0.4 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.5 | 5.829378 | 27.4824 | 14M:A7-OL1:0.5:clip:0.5 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.6 | 5.829378 | 27.4824 | 14M:A7-OL1:0.5:clip:0.6 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.5 | 5.829407 | 27.4827 | 14M:A7-OL1:0.5 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.7 | 5.845568 | 27.6391 | 14M:A7-OL1:0.5:clip:0.7 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.8 | 6.022109 | 28.2258 | 14M:A7-OL1:0.5:clip:0.8 |
| 14M_all_trained_and_clipped | 14M | A7-OL1 | 0.9 | 6.584488 | 28.9694 | 14M:A7-OL1:0.5:clip:0.9 |
