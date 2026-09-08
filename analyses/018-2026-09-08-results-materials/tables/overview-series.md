# Overview Series

Figure 01: all 30 trained endpoints and all 150 retained clipping evaluations, each series in parameter order. Thin clipping paths preserve actual p=0 measurements; trained paths retain dominated points. Plot loss window: 5.04–6.15. These are evaluated sweeps, not Pareto envelopes. Source O001.

| Series | Threshold / pressure / target p | Loss | S_model (%) | Evidence ID |
| --- | --- | --- | --- | --- |
| A0 | None | 5.208573 | 0.0000 | 14M:A0:None |
| A1-H | None | 5.269634 | 2.7141 | 14M:A1-H:None |
| A1-H-L1 | 0.05 | 5.206144 | 3.1416 | 14M:A1-H-L1:0.05 |
| A1-H-L1 | 0.1 | 5.165483 | 3.3363 | 14M:A1-H-L1:0.1 |
| A1-H-L1 | 0.5 | 5.112688 | 3.7876 | 14M:A1-H-L1:0.5 |
| A1-H-L1 | 1.0 | 5.102275 | 3.9493 | 14M:A1-H-L1:1.0 |
| A1-H-OL1 | 0.05 | 5.198063 | 3.1452 | 14M:A1-H-OL1:0.05 |
| A1-H-OL1 | 0.1 | 5.159397 | 3.3386 | 14M:A1-H-OL1:0.1 |
| A1-H-OL1 | 0.5 | 5.110236 | 3.7680 | 14M:A1-H-OL1:0.5 |
| A1-H-OL1 | 1.0 | 5.121185 | 3.9384 | 14M:A1-H-OL1:1.0 |
| A4 | 0.0 | 5.470516 | 7.2120 | 14M:A4:0.0 |
| A4 | 0.01 | 5.466488 | 7.4137 | 14M:A4:0.01 |
| A4 | 0.05 | 5.434115 | 8.2059 | 14M:A4:0.05 |
| A4 | 0.1 | 5.419673 | 8.9530 | 14M:A4:0.1 |
| A4 | 0.5 | 5.659678 | 10.2155 | 14M:A4:0.5 |
| A4-OL1 | 0.0 | 5.458276 | 7.8100 | 14M:A4-OL1:0.0 |
| A4-OL1 | 0.01 | 5.458309 | 8.5867 | 14M:A4-OL1:0.01 |
| A4-OL1 | 0.05 | 5.489760 | 10.4564 | 14M:A4-OL1:0.05 |
| A4-OL1 | 0.1 | 5.548254 | 11.3943 | 14M:A4-OL1:0.1 |
| A4-OL1 | 0.5 | 6.037982 | 12.7134 | 14M:A4-OL1:0.5 |
| A7 | 0.0 | 5.468393 | 7.2177 | 14M:A7:0.0 |
| A7 | 0.01 | 5.458820 | 7.6176 | 14M:A7:0.01 |
| A7 | 0.05 | 5.437875 | 9.1269 | 14M:A7:0.05 |
| A7 | 0.1 | 5.428668 | 10.4250 | 14M:A7:0.1 |
| A7 | 0.5 | 5.702895 | 15.3868 | 14M:A7:0.5 |
| A7-OL1 | 0.0 | 5.480181 | 7.0542 | 14M:A7-OL1:0.0 |
| A7-OL1 | 0.01 | 5.475801 | 7.7234 | 14M:A7-OL1:0.01 |
| A7-OL1 | 0.05 | 5.462800 | 9.8630 | 14M:A7-OL1:0.05 |
| A7-OL1 | 0.1 | 5.429488 | 11.7968 | 14M:A7-OL1:0.1 |
| A7-OL1 | 0.5 | 5.829407 | 27.4827 | 14M:A7-OL1:0.5 |
| A0 + clipping | 0.0 | 5.208594 | 0.0000 | 14M:clip:gelu-control:0.0 |
| A0 + clipping | 0.1 | 5.214453 | 1.2787 | 14M:clip:gelu-control:0.1 |
| A0 + clipping | 0.2 | 5.250270 | 2.5548 | 14M:clip:gelu-control:0.2 |
| A0 + clipping | 0.3 | 5.361901 | 3.8316 | 14M:clip:gelu-control:0.3 |
| A0 + clipping | 0.4 | 5.605384 | 5.1255 | 14M:clip:gelu-control:0.4 |
| A0 + clipping | 0.5 | 6.077731 | 6.4419 | 14M:clip:gelu-control:0.5 |
| A0 + clipping | 0.6 | 6.821173 | 7.7644 | 14M:clip:gelu-control:0.6 |
| A0 + clipping | 0.7 | 7.766948 | 9.1235 | 14M:clip:gelu-control:0.7 |
| A0 + clipping | 0.8 | 8.336118 | 10.5860 | 14M:clip:gelu-control:0.8 |
| A0 + clipping | 0.9 | 8.825610 | 11.9419 | 14M:clip:gelu-control:0.9 |
| A1-H + clipping | 0.0 | 5.269633 | 2.7141 | 14M:clip:relu-control:0.0 |
| A1-H + clipping | 0.1 | 5.274073 | 3.5684 | 14M:clip:relu-control:0.1 |
| A1-H + clipping | 0.2 | 5.307940 | 4.4184 | 14M:clip:relu-control:0.2 |
| A1-H + clipping | 0.3 | 5.405056 | 5.2719 | 14M:clip:relu-control:0.3 |
| A1-H + clipping | 0.4 | 5.602965 | 6.1293 | 14M:clip:relu-control:0.4 |
| A1-H + clipping | 0.5 | 5.976274 | 6.9945 | 14M:clip:relu-control:0.5 |
| A1-H + clipping | 0.6 | 6.521474 | 7.8929 | 14M:clip:relu-control:0.6 |
| A1-H + clipping | 0.7 | 7.306175 | 9.0518 | 14M:clip:relu-control:0.7 |
| A1-H + clipping | 0.8 | 8.107618 | 10.5585 | 14M:clip:relu-control:0.8 |
| A1-H + clipping | 0.9 | 8.869450 | 11.8789 | 14M:clip:relu-control:0.9 |
| a4z-one-sided-kappa-0 + clipping | 0.0 | 5.470511 | 7.2120 | 14M:clip:a4z-one-sided-kappa-0:0.0 |
| a4z-one-sided-kappa-0 + clipping | 0.1 | 5.470511 | 7.2120 | 14M:clip:a4z-one-sided-kappa-0:0.1 |
| a4z-one-sided-kappa-0 + clipping | 0.2 | 5.470511 | 7.2120 | 14M:clip:a4z-one-sided-kappa-0:0.2 |
| a4z-one-sided-kappa-0 + clipping | 0.3 | 5.470511 | 7.2120 | 14M:clip:a4z-one-sided-kappa-0:0.3 |
| a4z-one-sided-kappa-0 + clipping | 0.4 | 5.470428 | 7.2189 | 14M:clip:a4z-one-sided-kappa-0:0.4 |
| a4z-one-sided-kappa-0 + clipping | 0.5 | 5.470153 | 7.3099 | 14M:clip:a4z-one-sided-kappa-0:0.5 |
| a4z-one-sided-kappa-0 + clipping | 0.6 | 5.539702 | 8.0783 | 14M:clip:a4z-one-sided-kappa-0:0.6 |
| a4z-one-sided-kappa-0 + clipping | 0.7 | 5.957410 | 8.9564 | 14M:clip:a4z-one-sided-kappa-0:0.7 |
| a4z-one-sided-kappa-0 + clipping | 0.8 | 6.974517 | 10.2149 | 14M:clip:a4z-one-sided-kappa-0:0.8 |
| a4z-one-sided-kappa-0 + clipping | 0.9 | 8.247837 | 11.6306 | 14M:clip:a4z-one-sided-kappa-0:0.9 |
| a4z-one-sided-kappa-0p01 + clipping | 0.0 | 5.466524 | 7.4137 | 14M:clip:a4z-one-sided-kappa-0p01:0.0 |
| a4z-one-sided-kappa-0p01 + clipping | 0.1 | 5.466524 | 7.4137 | 14M:clip:a4z-one-sided-kappa-0p01:0.1 |
| a4z-one-sided-kappa-0p01 + clipping | 0.2 | 5.466524 | 7.4137 | 14M:clip:a4z-one-sided-kappa-0p01:0.2 |
| a4z-one-sided-kappa-0p01 + clipping | 0.3 | 5.466524 | 7.4137 | 14M:clip:a4z-one-sided-kappa-0p01:0.3 |
| a4z-one-sided-kappa-0p01 + clipping | 0.4 | 5.466524 | 7.4137 | 14M:clip:a4z-one-sided-kappa-0p01:0.4 |
| a4z-one-sided-kappa-0p01 + clipping | 0.5 | 5.466257 | 7.4565 | 14M:clip:a4z-one-sided-kappa-0p01:0.5 |
| a4z-one-sided-kappa-0p01 + clipping | 0.6 | 5.534816 | 8.1783 | 14M:clip:a4z-one-sided-kappa-0p01:0.6 |
| a4z-one-sided-kappa-0p01 + clipping | 0.7 | 5.958228 | 9.0027 | 14M:clip:a4z-one-sided-kappa-0p01:0.7 |
| a4z-one-sided-kappa-0p01 + clipping | 0.8 | 6.988377 | 10.1968 | 14M:clip:a4z-one-sided-kappa-0p01:0.8 |
| a4z-one-sided-kappa-0p01 + clipping | 0.9 | 8.158731 | 11.6214 | 14M:clip:a4z-one-sided-kappa-0p01:0.9 |
| a4z-one-sided-kappa-0p05 + clipping | 0.0 | 5.434159 | 8.2059 | 14M:clip:a4z-one-sided-kappa-0p05:0.0 |
| a4z-one-sided-kappa-0p05 + clipping | 0.1 | 5.434159 | 8.2059 | 14M:clip:a4z-one-sided-kappa-0p05:0.1 |
| a4z-one-sided-kappa-0p05 + clipping | 0.2 | 5.434159 | 8.2059 | 14M:clip:a4z-one-sided-kappa-0p05:0.2 |
| a4z-one-sided-kappa-0p05 + clipping | 0.3 | 5.434159 | 8.2059 | 14M:clip:a4z-one-sided-kappa-0p05:0.3 |
| a4z-one-sided-kappa-0p05 + clipping | 0.4 | 5.434159 | 8.2059 | 14M:clip:a4z-one-sided-kappa-0p05:0.4 |
| a4z-one-sided-kappa-0p05 + clipping | 0.5 | 5.434098 | 8.2130 | 14M:clip:a4z-one-sided-kappa-0p05:0.5 |
| a4z-one-sided-kappa-0p05 + clipping | 0.6 | 5.479514 | 8.7583 | 14M:clip:a4z-one-sided-kappa-0p05:0.6 |
| a4z-one-sided-kappa-0p05 + clipping | 0.7 | 5.864863 | 9.4169 | 14M:clip:a4z-one-sided-kappa-0p05:0.7 |
| a4z-one-sided-kappa-0p05 + clipping | 0.8 | 6.693103 | 10.1913 | 14M:clip:a4z-one-sided-kappa-0p05:0.8 |
| a4z-one-sided-kappa-0p05 + clipping | 0.9 | 8.091245 | 11.4622 | 14M:clip:a4z-one-sided-kappa-0p05:0.9 |
| a4z-one-sided-kappa-0p1 + clipping | 0.0 | 5.419630 | 8.9530 | 14M:clip:a4z-one-sided-kappa-0p1:0.0 |
| a4z-one-sided-kappa-0p1 + clipping | 0.1 | 5.419630 | 8.9530 | 14M:clip:a4z-one-sided-kappa-0p1:0.1 |
| a4z-one-sided-kappa-0p1 + clipping | 0.2 | 5.419630 | 8.9530 | 14M:clip:a4z-one-sided-kappa-0p1:0.2 |
| a4z-one-sided-kappa-0p1 + clipping | 0.3 | 5.419630 | 8.9530 | 14M:clip:a4z-one-sided-kappa-0p1:0.3 |
| a4z-one-sided-kappa-0p1 + clipping | 0.4 | 5.419630 | 8.9530 | 14M:clip:a4z-one-sided-kappa-0p1:0.4 |
| a4z-one-sided-kappa-0p1 + clipping | 0.5 | 5.419605 | 8.9537 | 14M:clip:a4z-one-sided-kappa-0p1:0.5 |
| a4z-one-sided-kappa-0p1 + clipping | 0.6 | 5.446599 | 9.3424 | 14M:clip:a4z-one-sided-kappa-0p1:0.6 |
| a4z-one-sided-kappa-0p1 + clipping | 0.7 | 5.775362 | 10.0120 | 14M:clip:a4z-one-sided-kappa-0p1:0.7 |
| a4z-one-sided-kappa-0p1 + clipping | 0.8 | 6.579617 | 10.6554 | 14M:clip:a4z-one-sided-kappa-0p1:0.8 |
| a4z-one-sided-kappa-0p1 + clipping | 0.9 | 7.759894 | 11.3782 | 14M:clip:a4z-one-sided-kappa-0p1:0.9 |
| a4z-one-sided-kappa-0p5 + clipping | 0.0 | 5.659684 | 10.2154 | 14M:clip:a4z-one-sided-kappa-0p5:0.0 |
| a4z-one-sided-kappa-0p5 + clipping | 0.1 | 5.659684 | 10.2154 | 14M:clip:a4z-one-sided-kappa-0p5:0.1 |
| a4z-one-sided-kappa-0p5 + clipping | 0.2 | 5.659684 | 10.2154 | 14M:clip:a4z-one-sided-kappa-0p5:0.2 |
| a4z-one-sided-kappa-0p5 + clipping | 0.3 | 5.659684 | 10.2154 | 14M:clip:a4z-one-sided-kappa-0p5:0.3 |
| a4z-one-sided-kappa-0p5 + clipping | 0.4 | 5.659684 | 10.2154 | 14M:clip:a4z-one-sided-kappa-0p5:0.4 |
| a4z-one-sided-kappa-0p5 + clipping | 0.5 | 5.659684 | 10.2154 | 14M:clip:a4z-one-sided-kappa-0p5:0.5 |
| a4z-one-sided-kappa-0p5 + clipping | 0.6 | 5.660417 | 10.2271 | 14M:clip:a4z-one-sided-kappa-0p5:0.6 |
| a4z-one-sided-kappa-0p5 + clipping | 0.7 | 5.705777 | 10.5631 | 14M:clip:a4z-one-sided-kappa-0p5:0.7 |
| a4z-one-sided-kappa-0p5 + clipping | 0.8 | 6.099449 | 11.3257 | 14M:clip:a4z-one-sided-kappa-0p5:0.8 |
| a4z-one-sided-kappa-0p5 + clipping | 0.9 | 6.824761 | 12.0673 | 14M:clip:a4z-one-sided-kappa-0p5:0.9 |
| relu-l1n-0p05 + clipping | 0.0 | 5.206157 | 3.1416 | 14M:clip:relu-l1n-0p05:0.0 |
| relu-l1n-0p05 + clipping | 0.1 | 5.211364 | 3.9914 | 14M:clip:relu-l1n-0p05:0.1 |
| relu-l1n-0p05 + clipping | 0.2 | 5.248624 | 4.8361 | 14M:clip:relu-l1n-0p05:0.2 |
| relu-l1n-0p05 + clipping | 0.3 | 5.350951 | 5.6843 | 14M:clip:relu-l1n-0p05:0.3 |
| relu-l1n-0p05 + clipping | 0.4 | 5.568858 | 6.5389 | 14M:clip:relu-l1n-0p05:0.4 |
| relu-l1n-0p05 + clipping | 0.5 | 5.973093 | 7.3957 | 14M:clip:relu-l1n-0p05:0.5 |
| relu-l1n-0p05 + clipping | 0.6 | 6.600953 | 8.2748 | 14M:clip:relu-l1n-0p05:0.6 |
| relu-l1n-0p05 + clipping | 0.7 | 7.435047 | 9.2050 | 14M:clip:relu-l1n-0p05:0.7 |
| relu-l1n-0p05 + clipping | 0.8 | 8.427544 | 10.2861 | 14M:clip:relu-l1n-0p05:0.8 |
| relu-l1n-0p05 + clipping | 0.9 | 9.082812 | 11.7112 | 14M:clip:relu-l1n-0p05:0.9 |
| relu-l1n-0p1 + clipping | 0.0 | 5.165466 | 3.3363 | 14M:clip:relu-l1n-0p1:0.0 |
| relu-l1n-0p1 + clipping | 0.1 | 5.170191 | 4.1862 | 14M:clip:relu-l1n-0p1:0.1 |
| relu-l1n-0p1 + clipping | 0.2 | 5.207756 | 5.0334 | 14M:clip:relu-l1n-0p1:0.2 |
| relu-l1n-0p1 + clipping | 0.3 | 5.315055 | 5.8804 | 14M:clip:relu-l1n-0p1:0.3 |
| relu-l1n-0p1 + clipping | 0.4 | 5.550581 | 6.7275 | 14M:clip:relu-l1n-0p1:0.4 |
| relu-l1n-0p1 + clipping | 0.5 | 5.979635 | 7.5807 | 14M:clip:relu-l1n-0p1:0.5 |
| relu-l1n-0p1 + clipping | 0.6 | 6.633155 | 8.4508 | 14M:clip:relu-l1n-0p1:0.6 |
| relu-l1n-0p1 + clipping | 0.7 | 7.553385 | 9.3610 | 14M:clip:relu-l1n-0p1:0.7 |
| relu-l1n-0p1 + clipping | 0.8 | 8.650510 | 10.2834 | 14M:clip:relu-l1n-0p1:0.8 |
| relu-l1n-0p1 + clipping | 0.9 | 9.159795 | 11.5860 | 14M:clip:relu-l1n-0p1:0.9 |
| relu-l1n-0p5 + clipping | 0.0 | 5.112683 | 3.7876 | 14M:clip:relu-l1n-0p5:0.0 |
| relu-l1n-0p5 + clipping | 0.1 | 5.117628 | 4.6370 | 14M:clip:relu-l1n-0p5:0.1 |
| relu-l1n-0p5 + clipping | 0.2 | 5.155895 | 5.4782 | 14M:clip:relu-l1n-0p5:0.2 |
| relu-l1n-0p5 + clipping | 0.3 | 5.271484 | 6.3172 | 14M:clip:relu-l1n-0p5:0.3 |
| relu-l1n-0p5 + clipping | 0.4 | 5.510881 | 7.1620 | 14M:clip:relu-l1n-0p5:0.4 |
| relu-l1n-0p5 + clipping | 0.5 | 5.986626 | 7.9989 | 14M:clip:relu-l1n-0p5:0.5 |
| relu-l1n-0p5 + clipping | 0.6 | 6.687065 | 8.8436 | 14M:clip:relu-l1n-0p5:0.6 |
| relu-l1n-0p5 + clipping | 0.7 | 7.495898 | 9.6755 | 14M:clip:relu-l1n-0p5:0.7 |
| relu-l1n-0p5 + clipping | 0.8 | 8.327347 | 10.5441 | 14M:clip:relu-l1n-0p5:0.8 |
| relu-l1n-0p5 + clipping | 0.9 | 9.188822 | 11.1798 | 14M:clip:relu-l1n-0p5:0.9 |
| relu-l1n-1 + clipping | 0.0 | 5.102270 | 3.9493 | 14M:clip:relu-l1n-1:0.0 |
| relu-l1n-1 + clipping | 0.1 | 5.106970 | 4.7982 | 14M:clip:relu-l1n-1:0.1 |
| relu-l1n-1 + clipping | 0.2 | 5.144074 | 5.6441 | 14M:clip:relu-l1n-1:0.2 |
| relu-l1n-1 + clipping | 0.3 | 5.245573 | 6.4902 | 14M:clip:relu-l1n-1:0.3 |
| relu-l1n-1 + clipping | 0.4 | 5.470511 | 7.3462 | 14M:clip:relu-l1n-1:0.4 |
| relu-l1n-1 + clipping | 0.5 | 5.898911 | 8.1795 | 14M:clip:relu-l1n-1:0.5 |
| relu-l1n-1 + clipping | 0.6 | 6.587026 | 9.0087 | 14M:clip:relu-l1n-1:0.6 |
| relu-l1n-1 + clipping | 0.7 | 7.460709 | 9.8149 | 14M:clip:relu-l1n-1:0.7 |
| relu-l1n-1 + clipping | 0.8 | 8.253852 | 10.6379 | 14M:clip:relu-l1n-1:0.8 |
| relu-l1n-1 + clipping | 0.9 | 8.990051 | 11.2974 | 14M:clip:relu-l1n-1:0.9 |
| relu-ol1-0p05 + clipping | 0.0 | 5.198069 | 3.1452 | 14M:clip:relu-ol1-0p05:0.0 |
| relu-ol1-0p05 + clipping | 0.1 | 5.203818 | 3.9941 | 14M:clip:relu-ol1-0p05:0.1 |
| relu-ol1-0p05 + clipping | 0.2 | 5.243323 | 4.8408 | 14M:clip:relu-ol1-0p05:0.2 |
| relu-ol1-0p05 + clipping | 0.3 | 5.351475 | 5.6874 | 14M:clip:relu-ol1-0p05:0.3 |
| relu-ol1-0p05 + clipping | 0.4 | 5.568223 | 6.5387 | 14M:clip:relu-ol1-0p05:0.4 |
| relu-ol1-0p05 + clipping | 0.5 | 5.967554 | 7.3816 | 14M:clip:relu-ol1-0p05:0.5 |
| relu-ol1-0p05 + clipping | 0.6 | 6.578153 | 8.2633 | 14M:clip:relu-ol1-0p05:0.6 |
| relu-ol1-0p05 + clipping | 0.7 | 7.491310 | 9.1820 | 14M:clip:relu-ol1-0p05:0.7 |
| relu-ol1-0p05 + clipping | 0.8 | 8.518643 | 10.2704 | 14M:clip:relu-ol1-0p05:0.8 |
| relu-ol1-0p05 + clipping | 0.9 | 9.083664 | 11.7052 | 14M:clip:relu-ol1-0p05:0.9 |
| relu-ol1-0p1 + clipping | 0.0 | 5.159387 | 3.3386 | 14M:clip:relu-ol1-0p1:0.0 |
| relu-ol1-0p1 + clipping | 0.1 | 5.165409 | 4.1894 | 14M:clip:relu-ol1-0p1:0.1 |
| relu-ol1-0p1 + clipping | 0.2 | 5.206190 | 5.0342 | 14M:clip:relu-ol1-0p1:0.2 |
| relu-ol1-0p1 + clipping | 0.3 | 5.323405 | 5.8751 | 14M:clip:relu-ol1-0p1:0.3 |
| relu-ol1-0p1 + clipping | 0.4 | 5.575266 | 6.7178 | 14M:clip:relu-ol1-0p1:0.4 |
| relu-ol1-0p1 + clipping | 0.5 | 6.018996 | 7.5556 | 14M:clip:relu-ol1-0p1:0.5 |
| relu-ol1-0p1 + clipping | 0.6 | 6.686208 | 8.3970 | 14M:clip:relu-ol1-0p1:0.6 |
| relu-ol1-0p1 + clipping | 0.7 | 7.647184 | 9.2922 | 14M:clip:relu-ol1-0p1:0.7 |
| relu-ol1-0p1 + clipping | 0.8 | 8.635984 | 10.2201 | 14M:clip:relu-ol1-0p1:0.8 |
| relu-ol1-0p1 + clipping | 0.9 | 9.266252 | 11.5615 | 14M:clip:relu-ol1-0p1:0.9 |
| relu-ol1-0p5 + clipping | 0.0 | 5.110230 | 3.7680 | 14M:clip:relu-ol1-0p5:0.0 |
| relu-ol1-0p5 + clipping | 0.1 | 5.116110 | 4.6156 | 14M:clip:relu-ol1-0p5:0.1 |
| relu-ol1-0p5 + clipping | 0.2 | 5.163076 | 5.4592 | 14M:clip:relu-ol1-0p5:0.2 |
| relu-ol1-0p5 + clipping | 0.3 | 5.291757 | 6.2909 | 14M:clip:relu-ol1-0p5:0.3 |
| relu-ol1-0p5 + clipping | 0.4 | 5.585054 | 7.1044 | 14M:clip:relu-ol1-0p5:0.4 |
| relu-ol1-0p5 + clipping | 0.5 | 6.130628 | 7.9112 | 14M:clip:relu-ol1-0p5:0.5 |
| relu-ol1-0p5 + clipping | 0.6 | 6.968931 | 8.6968 | 14M:clip:relu-ol1-0p5:0.6 |
| relu-ol1-0p5 + clipping | 0.7 | 7.917230 | 9.4801 | 14M:clip:relu-ol1-0p5:0.7 |
| relu-ol1-0p5 + clipping | 0.8 | 8.780806 | 10.3725 | 14M:clip:relu-ol1-0p5:0.8 |
| relu-ol1-0p5 + clipping | 0.9 | 9.262722 | 11.1060 | 14M:clip:relu-ol1-0p5:0.9 |
| relu-ol1-1 + clipping | 0.0 | 5.121213 | 3.9384 | 14M:clip:relu-ol1-1:0.0 |
| relu-ol1-1 + clipping | 0.1 | 5.129136 | 4.7924 | 14M:clip:relu-ol1-1:0.1 |
| relu-ol1-1 + clipping | 0.2 | 5.183859 | 5.6439 | 14M:clip:relu-ol1-1:0.2 |
| relu-ol1-1 + clipping | 0.3 | 5.349408 | 6.4791 | 14M:clip:relu-ol1-1:0.3 |
| relu-ol1-1 + clipping | 0.4 | 5.680604 | 7.2886 | 14M:clip:relu-ol1-1:0.4 |
| relu-ol1-1 + clipping | 0.5 | 6.231464 | 8.0762 | 14M:clip:relu-ol1-1:0.5 |
| relu-ol1-1 + clipping | 0.6 | 7.013678 | 8.8337 | 14M:clip:relu-ol1-1:0.6 |
| relu-ol1-1 + clipping | 0.7 | 7.875197 | 9.5778 | 14M:clip:relu-ol1-1:0.7 |
| relu-ol1-1 + clipping | 0.8 | 8.709804 | 10.3224 | 14M:clip:relu-ol1-1:0.8 |
| relu-ol1-1 + clipping | 0.9 | 9.263879 | 11.0500 | 14M:clip:relu-ol1-1:0.9 |
