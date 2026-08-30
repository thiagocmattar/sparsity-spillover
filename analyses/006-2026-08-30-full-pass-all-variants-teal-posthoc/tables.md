# Analysis 006 complete target-sparsity table

TEAL is uniform, per-checkpoint, per-site-layer magnitude clipping applied only
during evaluation. All points retain complete validation evidence; `visible` only
describes the loss-6 figure cap.

| source condition | trained family | trained dose | TEAL target | validation loss | loss delta | R_model (%) | visible | TEAL envelope |
| --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | :---: |
| gelu-control | GeLU control | - | 0.0 | 5.208583 | +0.000000 | 0.000001 | yes | no |
| gelu-control | GeLU control | - | 0.1 | 5.214453 | +0.005859 | 1.278706 | yes | no |
| gelu-control | GeLU control | - | 0.2 | 5.250270 | +0.041676 | 2.554786 | yes | no |
| gelu-control | GeLU control | - | 0.3 | 5.361901 | +0.153307 | 3.831641 | yes | no |
| gelu-control | GeLU control | - | 0.4 | 5.605384 | +0.396790 | 5.125544 | yes | no |
| gelu-control | GeLU control | - | 0.5 | 6.077731 | +0.869137 | 6.441923 | no | no |
| gelu-control | GeLU control | - | 0.6 | 6.821173 | +1.612579 | 7.764415 | no | no |
| gelu-control | GeLU control | - | 0.7 | 7.766948 | +2.558354 | 9.123471 | no | no |
| gelu-control | GeLU control | - | 0.8 | 8.336118 | +3.127524 | 10.585958 | no | no |
| gelu-control | GeLU control | - | 0.9 | 8.825610 | +3.617016 | 11.941881 | no | no |
| relu-control | A1-H ReLU control | - | 0.0 | 5.269646 | +0.000000 | 2.714130 | yes | no |
| relu-control | A1-H ReLU control | - | 0.1 | 5.274073 | +0.004439 | 3.568403 | yes | no |
| relu-control | A1-H ReLU control | - | 0.2 | 5.307940 | +0.038307 | 4.418365 | yes | no |
| relu-control | A1-H ReLU control | - | 0.3 | 5.405056 | +0.135422 | 5.271871 | yes | no |
| relu-control | A1-H ReLU control | - | 0.4 | 5.602965 | +0.333331 | 6.129325 | yes | no |
| relu-control | A1-H ReLU control | - | 0.5 | 5.976274 | +0.706641 | 6.994451 | yes | no |
| relu-control | A1-H ReLU control | - | 0.6 | 6.521474 | +1.251840 | 7.892904 | no | no |
| relu-control | A1-H ReLU control | - | 0.7 | 7.306175 | +2.036542 | 9.051787 | no | no |
| relu-control | A1-H ReLU control | - | 0.8 | 8.107618 | +2.837985 | 10.558519 | no | no |
| relu-control | A1-H ReLU control | - | 0.9 | 8.869450 | +3.599817 | 11.878925 | no | no |
| relu-l1n-0p05 | A1-H naive L1 | 0.05 | 0.0 | 5.206149 | +0.000000 | 3.141595 | yes | no |
| relu-l1n-0p05 | A1-H naive L1 | 0.05 | 0.1 | 5.211364 | +0.005207 | 3.991446 | yes | no |
| relu-l1n-0p05 | A1-H naive L1 | 0.05 | 0.2 | 5.248624 | +0.042467 | 4.836147 | yes | no |
| relu-l1n-0p05 | A1-H naive L1 | 0.05 | 0.3 | 5.350951 | +0.144794 | 5.684304 | yes | no |
| relu-l1n-0p05 | A1-H naive L1 | 0.05 | 0.4 | 5.568858 | +0.362701 | 6.538933 | yes | no |
| relu-l1n-0p05 | A1-H naive L1 | 0.05 | 0.5 | 5.973093 | +0.766936 | 7.395678 | yes | no |
| relu-l1n-0p05 | A1-H naive L1 | 0.05 | 0.6 | 6.600953 | +1.394796 | 8.274846 | no | no |
| relu-l1n-0p05 | A1-H naive L1 | 0.05 | 0.7 | 7.435047 | +2.228890 | 9.205014 | no | no |
| relu-l1n-0p05 | A1-H naive L1 | 0.05 | 0.8 | 8.427544 | +3.221387 | 10.286104 | no | no |
| relu-l1n-0p05 | A1-H naive L1 | 0.05 | 0.9 | 9.082812 | +3.876655 | 11.711245 | no | no |
| relu-l1n-0p1 | A1-H naive L1 | 0.1 | 0.0 | 5.165474 | +0.000000 | 3.336306 | yes | no |
| relu-l1n-0p1 | A1-H naive L1 | 0.1 | 0.1 | 5.170191 | +0.004725 | 4.186238 | yes | no |
| relu-l1n-0p1 | A1-H naive L1 | 0.1 | 0.2 | 5.207756 | +0.042290 | 5.033392 | yes | no |
| relu-l1n-0p1 | A1-H naive L1 | 0.1 | 0.3 | 5.315055 | +0.149589 | 5.880440 | yes | no |
| relu-l1n-0p1 | A1-H naive L1 | 0.1 | 0.4 | 5.550581 | +0.385115 | 6.727546 | yes | no |
| relu-l1n-0p1 | A1-H naive L1 | 0.1 | 0.5 | 5.979635 | +0.814169 | 7.580657 | yes | no |
| relu-l1n-0p1 | A1-H naive L1 | 0.1 | 0.6 | 6.633155 | +1.467689 | 8.450807 | no | no |
| relu-l1n-0p1 | A1-H naive L1 | 0.1 | 0.7 | 7.553385 | +2.387919 | 9.361026 | no | no |
| relu-l1n-0p1 | A1-H naive L1 | 0.1 | 0.8 | 8.650510 | +3.485043 | 10.283422 | no | no |
| relu-l1n-0p1 | A1-H naive L1 | 0.1 | 0.9 | 9.159795 | +3.994328 | 11.586003 | no | no |
| relu-l1n-0p5 | A1-H naive L1 | 0.5 | 0.0 | 5.112688 | +0.000000 | 3.787578 | yes | no |
| relu-l1n-0p5 | A1-H naive L1 | 0.5 | 0.1 | 5.117628 | +0.004945 | 4.636960 | yes | no |
| relu-l1n-0p5 | A1-H naive L1 | 0.5 | 0.2 | 5.155895 | +0.043212 | 5.478188 | yes | no |
| relu-l1n-0p5 | A1-H naive L1 | 0.5 | 0.3 | 5.271484 | +0.158801 | 6.317201 | yes | no |
| relu-l1n-0p5 | A1-H naive L1 | 0.5 | 0.4 | 5.510881 | +0.398198 | 7.161994 | yes | no |
| relu-l1n-0p5 | A1-H naive L1 | 0.5 | 0.5 | 5.986626 | +0.873943 | 7.998942 | yes | no |
| relu-l1n-0p5 | A1-H naive L1 | 0.5 | 0.6 | 6.687065 | +1.574382 | 8.843648 | no | no |
| relu-l1n-0p5 | A1-H naive L1 | 0.5 | 0.7 | 7.495898 | +2.383215 | 9.675529 | no | no |
| relu-l1n-0p5 | A1-H naive L1 | 0.5 | 0.8 | 8.327347 | +3.214664 | 10.544146 | no | no |
| relu-l1n-0p5 | A1-H naive L1 | 0.5 | 0.9 | 9.188822 | +4.076139 | 11.179762 | no | no |
| relu-l1n-1 | A1-H naive L1 | 1 | 0.0 | 5.102276 | +0.000000 | 3.949334 | yes | yes |
| relu-l1n-1 | A1-H naive L1 | 1 | 0.1 | 5.106970 | +0.004700 | 4.798234 | yes | yes |
| relu-l1n-1 | A1-H naive L1 | 1 | 0.2 | 5.144074 | +0.041804 | 5.644120 | yes | yes |
| relu-l1n-1 | A1-H naive L1 | 1 | 0.3 | 5.245573 | +0.143304 | 6.490231 | yes | yes |
| relu-l1n-1 | A1-H naive L1 | 1 | 0.4 | 5.470511 | +0.368241 | 7.346197 | yes | no |
| relu-l1n-1 | A1-H naive L1 | 1 | 0.5 | 5.898911 | +0.796641 | 8.179541 | yes | no |
| relu-l1n-1 | A1-H naive L1 | 1 | 0.6 | 6.587026 | +1.484756 | 9.008673 | no | no |
| relu-l1n-1 | A1-H naive L1 | 1 | 0.7 | 7.460709 | +2.358439 | 9.814948 | no | no |
| relu-l1n-1 | A1-H naive L1 | 1 | 0.8 | 8.253852 | +3.151582 | 10.637916 | no | no |
| relu-l1n-1 | A1-H naive L1 | 1 | 0.9 | 8.990051 | +3.887781 | 11.297370 | no | no |
| relu-ol1-0p05 | A1-H OL1 | 0.05 | 0.0 | 5.198062 | +0.000000 | 3.145227 | yes | no |
| relu-ol1-0p05 | A1-H OL1 | 0.05 | 0.1 | 5.203818 | +0.005749 | 3.994120 | yes | no |
| relu-ol1-0p05 | A1-H OL1 | 0.05 | 0.2 | 5.243323 | +0.045254 | 4.840796 | yes | no |
| relu-ol1-0p05 | A1-H OL1 | 0.05 | 0.3 | 5.351475 | +0.153405 | 5.687407 | yes | no |
| relu-ol1-0p05 | A1-H OL1 | 0.05 | 0.4 | 5.568223 | +0.370154 | 6.538742 | yes | no |
| relu-ol1-0p05 | A1-H OL1 | 0.05 | 0.5 | 5.967554 | +0.769484 | 7.381562 | yes | no |
| relu-ol1-0p05 | A1-H OL1 | 0.05 | 0.6 | 6.578153 | +1.380084 | 8.263337 | no | no |
| relu-ol1-0p05 | A1-H OL1 | 0.05 | 0.7 | 7.491310 | +2.293240 | 9.181971 | no | no |
| relu-ol1-0p05 | A1-H OL1 | 0.05 | 0.8 | 8.518643 | +3.320574 | 10.270378 | no | no |
| relu-ol1-0p05 | A1-H OL1 | 0.05 | 0.9 | 9.083664 | +3.885595 | 11.705248 | no | no |
| relu-ol1-0p1 | A1-H OL1 | 0.1 | 0.0 | 5.159391 | +0.000000 | 3.338579 | yes | no |
| relu-ol1-0p1 | A1-H OL1 | 0.1 | 0.1 | 5.165409 | +0.006022 | 4.189356 | yes | no |
| relu-ol1-0p1 | A1-H OL1 | 0.1 | 0.2 | 5.206190 | +0.046803 | 5.034234 | yes | no |
| relu-ol1-0p1 | A1-H OL1 | 0.1 | 0.3 | 5.323405 | +0.164017 | 5.875125 | yes | no |
| relu-ol1-0p1 | A1-H OL1 | 0.1 | 0.4 | 5.575266 | +0.415879 | 6.717803 | yes | no |
| relu-ol1-0p1 | A1-H OL1 | 0.1 | 0.5 | 6.018996 | +0.859609 | 7.555569 | no | no |
| relu-ol1-0p1 | A1-H OL1 | 0.1 | 0.6 | 6.686208 | +1.526821 | 8.396965 | no | no |
| relu-ol1-0p1 | A1-H OL1 | 0.1 | 0.7 | 7.647184 | +2.487797 | 9.292222 | no | no |
| relu-ol1-0p1 | A1-H OL1 | 0.1 | 0.8 | 8.635984 | +3.476597 | 10.220104 | no | no |
| relu-ol1-0p1 | A1-H OL1 | 0.1 | 0.9 | 9.266252 | +4.106864 | 11.561463 | no | no |
| relu-ol1-0p5 | A1-H OL1 | 0.5 | 0.0 | 5.110235 | +0.000000 | 3.767998 | yes | no |
| relu-ol1-0p5 | A1-H OL1 | 0.5 | 0.1 | 5.116110 | +0.005880 | 4.615573 | yes | no |
| relu-ol1-0p5 | A1-H OL1 | 0.5 | 0.2 | 5.163076 | +0.052846 | 5.459234 | yes | no |
| relu-ol1-0p5 | A1-H OL1 | 0.5 | 0.3 | 5.291757 | +0.181526 | 6.290894 | yes | no |
| relu-ol1-0p5 | A1-H OL1 | 0.5 | 0.4 | 5.585054 | +0.474824 | 7.104435 | yes | no |
| relu-ol1-0p5 | A1-H OL1 | 0.5 | 0.5 | 6.130628 | +1.020397 | 7.911152 | no | no |
| relu-ol1-0p5 | A1-H OL1 | 0.5 | 0.6 | 6.968931 | +1.858701 | 8.696843 | no | no |
| relu-ol1-0p5 | A1-H OL1 | 0.5 | 0.7 | 7.917230 | +2.807000 | 9.480073 | no | no |
| relu-ol1-0p5 | A1-H OL1 | 0.5 | 0.8 | 8.780806 | +3.670576 | 10.372512 | no | no |
| relu-ol1-0p5 | A1-H OL1 | 0.5 | 0.9 | 9.262722 | +4.152492 | 11.105975 | no | no |
| relu-ol1-1 | A1-H OL1 | 1 | 0.0 | 5.121030 | +0.000000 | 3.938363 | yes | no |
| relu-ol1-1 | A1-H OL1 | 1 | 0.1 | 5.129136 | +0.007923 | 4.792441 | yes | no |
| relu-ol1-1 | A1-H OL1 | 1 | 0.2 | 5.183859 | +0.062646 | 5.643851 | yes | no |
| relu-ol1-1 | A1-H OL1 | 1 | 0.3 | 5.349408 | +0.228195 | 6.479110 | yes | no |
| relu-ol1-1 | A1-H OL1 | 1 | 0.4 | 5.680604 | +0.559391 | 7.288607 | yes | no |
| relu-ol1-1 | A1-H OL1 | 1 | 0.5 | 6.231464 | +1.110251 | 8.076218 | no | no |
| relu-ol1-1 | A1-H OL1 | 1 | 0.6 | 7.013678 | +1.892465 | 8.833679 | no | no |
| relu-ol1-1 | A1-H OL1 | 1 | 0.7 | 7.875197 | +2.753984 | 9.577840 | no | no |
| relu-ol1-1 | A1-H OL1 | 1 | 0.8 | 8.709804 | +3.588591 | 10.322399 | no | no |
| relu-ol1-1 | A1-H OL1 | 1 | 0.9 | 9.263879 | +4.142666 | 11.050013 | no | no |
| a4z-one-sided-kappa-0 | A4-Z threshold | 0 | 0.0 | 5.470497 | +0.000000 | 7.212019 | yes | no |
| a4z-one-sided-kappa-0 | A4-Z threshold | 0 | 0.1 | 5.470511 | +0.000000 | 7.212024 | yes | no |
| a4z-one-sided-kappa-0 | A4-Z threshold | 0 | 0.2 | 5.470511 | +0.000000 | 7.212024 | yes | no |
| a4z-one-sided-kappa-0 | A4-Z threshold | 0 | 0.3 | 5.470511 | +0.000000 | 7.212024 | yes | no |
| a4z-one-sided-kappa-0 | A4-Z threshold | 0 | 0.4 | 5.470428 | -0.000083 | 7.218927 | yes | no |
| a4z-one-sided-kappa-0 | A4-Z threshold | 0 | 0.5 | 5.470153 | -0.000358 | 7.309896 | yes | no |
| a4z-one-sided-kappa-0 | A4-Z threshold | 0 | 0.6 | 5.539702 | +0.069191 | 8.078262 | yes | no |
| a4z-one-sided-kappa-0 | A4-Z threshold | 0 | 0.7 | 5.957410 | +0.486898 | 8.956409 | yes | no |
| a4z-one-sided-kappa-0 | A4-Z threshold | 0 | 0.8 | 6.974517 | +1.504006 | 10.214867 | no | no |
| a4z-one-sided-kappa-0 | A4-Z threshold | 0 | 0.9 | 8.247837 | +2.777326 | 11.630631 | no | no |
| a4z-one-sided-kappa-0p01 | A4-Z threshold | 0.01 | 0.0 | 5.466500 | +0.000000 | 7.413704 | yes | no |
| a4z-one-sided-kappa-0p01 | A4-Z threshold | 0.01 | 0.1 | 5.466524 | +0.000000 | 7.413702 | yes | no |
| a4z-one-sided-kappa-0p01 | A4-Z threshold | 0.01 | 0.2 | 5.466524 | +0.000000 | 7.413702 | yes | no |
| a4z-one-sided-kappa-0p01 | A4-Z threshold | 0.01 | 0.3 | 5.466524 | +0.000000 | 7.413702 | yes | no |
| a4z-one-sided-kappa-0p01 | A4-Z threshold | 0.01 | 0.4 | 5.466524 | +0.000000 | 7.413702 | yes | no |
| a4z-one-sided-kappa-0p01 | A4-Z threshold | 0.01 | 0.5 | 5.466257 | -0.000266 | 7.456461 | yes | no |
| a4z-one-sided-kappa-0p01 | A4-Z threshold | 0.01 | 0.6 | 5.534816 | +0.068293 | 8.178306 | yes | no |
| a4z-one-sided-kappa-0p01 | A4-Z threshold | 0.01 | 0.7 | 5.958228 | +0.491705 | 9.002739 | yes | no |
| a4z-one-sided-kappa-0p01 | A4-Z threshold | 0.01 | 0.8 | 6.988377 | +1.521853 | 10.196830 | no | no |
| a4z-one-sided-kappa-0p01 | A4-Z threshold | 0.01 | 0.9 | 8.158731 | +2.692208 | 11.621385 | no | no |
| a4z-one-sided-kappa-0p05 | A4-Z threshold | 0.05 | 0.0 | 5.434110 | +0.000000 | 8.205893 | yes | no |
| a4z-one-sided-kappa-0p05 | A4-Z threshold | 0.05 | 0.1 | 5.434159 | +0.000000 | 8.205886 | yes | no |
| a4z-one-sided-kappa-0p05 | A4-Z threshold | 0.05 | 0.2 | 5.434159 | +0.000000 | 8.205886 | yes | no |
| a4z-one-sided-kappa-0p05 | A4-Z threshold | 0.05 | 0.3 | 5.434159 | +0.000000 | 8.205886 | yes | no |
| a4z-one-sided-kappa-0p05 | A4-Z threshold | 0.05 | 0.4 | 5.434159 | +0.000000 | 8.205886 | yes | no |
| a4z-one-sided-kappa-0p05 | A4-Z threshold | 0.05 | 0.5 | 5.434098 | -0.000060 | 8.212998 | yes | no |
| a4z-one-sided-kappa-0p05 | A4-Z threshold | 0.05 | 0.6 | 5.479514 | +0.045356 | 8.758281 | yes | no |
| a4z-one-sided-kappa-0p05 | A4-Z threshold | 0.05 | 0.7 | 5.864863 | +0.430704 | 9.416940 | yes | no |
| a4z-one-sided-kappa-0p05 | A4-Z threshold | 0.05 | 0.8 | 6.693103 | +1.258944 | 10.191344 | no | no |
| a4z-one-sided-kappa-0p05 | A4-Z threshold | 0.05 | 0.9 | 8.091245 | +2.657086 | 11.462236 | no | no |
| a4z-one-sided-kappa-0p1 | A4-Z threshold | 0.1 | 0.0 | 5.419642 | +0.000000 | 8.953005 | yes | no |
| a4z-one-sided-kappa-0p1 | A4-Z threshold | 0.1 | 0.1 | 5.419630 | +0.000000 | 8.953012 | yes | no |
| a4z-one-sided-kappa-0p1 | A4-Z threshold | 0.1 | 0.2 | 5.419630 | +0.000000 | 8.953012 | yes | no |
| a4z-one-sided-kappa-0p1 | A4-Z threshold | 0.1 | 0.3 | 5.419630 | +0.000000 | 8.953012 | yes | no |
| a4z-one-sided-kappa-0p1 | A4-Z threshold | 0.1 | 0.4 | 5.419630 | +0.000000 | 8.953012 | yes | no |
| a4z-one-sided-kappa-0p1 | A4-Z threshold | 0.1 | 0.5 | 5.419605 | -0.000025 | 8.953740 | yes | yes |
| a4z-one-sided-kappa-0p1 | A4-Z threshold | 0.1 | 0.6 | 5.446599 | +0.026968 | 9.342357 | yes | yes |
| a4z-one-sided-kappa-0p1 | A4-Z threshold | 0.1 | 0.7 | 5.775362 | +0.355731 | 10.011998 | yes | no |
| a4z-one-sided-kappa-0p1 | A4-Z threshold | 0.1 | 0.8 | 6.579617 | +1.159986 | 10.655358 | no | no |
| a4z-one-sided-kappa-0p1 | A4-Z threshold | 0.1 | 0.9 | 7.759894 | +2.340263 | 11.378151 | no | no |
| a4z-one-sided-kappa-0p5 | A4-Z threshold | 0.5 | 0.0 | 5.659680 | +0.000000 | 10.215537 | yes | yes |
| a4z-one-sided-kappa-0p5 | A4-Z threshold | 0.5 | 0.1 | 5.659684 | +0.000000 | 10.215435 | yes | no |
| a4z-one-sided-kappa-0p5 | A4-Z threshold | 0.5 | 0.2 | 5.659684 | +0.000000 | 10.215435 | yes | no |
| a4z-one-sided-kappa-0p5 | A4-Z threshold | 0.5 | 0.3 | 5.659684 | +0.000000 | 10.215435 | yes | no |
| a4z-one-sided-kappa-0p5 | A4-Z threshold | 0.5 | 0.4 | 5.659684 | +0.000000 | 10.215435 | yes | no |
| a4z-one-sided-kappa-0p5 | A4-Z threshold | 0.5 | 0.5 | 5.659684 | +0.000000 | 10.215435 | yes | no |
| a4z-one-sided-kappa-0p5 | A4-Z threshold | 0.5 | 0.6 | 5.660417 | +0.000733 | 10.227092 | yes | yes |
| a4z-one-sided-kappa-0p5 | A4-Z threshold | 0.5 | 0.7 | 5.705777 | +0.046093 | 10.563148 | yes | yes |
| a4z-one-sided-kappa-0p5 | A4-Z threshold | 0.5 | 0.8 | 6.099449 | +0.439765 | 11.325651 | no | no |
| a4z-one-sided-kappa-0p5 | A4-Z threshold | 0.5 | 0.9 | 6.824761 | +1.165076 | 12.067310 | no | no |

Every row covers 500 validation documents, 338 complete blocks, and 692,224
input tokens; the 1,444-token tail is excluded. `R_model` is logical-product
opportunity, not measured speedup. One seed.
