# Analysis 005 tables

Every fraction below is recalculated from stored integer counts. Loss deltas are paired
to the target-zero point from the same control sweep. `Global frontier` is nondominance
over all twenty GeLU and ReLU points under lower loss and higher measured `R_model`.

| control | target sparsity | validation loss | loss delta | R_model (%) | a zero (%) | m zero (%) | h zero (%) | z zero (%) | global frontier |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| GeLU | 0.0 | 5.208594 | +0.000000 | 0.000001 | 0.000 | 0.000 | 0.000 | 0.000 | yes |
| GeLU | 0.1 | 5.214453 | +0.005859 | 1.278706 | 9.904 | 9.912 | 9.991 | 10.243 | yes |
| GeLU | 0.2 | 5.250270 | +0.041676 | 2.554786 | 19.742 | 19.758 | 20.000 | 20.635 | yes |
| GeLU | 0.3 | 5.361901 | +0.153307 | 3.831641 | 29.491 | 29.522 | 30.112 | 31.281 | no |
| GeLU | 0.4 | 5.605384 | +0.396790 | 5.125544 | 39.207 | 39.259 | 40.477 | 42.716 | no |
| GeLU | 0.5 | 6.077731 | +0.869137 | 6.441923 | 48.898 | 48.981 | 51.236 | 54.807 | no |
| GeLU | 0.6 | 6.821173 | +1.612579 | 7.764415 | 58.291 | 58.442 | 62.608 | 66.963 | no |
| GeLU | 0.7 | 7.766948 | +2.558354 | 9.123471 | 67.874 | 68.155 | 74.686 | 78.129 | yes |
| GeLU | 0.8 | 8.336118 | +3.127524 | 10.585958 | 79.021 | 79.325 | 85.982 | 91.578 | yes |
| GeLU | 0.9 | 8.825610 | +3.617016 | 11.941881 | 90.200 | 90.245 | 96.294 | 99.906 | yes |
| ReLU | 0.0 | 5.269633 | +0.000000 | 2.714133 | 0.000 | 0.000 | 63.448 | 0.000 | yes |
| ReLU | 0.1 | 5.274073 | +0.004439 | 3.568403 | 9.985 | 9.985 | 63.427 | 10.070 | yes |
| ReLU | 0.2 | 5.307940 | +0.038307 | 4.418365 | 19.896 | 19.897 | 63.406 | 20.250 | yes |
| ReLU | 0.3 | 5.405056 | +0.135422 | 5.271871 | 29.758 | 29.766 | 63.451 | 30.820 | yes |
| ReLU | 0.4 | 5.602965 | +0.333331 | 6.129325 | 39.613 | 39.643 | 63.470 | 41.849 | yes |
| ReLU | 0.5 | 5.976274 | +0.706641 | 6.994451 | 49.332 | 49.378 | 63.671 | 53.842 | yes |
| ReLU | 0.6 | 6.521474 | +1.251840 | 7.892904 | 58.907 | 58.988 | 64.640 | 66.819 | yes |
| ReLU | 0.7 | 7.306175 | +2.036542 | 9.051787 | 68.608 | 68.777 | 71.109 | 81.045 | yes |
| ReLU | 0.8 | 8.107618 | +2.837985 | 10.558519 | 79.665 | 79.831 | 83.277 | 95.875 | yes |
| ReLU | 0.9 | 8.869450 | +3.599817 | 11.878925 | 90.009 | 89.973 | 95.236 | 99.911 | no |

Coverage for every row: 500 validation documents, 338 complete 2,048-token blocks,
692,224 input tokens, and a reported 1,444-token excluded tail. There is one seed.
`R_model` is a logical zero-product opportunity, not measured speedup.
