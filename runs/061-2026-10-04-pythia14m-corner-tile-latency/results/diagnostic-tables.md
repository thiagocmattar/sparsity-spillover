# BF16 site structure and executed h/z work

Each site cell is exact-zero % / empty8x16-tile %. Pool all integer counts across338 blocks and six layers before division.

| Gate | κ | A zeros / tiles % | M zeros / tiles % | H zeros / tiles % | Z zeros / tiles % | H MMA issued % | Z MMA issued % |
|---|---:|---:|---:|---:|---:|---:|---:|
| scalar | 0 | 0.000 / 0.000 | 0.000 / 0.000 | 93.119 / 3.036 | 60.449 / 0.015 | 96.959 | 99.985 |
| tile | 0 | 0.000 / 0.000 | 0.000 / 0.000 | 57.515 / 57.515 | 44.925 / 44.925 | 42.485 | 55.075 |
| scalar | 0.1 | 0.000 / 0.000 | 0.000 / 0.000 | 97.780 / 43.673 | 92.370 / 20.687 | 50.797 | 77.893 |
| tile | 0.1 | 0.000 / 0.000 | 0.000 / 0.000 | 84.362 / 84.362 | 70.974 / 70.974 | 15.638 | 29.026 |

MMA percentages use the padded M8 consumer potential as denominator. Bypassed atoms include short-row SIMT substitution; integer SIMT work is retained in the machine summary.
Tile gates retain fewer element zeros but more empty tiles and bypass more h/z MMA atoms. This accompanies measured full-model gains; it does not measure avoided DRAM bytes or separately attribute the gain to H versus Z.
