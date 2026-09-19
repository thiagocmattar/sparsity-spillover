# Pooled structure and executed work

| Site | Sparsity (%) | Rows with <=2 nonzeros (%) | Mean nonzeros/row | Frozen issued MMA | Port issued MMA |
|---|---:|---:|---:|---:|---:|
| a | 75.6292 | 15.3706 | 31.1947 | 91,268,736 | 172,966,272 |
| m | 68.6494 | 0.0000 | 40.1288 | 132,907,008 | 265,814,016 |
| h | 99.8755 | 92.7546 | 0.6376 | 14,626,368 | 14,626,368 |
| z | 99.9170 | 99.7254 | 0.1062 | 235,984 | 235,984 |

| Site | Empty 16x16 tiles (%) | Empty 8x16 tiles (%) | Weight requests avoided within port layout (%) |
|---|---:|---:|---:|
| a | 8.4385 | 11.1218 | 12.9886 |
| m | 0.0000 | 0.0000 | 0.0000 |

BF16 executed operands; raw integer counts pooled before division. Source weight requests are not measured DRAM traffic. Bypass includes short-row scalar substitution. Port uses twice as many padded M16 opportunities at a/m as the frozen M16 layout.
