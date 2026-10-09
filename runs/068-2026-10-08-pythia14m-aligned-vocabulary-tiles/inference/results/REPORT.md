# Sparse vocabulary head - verified RTX5090 comparison

All values are milliseconds. Three fresh processes per model; 64 fixed inputs x 7 paired passes per process.
The gate, weights and full-vocabulary output contract remain matched. This is B1/T2048 inference.

| Head kappa | Pressure history | Dense head path | No-skip head path | Sparse head path | Head reduction vs dense | Dense forward | Sparse forward | Forward reduction |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|0.1|Ph+head|0.214132|0.339314|0.223645|-4.44%|0.983987|0.994169|-1.03%|
|0.5|Ph+head|0.213253|0.339114|0.180095|+15.55%|1.021812|0.990146|+3.10%|

The no-skip comparison isolates skipping within the custom package; the dense comparison includes layout and preparation differences.
Projection-only timings, individual process ratios, losses and exact error maxima are retained in comparison.json.
One model size, training seed, GPU and workload do not establish decoding or scale-independent performance.
