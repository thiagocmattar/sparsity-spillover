# Sparse vocabulary head — verified RTX5090 comparison

All values are milliseconds. Three fresh processes per model; 64 fixed inputs × 7 paired passes per process.
The gate, weights and full-vocabulary output contract remain matched. This is B1/T2048 inference.

| κ | Pressure history | Dense head path | No-skip head path | Sparse head path | Head reduction vs dense | Dense forward | Sparse forward | Forward reduction |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
|0|Ph|0.202762|0.331052|0.279838|-38.01%|0.998697|1.074223|-7.56%|
|0|Ph+head|0.203757|0.331502|0.275828|-35.37%|0.999435|1.071257|-7.19%|
|0.1|Ph|0.202987|0.331294|0.293425|-44.55%|0.999336|1.087872|-8.86%|
|0.1|Ph+head|0.204100|0.332040|0.291100|-42.63%|1.001191|1.086614|-8.53%|

The no-skip comparison isolates skipping within the custom package; the dense comparison includes layout and preparation differences.
Projection-only timings, individual process ratios, losses and exact error maxima are retained in comparison.json.
One model size, training seed, GPU and workload do not establish decoding or scale-independent performance.
