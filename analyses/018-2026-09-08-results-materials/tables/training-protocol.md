# Training Protocol

Realized recipe from per-attempt config/manifest. Random initialization and order seed 1234; AdamW (0.9,0.95), eps 1e-8, decay .1, clip norm 1; 1% warmup, cosine to 10% peak. 714 training blocks wrap beyond the complete pass.

| Scale | Parameters | Updates | Input tokens | Global batch | Microbatch | Accumulation | Peak LR | Minimum LR | Precision | Tokens/parameter | A0 loss |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 14M | 14067712 | 712 | 1493172224 | 1024 | 32 | 32 | 0.001 | 0.0001 | float16_dynamic | 106.14 | 5.208573 |
| 70M | 70426624 | 712 | 1493172224 | 1024 | 4 | 256 | 0.001 | 0.0001 | float16_dynamic | 21.20 | 4.099766 |
| 410M | 405334016 | 712 | 1493172224 | 1024 | 4 | 256 | 0.0003 | 3e-05 | float16_dynamic | 3.68 | 4.547456 |
