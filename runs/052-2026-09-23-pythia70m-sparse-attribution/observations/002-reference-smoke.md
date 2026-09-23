# Target-GPU reference smoke

Question: do the retained14M/70M references and their dense/skipping controls
execute on the new RTX5090 within the unchanged numerical bounds?

Method: `09_execute.py --stage smoke`, attempt `smoke-002`, using the pinned
environment and unchanged `reference-policy.json`. Each of the five checkpoints
uses four literal training blocks (8,192 input and8,188 prediction tokens),
native eager reference, BF16 CUDA graphs and complete output logits. This is
not full validation and is not used to select candidates or claim speedup.

All reference implementations passed. Native pooled smoke losses were:

| Size/condition | Kappa | Native smoke loss | Comparison blocks/s |
| --- | ---: | ---: | ---: |
|14M T2/Ph|.05|5.143221|8.14|
|14M T2/Ph|.1|5.168273|7.66|
|70M T2/Ph|.05|3.982362|9.06|
|70M T2/Ph|.1|3.959456|9.38|
|70M Base|none|3.893993|8.23|

Each custom reference's pooled loss matched its native result on these four
blocks; per-logit and relative-L2 gates also passed. Throughput here covers
comparison of all backends per block, not single-model throughput. Cold builds
dominated startup; the completed stage took1,401seconds. Warm checkpoint
preparation still takes substantial time and is included in the stage ETC.

Infrastructure attempt `smoke-001` stopped because Ninja was installed in the
virtualenv but that bin directory was absent from PATH. The new runtime wrapper
fixes PATH; no packages, kernels, gates or checkpoints changed. Failed logs remain
retained. The returned archive contains93 files; every byte/hash was verified
locally and51 artifact files were integrated under `artifacts/`.

The new E/F candidates have not yet been compiled or qualified. A passing
reference smoke does not establish their correctness or performance, nor does
it establish either cross-size threshold response or sparse attribution.

Evidence: [smoke summary](../results/smoke-summary.json) gives exact values,
qualification maps and source SHA256 records; [retrieval verification](../prelaunch/smoke-retrieval-verification.json)
pins the returned archive. Source scripts: `06_benchmark.py`, `09_execute.py`,
`13_collect.py` and `14_verify_archive.py`. No figure or manuscript change.
