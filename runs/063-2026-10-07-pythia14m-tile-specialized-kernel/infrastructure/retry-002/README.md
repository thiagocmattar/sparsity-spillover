# Pinned-runtime network retry 002

Setup001 stopped before any GPU check or scientific process. uv0.8.15 failed
to download/extract pinned `nvidia-nccl-cu12==2.28.9` under its30-second HTTP
timeout. The failed pipeline log, exit and PID were copied into immutable
`artifacts/attempts/setup-001/`; eight returned files passed local byte/SHA256
verification in `transfer/receipt-001.json`.
The initial return is also preserved under `retrieval-001/`, with all eight
original member paths/hashes and its inventory. Use that directory as the
inventory root for the historical snapshot: the live controller paths at the
run root subsequently contain the successful retry's logs.

[start.py](start.py) resumes on the same owned Pod and physical RTX5090. It
preserves the failed controller files and incomplete application venv, then
runs the original setup and execution scripts with HTTP timeout 300seconds,
five HTTP retries and two concurrent downloads. Package pins, sealed bundle,
CUDA sources, checkpoints, numerical bounds and measurement protocol stay
unchanged. The existing deadline guards remain armed and the original two-hour/
$3 envelope is retained. There is no second Pod.

Smoke and scientific attempts use fresh tag002. A setup failure is not a
numerical failure or benchmark result. Accepted reduction must use tag002 and
the corresponding second transfer inventory.
