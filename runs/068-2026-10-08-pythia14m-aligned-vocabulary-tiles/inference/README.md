# Run068 vocabulary-head endpoint benchmark

Completed on one RTX 5090 with the two fresh step-712 Run068 checkpoints.
The 16-token × 16-feature head gate matches the M16/K16 custom consumer;
head κ is 0.1/0.5 and h/z κ stays 0. All h/z projection kernels remain native
dense. The three modes share each checkpoint, gate and all 50,304 output logits:
native dense, custom no-skip and custom sparse.

All six processes qualify exactly against native logits and loss over all 338
validation blocks. All 83 returned files are hash-verified locally; the owned
Pod is deleted. The sparse head path is 4.44% slower / 15.55% faster than dense
at κ=0.1/0.5; complete forward is 1.03% slower / 3.10% faster. Threshold training
changes model quality, reported separately from within-checkpoint numerical
preservation.

- [Consolidated scientific report and PDFs](../results/REPORT.md)
- [Endpoint reduction](results/REPORT.md) and [all process values](results/comparison.json)
- [Raw full processes](artifacts/attempts/), [qualification](artifacts/verification-001.json)
- [Sealed inputs](provenance/trained-upload-inventory.json) and [retrieval](transfer/receipt-001.json)
- [Execution, costs and teardown](../CLOSEOUT.md)

Each process uses 64 fixed inputs × seven randomized paired passes per timing
boundary. Projection-only, gate+cast+projection and complete-forward timings
overlap and cannot be added as a decomposition. Host synchronization is primary;
CUDA events use separate replays. Softmax, token selection, loss, input staging
and static weight packing are outside timing. Only full-sequence B1/T2048 is
measured; cached decoding and integration with earlier sparse block kernels
remain outside this experiment.

`10_retrieve_remote.py` is an operational downloader added after sealing; it
does not participate in model execution or the 81-file runtime inventory.
