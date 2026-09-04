# Attempt 004 launch adjustment

Attempt 003 passed the pinned upstream positive control (TwELL
`476.9542 ms`; Torch `620.0687 ms`; `1.3001x`) and stopped in the Run 022
synthetic ELL preflight before A0 measurement. CUDA PyTorch does not implement
boolean advanced indexing on `UInt16`; the same operation passed the local CPU
test.

The attempt-local repair changes only the order of the two mask operations on
ELL column indices (contract range validation and exact unpacking):

```text
indices[valid].to(torch.int32)
indices.to(torch.int32)[valid]
indices[valid].to(torch.long)
indices.to(torch.long)[valid]
```

Both expressions select the same ELL column indices. The latter uses CUDA's
supported `Int32` indexing path. Before attempt 004, the full remote synthetic
preflight must pass, including overflow rejection, empty and dense rows, exact
pack round-trip, output-width constraints, and numerical correctness.
