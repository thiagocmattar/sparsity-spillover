# Attempt 002 outcome

The Ninja path correction worked. The official SparseLM0.5B control passed
(`475.739 ms` TwELL versus `604.049 ms` Torch, `1.270x`), the derived seven-shape
preflight passed, and batch 32 retained 15.20% memory headroom. A0 completed
all three validation passes and both full-model timing batches.

The worker then stopped before the first linear primitive was serialized.
Captured activation samples are inference tensors because validation runs under
`torch.inference_mode()`, but the frozen base Run-023 primitive function called
`torch.nn.functional.linear` outside an inference context. PyTorch correctly
rejected saving an inference tensor for backward.

Review of Run 023 showed that its accepted Attempt 005 used two documented
runtime patches that had not been folded into the base source: the exact
inference-context correction and an eager-attention context for canonical q/v
logical counts. Attempt 003 applies those already-validated semantics to the
Run-024 wrapper and adds a fail-closed verifier assertion. It changes no model,
data, kernel arithmetic, runtime timing context, batch, repetition, or tolerance.
Partial Attempt-002 outputs are not reused.
