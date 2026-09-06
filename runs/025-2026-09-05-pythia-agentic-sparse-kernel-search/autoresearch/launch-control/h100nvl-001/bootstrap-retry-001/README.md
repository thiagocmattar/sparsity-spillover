# H100 NVL bootstrap retry 001

The first bootstrap exited after verified extraction and before creating any
scientific phase artifacts. The cause was an infrastructure setup ordering bug:
`command -v nvcc` ran before `/usr/local/cuda/bin` was added to `PATH`. The
compiler exists at `/usr/local/cuda/bin/nvcc` on Pod `5du99d07qom9vk`.

This retry reuses the already hash-verified and extracted payload on that Pod.
It changes no checkpoint, model, input, kernel, timing, correctness, validation,
or phase definition. The failed bootstrap control directory is preserved
remotely as `bootstrap-control-attempt001`; the successful retry continues to
use `bootstrap-control` so the pre-verified closeout remains unchanged.
