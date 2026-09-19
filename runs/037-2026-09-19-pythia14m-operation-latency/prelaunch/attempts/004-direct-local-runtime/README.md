# Infrastructure adjustment 004: direct local executable

After all ten smoke processes completed and qualified, the idle original
shell was replaced by a continuation worker. Its recorded exit137 is the
intentional shell replacement, not a failed scientific evaluation. No GPU
child was interrupted. All 20 direct CUDA checks and ten full-model smoke
results are retained.

The continuation invokes the unchanged `03_execute.py`, `07_reduce.py` and
`08_collect.py` with `/tmp/run037-local/venv/bin/python`. The controller retains
the same seed2504 job order and all thirty original mode/replicate pairs.
Every scientific process uses this same environment and the same physical GPU.

The small `bridge/sitecustomize.py` retains PyTorch's original compiler
include/library paths, which resolve to the verified identical local files.
This prevents rebuilding the same CUDA extensions merely because the Python
executable moved. It changes no operator, tolerance, gate, weight, validation
coverage, timing input or repetition. `compiler-paths.json` records the paths.
The runtime/library sources were SHA256-verified during adjustment003.

Both the scientific deadline and the independent local/remote compute-stop
guards retain the approved original times. The original launch scripts,
failure/replacement records and continuation logs remain available for audit.
