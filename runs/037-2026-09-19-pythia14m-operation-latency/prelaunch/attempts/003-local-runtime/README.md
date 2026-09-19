# Infrastructure adjustment 003: local runtime/cache

The Pod's `/workspace` is a FUSE network filesystem. Warm smoke processes
took roughly 53–57 seconds, mostly imports and graph setup; the approved
scientific matrix and retrieval therefore risked approaching the deadline.
An unchanged local copy of the venv and compiled extensions was prepared
under `/tmp/run037-local/`, with per-file SHA256 verification. All source,
checkpoint, data, numerical bounds, kernel flags and benchmark parameters
remained unchanged. Logs and scientific outputs stayed on persistent storage.

The smoke controller was held between launches. Its active without-z child
finished normally, including all timing, correctness and diagnostics, before
copying began. The child result records about 50 seconds of actual execution;
the controller summary includes the subsequent administrative pause and must
not be used as its computational duration. No scientific process had started.

The first copy did not preserve modification times, causing unnecessary
recompilation in the next smoke process. `restore_cache_metadata_004.py`
restored timestamps only where bytes still matched the verified copy.
21,049 files were restored; 2,126 regenerated Python bytecode files and eight
in-progress build/cache files were retained rather than overwritten. No
Python source or library package version changed. Both the initial relocation
and its correction are retained under `artifacts/infrastructure/003-local-runtime/`.

Using the old logical venv path still incurred network path traversal.
A direct local Python import of torch and transformers took 2.59 seconds.
Adjustment 004 uses that direct executable for every scientific process,
while retaining the original include/library paths for extension compilation.
