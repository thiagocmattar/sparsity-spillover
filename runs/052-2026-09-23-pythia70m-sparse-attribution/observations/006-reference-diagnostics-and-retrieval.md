# Reference diagnostics and verified retrieval

Question: retain the full agreed diagnostic inventory even though the bounded
kernel search produced no promoted candidate.

Method: run the unchanged `08_diagnostics.py` on the fixed pre-search policy,
using all338 validation blocks for Base dense, and both dense/prior-C paths at
each moderate-kappa checkpoint. This covers all500 documents,692,224 input
tokens, and the declared1,444-token excluded tail. `19_reference_diagnostics.py`
only supervises these three processes and their existing absolute deadline.
The explicitly named `diagnostic-reference-policy.json` is not a final candidate
selection. Numerical qualification comes from the15 matched reference processes.

All five mode/condition collections completed. Exact/near-zero counts, RMS/L2,
weight statistics, per-row/group/tile occupancy, logical h/z work, independent
conversion-buffer checks, profiles and compiler evidence are retained. Pooling
integer counts across layers before division gives these prior-C statistics:

| Kappa | h exact zeros | Mean h nonzeros /2048 | z exact zeros | Mean z nonzeros /512 |
|---|---:|---:|---:|---:|
|.05|98.322769%|34.349700|93.320496%|34.199061|
|.1|98.918745%|22.144097|96.845088%|16.153151|

These are activation properties; the retained policy uses sparse h in five
layers and dense z in every layer. Requested-weight/logical counters are not
hardware DRAM/L2 traffic. No gradient interaction is reconstructed from inference.
Original training records, checkpoints and cache identities remain retained.

Retrieval: the completed-search archive has483 files,182,034,434 uncompressed
bytes and26,673,007 compressed bytes. Archive SHA256 is
`7e57f74fd548665a1abf253de7b70eba83e3b79fe99316fdd6f3f90ff8a4db80`.
Every member's path, size and SHA256 passed local verification. This includes
failed attempts, both complete screens, raw timings, reference diagnostics,
profiles, new CUDA binaries and43 build-cache binary/build-log files. Source
and locally generated summaries/observations are additionally retained in Git.

Compute: all workers completed. Provider listing confirmed only the requested
Pod `qa00sm05lmrs7w`, still RUNNING. It remains available for user guidance at
the approved rate, with the independent stop guard at15:49:34 UTC /12:49:34
Brasilia on23 September2026. No extension or termination was performed.

Sources: `08_diagnostics.py`, `19_reference_diagnostics.py`, retained Run049
diagnostic/profiling helpers, `14_verify_archive.py`,
`prelaunch/retrieve_stage.py`, `results/reference-diagnostic-summary.json`,
`prelaunch/retrieval-completed-search-001.json`, `results/compute-retained.json`.
No figure or manuscript change was generated.
