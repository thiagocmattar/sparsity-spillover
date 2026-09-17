# Four conditions fully recovered

Question: can the scientific artifacts from the four accessible completed
workers be fully recovered despite stalled archive downloads?

Method: preserve partial downloads, verify complete members against the
original inventory, stage only missing files onto the Pod root disk, stream
stalled transfers over SSH, verify archive hashes and every required file,
then run the unchanged standalone condition verifier.

Coverage: kappa=0, 0.01, 0.05 and 0.1; all 2,848 optimizer boundaries, all
required model/recovery checkpoints, final full-validation outputs and
diagnostics. Each condition has 712 updates, 1,493,172,224 input tokens and
complete 500-document/338-block validation with 1,444 excluded tail tokens.

Result: all four conditions pass full local verification. Their final losses
and logical opportunities are unchanged from observation 001. Their Pods were
deleted after verified recovery. Kappa=0.5 remains on its stopped original
volume because Runpod refuses both GPU and CPU-only resumes.

Caveats: the five-condition cohort is incomplete, and no claim about
high-threshold sufficiency or Q/K/V necessity is possible. The four-condition
matched interim comparison belongs to Analysis 022, observation 001.

Sources/scripts: `19_retrieve_missing.py`, `21_stream_staged_archive.py`,
`03_verify.py`, `prelaunch/cloud/*/retrieval-receipt.json`, and
`prelaunch/recovered-metadata-audit.json`. Infrastructure record 005 records
the transfer retries, expanded user authorization, and remaining blocker.
