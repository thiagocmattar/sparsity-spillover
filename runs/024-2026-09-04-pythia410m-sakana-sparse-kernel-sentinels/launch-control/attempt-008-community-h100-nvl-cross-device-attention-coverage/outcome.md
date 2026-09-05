# Attempt 008 outcome

The focused no-cap/null-ratio correction passed synthetic smoke checks. The
recovery then reran only A7-OL1 `kappa=0.5` from the start with its original
condition index on the same Pod and physical H100 NVL. It completed in
823.56 seconds, serialized the sixth condition, rebuilt the cohort, and passed
the fail-closed verifier.

SHA-256 manifests before and after recovery prove that the five retained
condition JSONs were byte-identical. No within-condition timing segments were
spliced. A separate local copy passed an independently isolated verifier; its
regenerated verification payload is semantically identical to the remote
payload (the byte hash differs only because Windows translated output
newlines).

The final archive SHA-256 is
`c32755bfeaa6e55ba7f2ec1d65dc501af4c0f2769cf07979536a5ae110643a0c`.
It and all eight internal condition/cohort/verification hashes were verified
after retrieval. Pod `q10a8y77vkkavg` was deleted; the final account audit
found zero Pods, zero endpoints, and only the unchanged pre-existing network
volume.
