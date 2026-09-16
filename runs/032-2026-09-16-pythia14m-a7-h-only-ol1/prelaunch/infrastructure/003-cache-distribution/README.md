# Cache distribution retry before three scientific workers start

The first directory transfer completed and the kappa=0 receiver independently
verified its full cache hashes and order. Three concurrent sends failed because
runpodctl 2.12.0 creates the same temporary directory ZIP basename for all sends.
The sender reported `minipile-pythia-14m-full.zip file already exists!`; receivers
reported `room not ready`. Their original logs remain separate and untouched.

The retry transfers the train token binary directly, avoiding temporary ZIP
creation; the two metadata files and small validation binary travel over SSH.
Each receiver still independently checks every prescribed full hash and the
realized training order, then runs exact A100 preflight before scientific launch.
No scientific source, configuration, cache content, or completed attempt changes.
