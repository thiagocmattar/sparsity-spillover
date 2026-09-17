# Draft only — not sent

Destination: Runpod customer support,
https://contact.runpod.io/hc/en-us/requests/new

Subject: Recover retained /workspace data from Pod 8o1uyxgh6vb6ym

Please help restore access to the retained 25 GB /workspace volume of Pod
8o1uyxgh6vb6ym (run032-k0p5, US-MO-1, A100 SXM4 80 GB). The container stopped
on 17 September 2026 at 03:13:34 UTC. Its retained files are needed for recovery;
completion of the original workload has not yet been verified.

GPU resume reports: "There are not enough free GPUs on the host machine to
start this pod." CPU-only resume with gpuCount=0 and computeType=CPU reports:
"There are not enough free memory on the host machine to start this pod."
The CPU-only request also failed with syncMachine=true. The account owner has
added funds. Recovery retries continue every five minutes.

Could you enable minimal CPU-only access for file retrieval, or provide a
supported way to migrate/copy the existing volume to accessible storage?
Please preserve the source Pod and its retained volume until we confirm that
all original files have been copied and their SHA-256 hashes verified.
Do not reset, terminate, or delete the source as part of troubleshooting.

No credentials, research artifacts, or dataset contents are attached.
