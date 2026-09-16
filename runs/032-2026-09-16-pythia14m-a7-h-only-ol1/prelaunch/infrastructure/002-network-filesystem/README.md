# Environment and cache placement retry, before scientific execution

Three Pods mount /workspace through network-backed storage. Their initial
virtual environments were moved aside and installation restarted under
/opt/run032-venv; the two block-backed workers retain their environment under
/workspace with an /opt symlink. Logs and artifacts remain under /workspace.

Invoking the first Pod's local environment through its /workspace symlink still
made library lookups traverse the network filesystem. At 20:19 UTC its cache
process had no log output after about three minutes and was waiting in
request_wait_answer. A direct /opt/run032-venv/bin/python datasets import passed
immediately. The recorded readiness process trees were stopped and restarted
using the direct /opt path. No scientific attempt existed. Package versions,
training source, configuration, initialization, and data contract are unchanged.

All workers use /opt/run032-cache-parent for training-cache IO, exposed at the
unchanged repository cache path by a symlink. One pinned Hugging Face rebuild
on the kappa=0.5 worker supplies the immutable bytes to the other four through
encrypted runpodctl transfer. The small cloud-to-cloud transfer probe passed
with SHA-256 7f9a5854fe82a2b27746551b7a249eebefb9d8bdb316a777d67a8f181a214183.
Each scientific worker must independently verify the prescribed full cache
hashes, metadata, initialization, and realized order before launch.
