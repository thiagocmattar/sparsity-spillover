"""Keep persistent master; map byte-identical verified training tokens to local disk."""
import hashlib,json,pathlib,time
source=pathlib.Path('/workspace/sparsity-spillover/data/tokenized/minipile-pythia-14m-full/train/tokens.int32.bin')
fast=pathlib.Path('/opt/run055-train-tokens.int32.bin')
started=time.time()
assert fast.stat().st_size==5966845664
with fast.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
assert digest=='da82a2ea2e0080c7fd681c7a93b07d3d9ff3d5357a8640895a82d536a1eaf97c'
assert not source.is_symlink() and source.stat().st_size==fast.stat().st_size
master=source.with_name('tokens.int32.persistent.bin')
assert not master.exists()
source.rename(master)
source.symlink_to(fast)
receipt=dict(status='verified',bytes=fast.stat().st_size,sha256=digest,persistent_master=str(master),
    runtime_path=str(source),resolved_runtime_path=str(fast),hash_seconds=time.time()-started,
    note='Cache bytes and metadata unchanged; shared-filesystem read bottleneck avoided. Initial input checker already held the persistent original file open before this placement switch.')
pathlib.Path('/workspace/run055-control/fast-cache.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt),flush=True)
