"""Start setup with all background process descriptors detached from SSH."""
import io
import remote

c=remote.connect()
try:
    s=c.open_sftp()
    worker=b'''#!/usr/bin/env bash
cd /workspace/run037
bash 04_setup.sh > runtime/setup-001.log 2>&1
code=$?
printf '%s\\n' "$code" > runtime/setup-001.exit
'''
    s.putfo(io.BytesIO(worker),'/workspace/run037/runtime/setup-worker-001.sh')
    command="""cd /workspace/run037
test ! -e runtime/setup-001.pid || exit 2
setsid bash runtime/setup-worker-001.sh < /dev/null > runtime/setup-launch-001.log 2>&1 &
printf '%s\\n' "$!" > runtime/setup-001.pid
cat runtime/setup-001.pid
"""
    print(remote.execute(c,command))
    s.close()
finally: c.close()
