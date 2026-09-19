"""Read-only diagnosis of the measured startup delay."""
import remote

if __name__ == '__main__':
    client = remote.connect()
    try:
        print(remote.execute(client, "date -u\ndu -sh /workspace/run037/runtime/venv /workspace/run037-extensions\nps -eo pid,ppid,comm,etimes,time,pcpu | head -45\ncat /workspace/run037/artifacts/smoke/summary-001.json", timeout=60))
    finally:
        client.close()
