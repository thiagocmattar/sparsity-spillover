"""Resume the unchanged sealed upload with the training transport's window sizes."""
import importlib.util
from pathlib import Path
import socket
import sys
import paramiko

RUN = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(RUN))
spec = importlib.util.spec_from_file_location('run068_head_transport', RUN / '09_remote.py')
transport = importlib.util.module_from_spec(spec)
spec.loader.exec_module(transport)
client, lease = transport.remote.connect('benchmark')
try:
    channel = client.get_transport()
    channel.sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 8 * 1024**2)
    client.open_sftp = lambda: paramiko.SFTPClient.from_transport(
        channel, window_size=64 * 1024**2, max_packet_size=1024**2)
    transport.prepare(client, lease, '35a4c2b7d470191f52e439df961ab77c9b5daec2')
finally:
    client.close()
