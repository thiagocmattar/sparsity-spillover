"""Run a literal scoped maintenance command, supplied as an argument."""
import sys
from remote import connect, execute

client = connect()
try:
    print(execute(client, sys.argv[1], timeout=60), flush=True)
finally:
    client.close()
