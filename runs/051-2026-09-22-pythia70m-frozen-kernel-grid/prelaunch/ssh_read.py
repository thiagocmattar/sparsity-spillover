"""Bounded SSH inspection for this retained pod."""
import argparse
from transport import connect,execute
p=argparse.ArgumentParser();p.add_argument('command');a=p.parse_args()
c=connect()
try:print(execute(c,a.command))
finally:c.close()
