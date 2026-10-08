set -euo pipefail
test -f /workspace/run066-source-002.bundle
test ! -e /workspace/run066-source-002.interrupted.bundle
mv -- /workspace/run066-source-002.bundle /workspace/run066-source-002.interrupted.bundle
stat -c '%n %s bytes' /workspace/run066-source-002.interrupted.bundle
