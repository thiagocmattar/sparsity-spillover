set -eu
/workspace/run044-latency/runtime/venv/bin/python - <<'PY'
import hashlib,json
from pathlib import Path
root=Path('/workspace/run044-latency')
expected=(root/'provenance/pip-freeze.txt').read_text().splitlines()
actual=(root/'runtime/pip-freeze.txt').read_text().splitlines()
expected={line.strip() for line in expected if line.strip() and not line.startswith('#')}
actual={line.strip() for line in actual if line.strip() and not line.startswith('#')}
assert expected==actual,{'missing':sorted(expected-actual),'unexpected':sorted(actual-expected)}
result={'pinned_packages_matched':len(expected),'source':'provenance/pip-freeze.txt',
    'installed':'runtime/pip-freeze.txt','pins_sha256':hashlib.sha256((root/'provenance/pip-freeze.txt').read_bytes()).hexdigest()}
destination=root/'runtime/pin-verification.json'
assert not destination.exists()
destination.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
PY
