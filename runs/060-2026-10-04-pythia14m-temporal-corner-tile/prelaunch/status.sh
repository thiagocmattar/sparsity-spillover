set -eu
control=/workspace/run060-control
run=/workspace/sparsity-spillover/runs/060-2026-10-04-pythia14m-temporal-corner-tile
date -u +%FT%TZ
nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu --format=csv,noheader
df -h /workspace
for marker in environment-ready cache-ready preflight-passed completed pipeline.exit seal.exit; do
  if [ -f "$control/$marker" ]; then printf 'MARKER %s ' "$marker"; cat "$control/$marker"; printf '\n'; fi
done
tail -n 5 "$control/pipeline.log" 2>/dev/null || true
tail -n 2 "$control/seal.log" 2>/dev/null || true
tail -n 5 "$control/cache-build.log" 2>/dev/null || true
for log in "$control"/preflight-*.log "$control"/training-*.log; do
  [ -f "$log" ] || continue
  printf 'LOG %s ' "${log##*/}"
  stat --printf='bytes=%s modified=%y\n' "$log"
  if grep -qE 'Traceback|RuntimeError|ValueError|CUDA out of memory' "$log"; then tail -n 8 "$log"; fi
done
if [ -f "$control/environment-ready" ]; then /workspace/run060-venv/bin/python "$run/04_monitor.py"; fi
python3 - "$run" <<'PY'
import json, sys
from pathlib import Path
root=Path(sys.argv[1])/'artifacts/attempts'
for manifest_path in sorted(root.glob('*/manifest.json')):
    manifest=json.loads(manifest_path.read_text())
    events=manifest_path.parent/'events.jsonl'
    rows=[]; skipped=0; overflows=0
    if events.exists():
        for line in events.read_text().splitlines():
            try: row=json.loads(line)
            except json.JSONDecodeError: continue
            if row.get('event')=='validation': rows.append(row)
            if row.get('event')=='train':
                skipped+=int(row.get('optimizer_step_skipped',False))
                overflows+=int(row.get('gradient_overflow',False))
    print(json.dumps({'condition':manifest['condition']['id'],'latest_full_validation':
        {key:rows[-1][key] for key in ('step','loss')} if rows else None,
        'skipped_updates':skipped,'overflow_updates':overflows}))
    files=sorted((manifest_path.parent/'diagnostics').glob('step_*/activation_statistics.json'))
    final=manifest_path.parent/'diagnostics/activation_statistics.json'
    if final.exists(): files.append(final)
    if files:
        path=files[-1]; data=json.loads(path.read_text())
        print(json.dumps({'condition':manifest['condition']['id'],'diagnostic':path.parent.name,
            'empty_8x16_percent':{r['name']:100*r['tiles']['8x16']['empty']/r['tiles']['8x16']['total']
                for r in data['occupancy_pooled_by_site']}}))
PY
cat "$control/guard.log" 2>/dev/null || true
