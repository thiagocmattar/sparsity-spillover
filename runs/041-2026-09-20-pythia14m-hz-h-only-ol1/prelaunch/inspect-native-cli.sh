python3 - <<'PY'
import os,json
print(json.dumps({'pod_id':os.environ.get('RUNPOD_POD_ID'),'native_api_key_present':bool(os.environ.get('RUNPOD_API_KEY'))}))
PY
