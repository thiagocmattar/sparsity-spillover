set -eu
date -u
tail -n 8 /workspace/run057-control/pipeline.log
for log in /workspace/run057-control/training-*.log; do
    test ! -f "$log" || { echo "$log"; tail -n 3 "$log"; }
done
pgrep -af '02_train.py|01_validate_preflights.py|08_pipeline.sh' || true
