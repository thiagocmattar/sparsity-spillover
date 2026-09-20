date -u
nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv
ps -eo pid,ppid,etime,pcpu,pmem,args --sort=-pcpu | head -16
tail -20 /workspace/run043-latency/artifacts/smoke/smoke-c00-r1-001.log
find /workspace/run043-latency/runtime/extensions -maxdepth 3 -type f -printf '%TY-%Tm-%TdT%TH:%TM:%TS %s %p\n' | tail -15
