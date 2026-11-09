#!/usr/bin/env bash
set -euo pipefail
source /root/yuchuanxue/A10/activate-hw10.sh
A10_RESULTS=/root/yuchuanxue/A10/q11-results
mkdir -p "$A10_RESULTS"
exec 9>"$A10_RESULTS/run.lock"
flock -n 9 || { echo 'An A10 Q1.1 run already holds the lock.'; exit 1; }
if [[ -f models/GPTNeoX-160M-WikiText-512-eager/model.safetensors || -f models/GPTNeoX-160M-WikiText-512-eager/q11_metrics.json ]]; then
    echo 'Saved Q1.1 output already exists; refusing to overwrite it.'
    exit 1
fi
python - <<'PY'
import subprocess, time
for check in range(3):
    raw = subprocess.check_output(['nvidia-smi','-i','3','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'], text=True)
    mem, util = [int(v.strip()) for v in raw.strip().split(',')]
    print(f'GPU3 preflight {check + 1}/3: {mem} MiB, {util}% utilization', flush=True)
    if mem > 512 or util > 5:
        raise SystemExit('GPU3 is occupied. Training was not started.')
    if check < 2:
        time.sleep(10)
PY
{
    date -u '+%Y-%m-%d %H:%M:%S UTC'
    hostname
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "$CUDA_VISIBLE_DEVICES"
    nvidia-smi -i 3 --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
    sha256sum lm_train.py configs/512_eager_wikitext.json models/GPTNeoX-160m/model.safetensors models/GPTNeoX-160m/modeling_custom.py
    python - <<'PY'
import torch, transformers, datasets, accelerate
from importlib.metadata import version
print('torch',torch.__version__,'CUDA',torch.version.cuda)
print('transformers',transformers.__version__,'datasets',datasets.__version__,'accelerate',accelerate.__version__,'flash-attn',version('flash-attn'))
assert torch.cuda.device_count() == 1
print('logical cuda:0',torch.cuda.get_device_name(0))
PY
} > "$A10_RESULTS/run-metadata.txt" 2>&1
nvidia-smi -i 3 --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -l 1 > "$A10_RESULTS/gpu-usage.csv" &
A10_MONITOR_PID=$!
trap 'kill "$A10_MONITOR_PID" 2>/dev/null || true' EXIT
set +e
python lm_train.py configs/512_eager_wikitext.json > "$A10_RESULTS/training.log" 2>&1
A10_STATUS=$?
set -e
printf '%s\n' "$A10_STATUS" > "$A10_RESULTS/exit-code"
if [[ "$A10_STATUS" == 0 ]]; then
    cp models/GPTNeoX-160M-WikiText-512-eager/q11_metrics.json "$A10_RESULTS/metrics.json"
    cp models/GPTNeoX-160M-WikiText-512-eager/trainer_state.json "$A10_RESULTS/trainer_state.json"
fi
exit "$A10_STATUS"
