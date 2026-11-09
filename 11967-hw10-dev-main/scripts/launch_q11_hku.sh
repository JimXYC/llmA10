#!/usr/bin/env bash
# Run inside the school's existing gpu-interactive allocation.
set -euo pipefail
A10_CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
A10_REPO_ROOT="$(cd "$A10_CODE_ROOT/.." && pwd)"
A10_RESULTS="$A10_REPO_ROOT/q11-results/hku"
A10_PYTHON="${A10_PYTHON:-$HOME/hw10-env/bin/python}"
cd "$A10_CODE_ROOT"
mkdir -p "$A10_RESULTS"
exec 9>"$A10_RESULTS/run.lock"
flock -n 9 || { echo 'Q1.1 is already running.'; exit 1; }
# Keep SLURM's CUDA_VISIBLE_DEVICES; never select a physical GPU number here.
"$A10_PYTHON" - <<'PY'
import torch
assert torch.cuda.is_available(), 'Run inside gpu-interactive, with the HW10 environment installed.'
assert torch.cuda.device_count() == 1, 'Expected the single GPU allocated for this experiment.'
PY
if [[ -f models/GPTNeoX-160M-WikiText-512-eager/model.safetensors ]]; then
    echo 'Saved Q1.1 model exists; refusing to overwrite it.'
    exit 1
fi
export WANDB_MODE=offline
export WANDB_DIR="$A10_RESULTS"
export HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
export PYTHONUNBUFFERED=1
{
    date -u '+%Y-%m-%d %H:%M:%S UTC'
    hostname
    git rev-parse HEAD
    printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-unset}"
    nvidia-smi --query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu --format=csv
    sha256sum lm_train.py configs/512_eager_wikitext.json models/GPTNeoX-160m/model.safetensors models/GPTNeoX-160m/modeling_custom.py
    "$A10_PYTHON" -m pip freeze
} > "$A10_RESULTS/run-metadata.txt" 2>&1
nvidia-smi --query-gpu=timestamp,index,memory.used,utilization.gpu --format=csv -l 1 > "$A10_RESULTS/gpu-usage.csv" &
A10_MONITOR=$!
trap 'kill "$A10_MONITOR" 2>/dev/null || true' EXIT
set +e
"$A10_PYTHON" lm_train.py configs/512_eager_wikitext.json > "$A10_RESULTS/training.log" 2>&1
A10_STATUS=$?
set -e
printf '%s\n' "$A10_STATUS" > "$A10_RESULTS/exit-code"
if [[ "$A10_STATUS" == 0 ]]; then
    cp models/GPTNeoX-160M-WikiText-512-eager/q11_metrics.json "$A10_RESULTS/metrics.json"
    cp models/GPTNeoX-160M-WikiText-512-eager/trainer_state.json "$A10_RESULTS/trainer_state.json"
fi
exit "$A10_STATUS"
