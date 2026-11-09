"""Pure planning/search helpers. Importing this module performs no GPU work."""
import json
from pathlib import Path

CONFIGS = {v: f'2048_flash_gc_{v}_wikitext.json' for v in ('retrofit', 'scratch')}


def plan_run(code_root, variant):
    code_root = Path(code_root).resolve()
    config_path = code_root/'configs'/CONFIGS[variant]
    c = json.loads(config_path.read_text())
    expected = 'models/GPTNeoX-160M-WikiText-512-flash' if variant == 'retrofit' else 'models/GPTNeoX-160m'
    if c['model_to_train'] != expected:
        raise ValueError('Q1.3 must use the specified Q1.2 checkpoint or original initialization.')
    if not (c['seq_len'] == 2048 and c['attention_type'] == 'flash_attention_2' and c['gradient_checkpointing']):
        raise ValueError('Q1.3 requires length 2048, FlashAttention 2 and checkpointing.')
    source = (code_root/c['model_to_train']).resolve()
    output = (code_root/c['output_dir']).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('Input checkpoint and output must be separate.')
    return dict(variant=variant, config_file=str(config_path), config=c, model=str(source),
                output=str(output), results=str(code_root.parent/'q13-results/hku'/variant),
                tokens_per_update=c['seq_len']*c['per_device_train_batch_size']*c['gradient_accumulation_steps'])


def search_batch_limit(attempt, cap=64):
    """Exponential search then bisection; only explicit CUDA OOM is a boundary."""
    if cap < 1:
        raise ValueError('cap must be positive')
    low, candidate, failed = 0, 1, None
    while True:
        status = attempt(candidate)
        if status not in ('ok', 'oom'):
            raise RuntimeError('Probe failed for a reason other than CUDA OOM; search aborted.')
        if status == 'oom':
            failed = candidate
            break
        low = candidate
        if low == cap:
            return dict(largest_successful_batch_size=low, maximum_batch_size=None,
                        first_failing_batch_size=None, search_cap=cap, maximum_confirmed=False)
        candidate = min(2*candidate, cap)
    high = failed
    while high-low > 1:
        mid = (low+high)//2
        status = attempt(mid)
        if status == 'ok': low = mid
        elif status == 'oom': high = mid
        else: raise RuntimeError('Non-OOM probe failure; search aborted.')
    return dict(largest_successful_batch_size=low, maximum_batch_size=low,
                first_failing_batch_size=high, search_cap=cap, maximum_confirmed=True)
