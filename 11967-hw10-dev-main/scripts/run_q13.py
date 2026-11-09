"""Launch one Q1.3 variant, or inspect its plan without loading ML libraries."""
import argparse
import fcntl
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from q13_plan import CONFIGS, plan_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument('variant', choices=CONFIGS)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    code = Path(__file__).resolve().parents[1]
    plan = plan_run(code, a.variant)
    print(json.dumps(plan, indent=2), flush=True)
    if a.dry_run:
        return 0
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run inside an allocated gpu-interactive session.')
    source, output, results = (Path(plan[k]) for k in ('model', 'output', 'results'))
    if not (source/'model.safetensors').is_file():
        raise SystemExit('Input checkpoint is missing; no training started.')
    if output.exists() and any(output.iterdir()):
        raise SystemExit('Output directory is not empty; refusing to overwrite it.')
    if results.exists() and any(results.iterdir()):
        raise SystemExit('Prior result artifacts exist; archive them before a deliberate retry.')
    results.mkdir(parents=True, exist_ok=True)
    lock = (results/'run.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = os.environ.copy()
    env.update(WANDB_MODE='offline', WANDB_DIR=str(results), HF_HUB_OFFLINE='1',
               HF_DATASETS_OFFLINE='1', HF_HUB_DISABLE_XET='1', TOKENIZERS_PARALLELISM='false',
               OMP_NUM_THREADS='4', PYTHONUNBUFFERED='1')
    status, monitor = 1, None
    try:
        with (results/'install-source.log').open('w') as log:
            subprocess.run([sys.executable, str(code/'scripts/install_custom_model.py'), '--model-dir', str(source)], check=True, stdout=log, stderr=subprocess.STDOUT)
        with (results/'run-metadata.txt').open('w') as log:
            log.write(json.dumps(plan, indent=2)+'\n'); log.flush()
            commands = [['date','-u'], ['hostname'], ['git','rev-parse','HEAD'],
                        ['sha256sum', 'lm_train.py', plan['config_file'], str(source/'model.safetensors'), str(source/'modeling_custom.py')],
                        ['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,utilization.gpu','--format=csv'],
                        [sys.executable,'-c','import torch; assert torch.cuda.is_available() and torch.cuda.device_count()==1; print(torch.__version__, torch.cuda.get_device_name(0))'],
                        [sys.executable,'-m','pip','freeze']]
            for cmd in commands:
                subprocess.run(cmd, cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        shutil.copy2(plan['config_file'], results/'training-config.json')
        with (results/'gpu-usage.csv').open('w') as gpu, (results/'training.log').open('w') as log:
            monitor = subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,memory.used,utilization.gpu','--format=csv','-l','1'], stdout=gpu)
            status = subprocess.run([sys.executable, str(code/'lm_train.py'), plan['config_file']], cwd=code, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
        if status == 0:
            m = json.loads((output/'q11_metrics.json').read_text())
            state = json.loads((output/'trainer_state.json').read_text())
            if not (m['seq_len']==2048 and m['gradient_checkpointing'] and m['train']['epoch']>0.99 and state['global_step']==state['max_steps']):
                raise RuntimeError('Training ended without complete Q1.3 metrics.')
            shutil.copy2(output/'q11_metrics.json', results/'metrics.json')
            shutil.copy2(output/'trainer_state.json', results/'trainer_state.json')
    except Exception as exc:
        status = 1
        (results/'runner-error.txt').write_text(str(exc)+'\n')
    finally:
        if monitor is not None:
            monitor.terminate()
            try: monitor.wait(timeout=10)
            except subprocess.TimeoutExpired:
                monitor.kill(); monitor.wait()
        (results/'exit-code').write_text(str(status)+'\n')
        lock.close()
    return status


if __name__ == '__main__':
    raise SystemExit(main())
