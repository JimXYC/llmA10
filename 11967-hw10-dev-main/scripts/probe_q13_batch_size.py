"""Q1.3.A: measure training batch limits with isolated actual-Trainer workers.

Running this script on a GPU performs short optimization steps, not inference.
Use --dry-run to inspect the plan without ML imports, CUDA or output writes.
"""
import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from q13_plan import search_batch_limit

MODES = ('eager_no_gc', 'flash_gc')


def worker(a):
    import torch
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, set_seed
    record = dict(mode=a.mode, batch_size=a.batch_size, seq_len=2048, optimizer_steps=a.steps,
                  gradient_accumulation_steps=1, use_cache=False)
    status = 1
    try:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expected one GPU from an existing allocation.')
        set_seed(42)
        enabled = a.mode == 'flash_gc'
        attention = 'flash_attention_2' if enabled else 'eager'
        model = AutoModelForCausalLM.from_pretrained(a.model_dir, torch_dtype=torch.bfloat16,
                                                   attn_implementation=attention, trust_remote_code=True)
        model.config.use_cache = False
        tokenizer = AutoTokenizer.from_pretrained(a.model_dir)
        tokenizer.pad_token = tokenizer.eos_token
        dataset = load_dataset('cjiao/wikitext-tokenized', split='train_2048')
        count = a.batch_size * a.steps
        if count > len(dataset):
            raise RuntimeError('Requested probe exceeds the available full-batch training samples.')
        dataset = dataset.select(range(count))
        with tempfile.TemporaryDirectory(prefix='q13-trainer-') as temp:
            args = TrainingArguments(output_dir=temp, max_steps=a.steps, bf16=True,
                per_device_train_batch_size=a.batch_size, gradient_accumulation_steps=1,
                gradient_checkpointing=enabled, gradient_checkpointing_kwargs={'use_reentrant':False} if enabled else None,
                learning_rate=1e-4, optim='adamw_torch', adam_beta1=0.9, adam_beta2=0.95,
                weight_decay=0.1, seed=42, data_seed=42, report_to='none',
                save_strategy='no', eval_strategy='no', logging_strategy='no', logging_nan_inf_filter=False,
                dataloader_num_workers=0, disable_tqdm=True)
            trainer = Trainer(model=model, args=args, train_dataset=dataset,
                              data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False), tokenizer=tokenizer)
            torch.cuda.reset_peak_memory_stats()
            result = trainer.train()
            if trainer.state.global_step != a.steps or not math.isfinite(result.metrics['train_loss']):
                raise RuntimeError('Incomplete or non-finite probe; no memory limit inferred.')
            torch.cuda.synchronize()
            record.update(status='ok', gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
                          train_runtime=result.metrics['train_runtime'],
                          peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
                          peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2)
            status = 0
    except torch.cuda.OutOfMemoryError as exc:
        record.update(status='oom', error=str(exc)); status=0
    except Exception as exc:
        record.update(status='error', error=str(exc))
    a.record.write_text(json.dumps(record,indent=2)+'\n')
    return status


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dry-run',action='store_true')
    p.add_argument('--cap',type=int,default=64)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--timeout',type=int,default=300,help='Seconds per isolated candidate.')
    p.add_argument('--model-dir',type=Path,default=Path('models/GPTNeoX-160M-WikiText-512-flash'))
    p.add_argument('--results',type=Path,default=Path('../q13-results/batch-probe'))
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    p.add_argument('--mode',choices=MODES,help=argparse.SUPPRESS)
    p.add_argument('--batch-size',type=int,help=argparse.SUPPRESS)
    p.add_argument('--record',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.steps<2 or a.cap<1 or a.timeout<1:
        p.error('Require steps>=2, cap>=1, timeout>=1.')
    if a.worker:
        if a.mode is None or a.batch_size is None or a.record is None:
            p.error('Worker requires mode, batch size and record path.')
        return worker(a)
    code=Path(__file__).resolve().parents[1]
    model=(code/a.model_dir).resolve()
    results=(code/a.results).resolve()
    print(json.dumps(dict(model=str(model),results=str(results),modes=MODES,seq_len=2048,
                          optimizer_steps=a.steps,search_cap=a.cap,dry_run=a.dry_run),indent=2),flush=True)
    if a.dry_run:
        return 0
    if not os.environ.get('SLURM_JOB_ID'):
        raise SystemExit('Run inside gpu-interactive; no GPU was requested by this script.')
    if not (model/'model.safetensors').is_file():
        raise SystemExit('Missing Q1.2 model checkpoint.')
    if results.exists() and any(results.iterdir()):
        raise SystemExit('Probe results exist; archive them before rerunning.')
    results.mkdir(parents=True,exist_ok=True)
    lock=(results/'run.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX | fcntl.LOCK_NB)
    env=os.environ.copy()
    env.update(HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1',HF_HUB_DISABLE_XET='1',WANDB_MODE='disabled',
               OMP_NUM_THREADS='4',TOKENIZERS_PARALLELISM='false',PYTHONUNBUFFERED='1')
    summaries={}
    for mode in MODES:
        attempts=[]
        def attempt(batch):
            record=results/f'{mode}-batch{batch}.json'
            cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--mode',mode,
                 '--batch-size',str(batch),'--steps',str(a.steps),'--model-dir',str(model),'--record',str(record)]
            with (results/f'{mode}-batch{batch}.log').open('w') as log:
                try:
                    process=subprocess.run(cmd,cwd=code,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=a.timeout)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(f'Probe timeout at {mode} batch {batch}; not classified as OOM.') from exc
            if not record.exists():
                raise RuntimeError(f'Worker produced no record at {mode} batch {batch}; not classified as OOM.')
            data=json.loads(record.read_text());attempts.append(data)
            if process.returncode!=0 or data['status'] not in ('ok','oom'):
                raise RuntimeError(f'Non-OOM worker failure: {data}')
            print(f"{mode} batch={batch}: {data['status']}",flush=True)
            return data['status']
        result=search_batch_limit(attempt,a.cap)
        summaries[mode]={**result,'attempts':attempts}
        (results/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))
    lock.close()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
