"""Planning estimate only, derived from the measured Q1.2 run; no GPU work."""
import argparse
import json
from pathlib import Path


def estimate(repo):
    baseline=json.loads((repo/'q12-results/hku/metrics.json').read_text())
    hidden,layers,vocab=768,12,50304  # supplied Pythia/GPTNeoX 160M architecture
    target_examples,target_len=28531,2048
    tokens_ratio=target_examples*target_len/baseline['train_tokens']
    # Leading dense FLOPs/token: projections+MLP=24H^2; QK/AV=4LH.
    # Backward ~=2 forward; checkpointing adds one block forward.
    old_block=layers*(24*hidden**2+4*baseline['seq_len']*hidden)
    new_block=layers*(24*hidden**2+4*target_len*hidden)
    head=2*hidden*vocab
    relative_compute=tokens_ratio*(4*new_block+3*head)/(3*old_block+3*head)
    minutes=baseline['train']['train_runtime']/60
    return dict(status='planning_estimate_not_a_q13_measurement',hardware=baseline['gpu'],
                baseline_q12_minutes=minutes,baseline_tokens=baseline['train_tokens'],
                q13_training_examples=target_examples,q13_tokens=target_examples*target_len,
                assumed_architecture=dict(hidden_size=hidden,layers=layers,vocab_size=vocab),
                approximate_compute_multiplier=relative_compute,
                central_compute_only_minutes_per_run=minutes*relative_compute,
                planned_minutes_per_run=[25,40],two_training_runs_minutes=[50,80],
                batch_probe_extra_minutes=[5,15],
                conditions='One epoch per variant, existing school environment and RTX 4080 SUPER; batch 4 x accumulation 2. Excludes queueing/download/setup.',
                uncertainty='Not benchmarked at 2048 with checkpointing. Kernel efficiency, allocator behavior and a smaller feasible batch can change runtime. Refine only after an explicitly authorized GPU probe/run.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path)
    a=p.parse_args()
    record=estimate(Path(__file__).resolve().parents[2])
    print(json.dumps(record,indent=2))
    if a.output:
        a.output.parent.mkdir(parents=True,exist_ok=True)
        a.output.write_text(json.dumps(record,indent=2)+'\n')
