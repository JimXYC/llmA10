"""Fill Q1.2 A/B/C only after a completed full FlashAttention training run."""
import argparse
import csv
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
a = p.parse_args()
root = a.root
m = json.loads((root/'q12-results/hku/metrics.json').read_text())
e = json.loads((root/'q11-results/hku/metrics.json').read_text())
c = json.loads((root/'11967-hw10-dev-main/configs/512_flash_wikitext.json').read_text())
b = json.loads((root/'11967-hw10-dev-main/configs/512_eager_wikitext.json').read_text())
state = json.loads((root/'q12-results/hku/trainer_state.json').read_text())
v = json.loads((root/'q12-results/validation.json').read_text())
assert (root/'q12-results/hku/exit-code').read_text().strip() == '0'
assert m['attention_type'] == 'flash_attention_2' and m['seq_len'] == 512
assert m['train']['epoch'] > 0.99 and state['global_step'] == state['max_steps']
assert {k for k in b.keys() | c.keys() if b.get(k) != c.get(k)} == {'attention_type', 'run_name', 'output_dir'}
assert m['gpu'] == e['gpu'] and m['torch_version'] == e['torch_version']
assert v['tests'] == 36 and v['failures'] == v['errors'] == v['skipped'] == 0
p512, p2048 = (float(m['validation_'+s]['perplexity']) for s in ('512', '2048'))
e512, e2048 = (float(e['validation_'+s]['perplexity']) for s in ('512', '2048'))
time, old_time = float(m['train']['train_runtime']), float(e['train']['train_runtime'])
alloc, old_alloc = float(m['peak_allocated_mib']), float(e['peak_allocated_mib'])
assert all(math.isfinite(x) and x > 0 for x in (p512, p2048, time, alloc))
with (root/'q12-results/hku/gpu-usage.csv').open() as f:
    rows = csv.reader(f)
    next(rows)
    sampled_peak = max(float(r[2].split()[0]) for r in rows if len(r) >= 4)
part_a = r'''\subsection*{1.2.A}
I completed \texttt{GPTNeoXFlashAttention2.forward()} in the supplied custom model. Q/K/V are transposed from [batch, heads, sequence, head dimension] to [batch, sequence, heads, head dimension]. The helper \nolinkurl{_flash_attention_forward} receives the padding mask, causal-alignment flag and train/eval dropout. Its output heads are merged before the final projection.

I wrote 36 CUDA tests, all of which passed with no skips on an RTX 4080 SUPER using PyTorch 2.4.1 and FlashAttention 2.6.3. Small two-layer models with identical weights compare eager and flash logits, losses and every parameter gradient in FP16/BF16, with an additional FP32 eager reference. Coverage includes batch sizes 1/2, lengths 1/17/512/2048, left/right padding, strict causal non-leakage, and single/multiple-token cached decoding. Separate checks cover train/eval dropout and a real 160M-checkpoint forward/backward pass. Consistency comparisons disable dropout and use dtype-dependent tolerances; the causal-prefix check requires exact equality. Gradient comparisons use relative L2 and normalized maximum errors to handle cancellation near zero.

\subsection*{1.2.B}
'''
part_b = ('I trained from the same supplied initialization for one epoch, keeping all Q1.1 hyperparameters and the same GPU model. '
          'Only the attention backend was changed to FlashAttention 2; gradient checkpointing remained disabled.\n\n')
part_b += rf'''\begin{{center}}
\begin{{tabular}}{{lrr}}
\toprule
Metric & Q1.1: eager & Q1.2: flash \\
\midrule
Validation perplexity, length 512 & {e512:.2f} & {p512:.2f} \\
Validation perplexity, length 2048 & {e2048:.2f} & {p2048:.2f} \\
Training time (seconds) & {old_time:.2f} & {time:.2f} \\
Peak training allocated VRAM (MiB) & {old_alloc:.2f} & {alloc:.2f} \\
\bottomrule
\end{{tabular}}
\end{{center}}

'''
part_b += (f'The flash run took {time/60:.2f} minutes. Timing and memory definitions match Q1.1: Trainer runtime includes epoch-end 512-token validation and excludes model saving and the final validations; the allocated-memory peak is captured before those final validations. '
           f'Flash peak reserved CUDA memory was {m["peak_reserved_mib"]:.0f} MiB. '
           f'The device-memory peak sampled once per second over the full flash run was {sampled_peak:,.0f} MiB.\n\n')
part_c = '\\subsection*{1.2.C}\n'
relative_ppl = max(abs(p512/e512-1), abs(p2048/e2048-1))
if relative_ppl < 0.1:
    part_c += ('The perplexities are similar across the two attention backends. FlashAttention computes the same mathematical attention operation, while floating-point reduction order and low-precision rounding can change individual updates and the resulting training trajectory. The much higher 2048-token perplexity persists because both models were trained only at length 512.\n\n')
else:
    part_c += (f'The validation perplexities differ by {p512-e512:+.2f} (length 512) and {p2048-e2048:+.2f} (length 2048). These differences are larger than expected from an interchangeable attention backend alone and warrant further investigation, despite the forward/backward unit tests passing.\n\n')
if time < old_time:
    part_c += f'Flash training was {old_time/time:.2f} times as fast, reducing measured runtime by {100*(1-time/old_time):.1f}\\%. '
else:
    part_c += f'Flash training took {100*(time/old_time-1):.1f}\\% longer in this measurement. '
if alloc < old_alloc:
    part_c += f'Peak allocated training memory fell by {100*(1-alloc/old_alloc):.1f}\\%.\n\n'
else:
    part_c += f'Peak allocated training memory changed by {100*(alloc/old_alloc-1):+.1f}\\%.\n\n'
with (root/'q11-results/hku/gpu-usage.csv').open() as f:
    rows = csv.reader(f)
    next(rows)
    old_sampled_peak = max(float(r[2].split()[0]) for r in rows if len(r) >= 4)
if sampled_peak > old_sampled_peak:
    part_c += (f'The full-run device high-water mark was higher ({sampled_peak:,.0f} versus {old_sampled_peak:,.0f} MiB), with the flash peak occurring at the end of the run. '
               'This measure includes \\href{https://docs.pytorch.org/docs/2.4/notes/cuda.html#memory-management}{unused allocator blocks} and covers final evaluation, while the training tensor-allocation peak is captured beforehand.\n\n')
part_c += r'''FlashAttention tiles Q/K/V computation and performs online softmax using on-chip memory, avoiding the storage of a complete sequence-by-sequence attention matrix in GPU global memory. During backpropagation it recomputes attention intermediates from these tiles. This reduces attention-intermediate memory and global-memory traffic. Attention arithmetic remains quadratic in sequence length. FlashAttention-2 also improves GPU occupancy and work partitioning while reducing non-matmul work and inter-warp communication. These mechanisms explain potential speed and memory gains; other model operations and runtime overhead limit the end-to-end improvement. See \href{https://arxiv.org/abs/2205.14135}{Dao et al. (2022)} and \href{https://arxiv.org/abs/2307.08691}{Dao (2023)}.

'''
tex_path = root/'Assignment.tex'
tex = tex_path.read_text()
start = tex.index('\\subsection*{1.2.A}')
end = tex.index('\\section*{Question 1.3 (5 points)}')
tex_path.write_text(tex[:start]+part_a+part_b+part_c+'\\clearpage\n'+tex[end:])
summary = dict(ppl_512=p512, ppl_2048=p2048, training_seconds=time,
               speedup=old_time/time, runtime_reduction_percent=100*(1-time/old_time),
               allocated_memory_reduction_percent=100*(1-alloc/old_alloc),
               peak_allocated_mib=alloc, peak_reserved_mib=m['peak_reserved_mib'],
               peak_full_run_device_sampled_mib=sampled_peak,
               timing_definition='Same as Q1.1; Trainer runtime includes epoch-end 512-token evaluation.',
               memory_definition='Training PyTorch peaks before final evaluations; sampled device peak over full run.')
(root/'q12-results/hku/measurement-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
