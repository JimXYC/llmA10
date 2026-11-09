"""Fill only HW10 Q1.1 using completed experiment metrics; no reference values."""
import argparse
import csv
import json
import math
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--metrics', type=Path, required=True)
p.add_argument('--gpu-csv', type=Path)
p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
a = p.parse_args()
m = json.loads(a.metrics.read_text())
c = json.loads((a.root/'11967-hw10-dev-main/configs/512_eager_wikitext.json').read_text())
p512 = float(m['validation_512']['perplexity'])
p2048 = float(m['validation_2048']['perplexity'])
seconds = float(m['train']['train_runtime'])
allocated = float(m['peak_allocated_mib'])
reserved = float(m['peak_reserved_mib'])
assert all(math.isfinite(v) and v > 0 for v in (p512, p2048, seconds, allocated, reserved))
assert m['attention_type'] == 'eager' and m['seq_len'] == 512
assert m['train']['epoch'] > 0.99, 'Refusing to report an incomplete or smoke-test run'
sampled_peak = None
if a.gpu_csv:
    with a.gpu_csv.open() as f:
        rows = csv.reader(f)
        next(rows)
        sampled_peak = max(float(row[2].strip().split()[0]) for row in rows if len(row) >= 4)
bs = c['per_device_train_batch_size']
ga = c.get('gradient_accumulation_steps', 1)
setup = (
    f'I trained the supplied randomly initialized 160M-parameter GPTNeoX model for one epoch '
    f'on the pre-tokenized WikiText training split ({m["train_examples"]:,} sequences of length 512; '
    f'{m["train_tokens"]:,} tokens), using eager attention and BF16 on one {m["gpu"]}. '
    f'The per-device batch size was {bs}, with {ga} gradient accumulation steps '
    f'(effective batch size {bs*ga}). I used AdamW with peak learning rate '
    r'$6\times10^{-4}$, $\beta_1=0.9$, $\beta_2=0.95$, weight decay 0.1, '
    r'1\% warmup, cosine decay, gradient clipping at 1.0, and seed 42. '
    'Gradient checkpointing was disabled.\n\n'
)
table = rf'''\begin{{center}}
\begin{{tabular}}{{lr}}
\toprule
Metric & Measured value \\
\midrule
Validation perplexity, length 512 & {p512:.2f} \\
Validation perplexity, length 2048 & {p2048:.2f} \\
Training time & {seconds:.2f} s ({seconds/60:.2f} min) \\
Peak training VRAM, PyTorch allocated & {allocated:.2f} MiB \\
\bottomrule
\end{{tabular}}
\end{{center}}

'''
measurement = (
    'Training time is the Hugging Face Trainer runtime, including epoch-end validation '
    'at length 512, and excluding model saving and the two final validation runs. '
    f'Peak allocated memory was captured before the final validations; peak reserved '
    f'CUDA memory over that interval was {reserved:.2f} MiB. '
)
if sampled_peak is not None:
    measurement += f'The maximum device VRAM sampled by \\texttt{{nvidia-smi}} once per second over the full run, including final evaluation, was {sampled_peak:,.0f} MiB. '
measurement += 'All memory values are in MiB.\n\n'
if p2048 > p512:
    reasoning = (
        f'The perplexity on the 2048-token set is higher ({p2048:.2f} versus {p512:.2f}). '
        'This is consistent with a mismatch between training and evaluation context lengths. '
        'During training, the model sees positions 0--511 and relative attention distances up to 511. '
        'Evaluation at length 2048 introduces larger positions and relative distances that were not '
        'represented in training. Although rotary position embeddings (RoPE) can be evaluated at these '
        'positions, the learned attention patterns need not extrapolate reliably. The additional '
        'context therefore does not guarantee lower perplexity for a model trained only at length 512. '
        'Continuing training at length 2048 would expose the model to the longer-range attention '
        'patterns used at evaluation.\n\n'
    )
else:
    reasoning = (
        f'The measured perplexity on the 2048-token set is {p2048:.2f}, compared with {p512:.2f} '
        'on the 512-token set. The longer sequence provides more preceding context for prediction, '
        'which can improve perplexity, while also introducing positions and relative distances beyond '
        'those encountered during 512-token training. RoPE can be evaluated at those positions, '
        'although extrapolation of the learned attention patterns is not guaranteed. In this run '
        'the longer-context benefit was sufficient to offset that mismatch. These validation sets '
        'also use different segmentation, so their scores should not be interpreted as differing '
        'only in the amount of available context.\n\n'
    )
tex_path = a.root/'Assignment.tex'
tex = tex_path.read_text()
start = tex.index('\\subsection*{1.1.A}')
end = tex.index('\\section*{Question 1.2 (5 points)}')
tex = tex[:start]+'\\subsection*{1.1.A}\n'+setup+table+measurement+'\\subsection*{1.1.B}\n'+reasoning+'\\clearpage\n'+tex[end:]
tex_path.write_text(tex)
print(f'Filled Q1.1 using {a.metrics}; Q1.2/Q1.3 content preserved.')
