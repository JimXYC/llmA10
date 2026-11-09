"""Populate Q1.3.A--E only from completed paired runs and measured batch limits."""
import argparse
import csv
import json
import math
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
a=p.parse_args();root=a.root
probe=json.loads((root/'q13-results/batch-probe/summary.json').read_text())
prior={q:json.loads((root/f'q{q}-results/hku/metrics.json').read_text()) for q in ('11','12')}
data={};configs={};device_peaks={}
for variant in ('retrofit','scratch'):
    folder=root/f'q13-results/hku/{variant}'
    assert (folder/'exit-code').read_text().strip()=='0'
    m=json.loads((folder/'metrics.json').read_text())
    state=json.loads((folder/'trainer_state.json').read_text())
    c=json.loads((folder/'training-config.json').read_text())
    assert m['seq_len']==2048 and m['attention_type']=='flash_attention_2' and m['gradient_checkpointing']
    assert m['train']['epoch']>0.99 and state['global_step']==state['max_steps']
    assert c['gradient_checkpointing_kwargs']=={'use_reentrant':False}
    assert m['gpu']==prior['12']['gpu'] and m['torch_version']==prior['12']['torch_version']
    for length in ('512','2048'):
        d=m['validation_'+length]
        assert math.isfinite(d['perplexity']) and math.isclose(math.exp(d['eval_loss']),d['perplexity'])
    assert m['train']['train_runtime']>0 and math.isfinite(m['train']['train_loss'])
    with (folder/'gpu-usage.csv').open() as f:
        rows=csv.reader(f);next(rows)
        device_peaks[variant]=max(float(r[2].split()[0]) for r in rows if len(r)>=4)
    data[variant]=m;configs[variant]=c
changes={k for k in configs['retrofit'].keys()|configs['scratch'].keys() if configs['retrofit'].get(k)!=configs['scratch'].get(k)}
assert changes=={'model_to_train','run_name','output_dir'}
assert configs['retrofit']['model_to_train']=='models/GPTNeoX-160M-WikiText-512-flash'
assert configs['scratch']['model_to_train']=='models/GPTNeoX-160m'
for mode in ('eager_no_gc','flash_gc'):
    b=probe[mode];assert b['maximum_confirmed']
    limit=b['maximum_batch_size'];trials={d['batch_size']:d for d in b['attempts']}
    assert trials[limit]['status']=='ok' and trials[limit+1]['status']=='oom'
    assert trials[limit]['optimizer_steps']==3 and trials[limit]['seq_len']==2048
r,s=data['retrofit'],data['scratch'];c=configs['retrofit']
ppl=lambda m,n:m['validation_'+str(n)]['perplexity']
bs,ga=c['per_device_train_batch_size'],c['gradient_accumulation_steps']
low=probe['eager_no_gc']['maximum_batch_size'];high=probe['flash_gc']['maximum_batch_size']
assert bs<=high
text=rf'''\subsection*{{1.3.A}}
On the RTX 4080 SUPER, I measured per-device batch limits using BF16, length 2048, the same Q1.2 checkpoint, accumulation 1, and a fresh process for every candidate. Each successful candidate completed three actual Trainer optimizer steps, including optimizer-state allocation; generation KV caching was disabled in both settings.

\begin{{center}}
\begin{{tabular}}{{lrr}}
\toprule
Setting & Largest passed batch & Next batch (OOM) \\
\midrule
Eager, checkpointing off & {low} & {low+1} \\
FlashAttention 2, checkpointing on & {high} & {high+1} \\
\bottomrule
\end{{tabular}}
\end{{center}}

These limits describe this fixed-shape three-step training probe. For the complete runs I used batch {bs} with {ga} accumulation steps to retain memory headroom.

\subsection*{{1.3.B}}
Gradient checkpointing keeps selected activations from the forward pass and recomputes missing intermediates during backpropagation. It reduces activation memory at the cost of additional computation, which can make training slower (\href{{https://github.com/cybertronai/gradient-checkpointing}}{{linked reference}}).

\subsection*{{1.3.C}}
I continued from Q1.2 for one epoch on {r['train_examples']:,} length-2048 WikiText sequences ({r['train_tokens']:,} tokens), using FlashAttention 2, non-reentrant checkpointing and BF16. Batch {bs} and accumulation {ga} give {bs*ga*2048:,} tokens per update; I used peak learning rate $10^{{-4}}$, 1\% warmup, cosine decay, AdamW, and seed 42. The matched scratch control uses the same settings.

\begin{{center}}
\begin{{tabular}}{{lrr}}
\toprule
Metric & Continued from Q1.2 & From scratch \\
\midrule
Validation perplexity, length 512 & {ppl(r,512):.2f} & {ppl(s,512):.2f} \\
Validation perplexity, length 2048 & {ppl(r,2048):.2f} & {ppl(s,2048):.2f} \\
Training time (seconds) & {r['train']['train_runtime']:.2f} & {s['train']['train_runtime']:.2f} \\
Peak allocated training VRAM (MiB) & {r['peak_allocated_mib']:.2f} & {s['peak_allocated_mib']:.2f} \\
\bottomrule
\end{{tabular}}
\end{{center}}

Timing includes epoch-end validation and excludes saving/final validations, as in Q1.1/Q1.2. Allocated training peaks are captured before final validation. Reserved peaks were {r['peak_reserved_mib']:.0f}/{s['peak_reserved_mib']:.0f} MiB, and whole-run device sampling peaks were {device_peaks['retrofit']:.0f}/{device_peaks['scratch']:.0f} MiB (continued/scratch).

\subsection*{{1.3.D}}
'''
old=ppl(prior['12'],2048);new=ppl(r,2048)
if new<old:
    text+=(f'The 2048-token perplexity decreased from {old:.2f} in Q1.2 ({ppl(prior["11"],2048):.2f} in Q1.1) to {new:.2f}. '
           'The model now trains on the longer positions and relative attention distances used at evaluation, reducing the context-length mismatch; additional WikiText training also improves its predictions.\n\n')
else:
    text+=(f'The 2048-token perplexity changed from {old:.2f} in Q1.2 to {new:.2f}. '
           'This run did not improve the long-context validation score despite direct training at length 2048; training length alone does not guarantee improvement, and the continued-training schedule and validation behavior merit further inspection.\n\n')
text+='\\subsection*{1.3.E}\n'
text+=(f'The scratch model starts from the supplied initialization, with the same 2048-token data, hyperparameters and one-epoch budget as the continuation run. '
       f'It reached perplexities {ppl(s,512):.2f}/{ppl(s,2048):.2f}, compared with {ppl(r,512):.2f}/{ppl(r,2048):.2f} for continuation (512/2048). ')
if ppl(s,512)>ppl(r,512) and ppl(s,2048)>ppl(r,2048):
    text+='The lower continuation perplexities are consistent with the benefit of the language-model knowledge already learned in Q1.2. '
else:
    text+='The observed scores do not show a uniform advantage for continuation at both lengths. '
text+=('The continued model has also received the earlier 512-token training in Q1.2, giving it more total training exposure and compute; this comparison does not isolate curriculum effects at equal total training budget.\n')
path=root/'Assignment.tex';tex=path.read_text()
start=tex.index('\\subsection*{1.3.A}');end=tex.index('\\end{document}',start)
path.write_text(tex[:start]+text+tex[end:])
summary={'status':'complete','batch_limits':{'eager_no_gc':low,'flash_gc':high},'chosen_batch':bs,'gradient_accumulation':ga,
         'variants':{v:{'ppl_512':ppl(m,512),'ppl_2048':ppl(m,2048),'training_seconds':m['train']['train_runtime'],
                       'peak_allocated_mib':m['peak_allocated_mib'],'peak_reserved_mib':m['peak_reserved_mib'],
                       'peak_device_sampled_mib':device_peaks[v]} for v,m in data.items()}}
(root/'q13-results/measurement-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
