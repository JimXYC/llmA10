"""Check eager BF16 training and memory on the allocated GPU; save no model."""
import argparse
import json
import time
from pathlib import Path
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForLanguageModeling, set_seed

p = argparse.ArgumentParser()
p.add_argument('--batch-size', type=int, default=16)
p.add_argument('--output', required=True)
a = p.parse_args()
set_seed(42)
assert torch.cuda.is_available(), 'No GPU is visible in this allocation'
model = AutoModelForCausalLM.from_pretrained('models/GPTNeoX-160m', torch_dtype=torch.bfloat16, attn_implementation='eager', trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained('models/GPTNeoX-160m')
tokenizer.pad_token = tokenizer.eos_token
collate = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
data = load_dataset('cjiao/wikitext-tokenized', split='train_512')
batch = {k: v.cuda() for k, v in collate([data[i] for i in range(a.batch_size)]).items()}
optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4, betas=(0.9, 0.95), weight_decay=0.1)
model.train()
torch.cuda.reset_peak_memory_stats()
start = time.perf_counter()
losses = []
for _ in range(3):
    optimizer.zero_grad(set_to_none=True)
    loss = model(**batch).loss
    assert torch.isfinite(loss), 'Non-finite training loss'
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    losses.append(loss.item())
torch.cuda.synchronize()
record = dict(batch_size=a.batch_size, seq_len=512, losses=losses, seconds=time.perf_counter()-start,
              peak_allocated_mib=torch.cuda.max_memory_allocated()/1024**2,
              peak_reserved_mib=torch.cuda.max_memory_reserved()/1024**2,
              total_memory_mib=torch.cuda.get_device_properties(0).total_memory/1024**2,
              gpu=torch.cuda.get_device_name(0), torch=torch.__version__)
Path(a.output).write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record), flush=True)
