"""Install versioned Q1.2 source into a downloaded model, preserving its weights."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--model-dir', type=Path, default=Path(__file__).resolve().parents[1]/'models/GPTNeoX-160m')
a = p.parse_args()
source = Path(__file__).resolve().parents[1]/'model_src/modeling_custom.py'
config = json.loads((a.model_dir/'config.json').read_text())
if config.get('auto_map', {}).get('AutoModelForCausalLM') != 'modeling_custom.GPTNeoXForCausalLM':
    raise SystemExit('Expected the HW10 custom GPTNeoX checkpoint; source was not installed.')
target = a.model_dir/'modeling_custom.py'
if target.read_bytes() == source.read_bytes():
    print('Q1.2 model source is already installed.')
else:
    backup = a.model_dir/'modeling_custom.before_q12.py'
    if not backup.exists():
        shutil.copy2(target, backup)
    shutil.copy2(source, target)
    print('Installed Q1.2 source; preserved original source and model weights.')
print('modeling_custom.py SHA256:', hashlib.sha256(target.read_bytes()).hexdigest())
