"""Download the supplied public initialization at one recorded HF revision."""
import json
import urllib.request
from pathlib import Path

repo = 'jmvcoelho/GPTNeoX-160m'
with urllib.request.urlopen(f'https://huggingface.co/api/models/{repo}', timeout=60) as response:
    info = json.load(response)
revision = info['sha']
files = ['config.json', 'config_custom.py', 'generation_config.json', 'modeling_custom.py',
         'model.safetensors', 'special_tokens_map.json', 'tokenizer.json', 'tokenizer_config.json']
root = Path('models/GPTNeoX-160m')
root.mkdir(parents=True, exist_ok=True)
for name in files:
    print(f'Downloading {name} at {revision}', flush=True)
    partial = root / (name + '.part')
    urllib.request.urlretrieve(f'https://huggingface.co/{repo}/resolve/{revision}/{name}', partial)
    partial.replace(root / name)
(root / 'initialization_revision.json').write_text(json.dumps(dict(repo=repo, revision=revision, files=files), indent=2)+'\n')
print('Initialization download complete.', flush=True)
