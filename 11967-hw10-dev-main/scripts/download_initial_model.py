"""Download the supplied public initialization at one recorded HF revision."""
import json
import subprocess
import sys
import urllib.request
from pathlib import Path

provenance = json.loads((Path(__file__).resolve().parents[1]/'model_src/SOURCE.json').read_text())
repo = provenance['repository']
revision = provenance['revision']
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

subprocess.run([sys.executable, str(Path(__file__).resolve().with_name("install_custom_model.py"))], check=True)
