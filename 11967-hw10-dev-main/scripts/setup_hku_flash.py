"""Install the official FA2 wheel matching the unchanged HW10 PyTorch runtime."""
import argparse
import json
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

import torch

p = argparse.ArgumentParser()
p.add_argument('--receipt', type=Path, default=Path('../q12-results/flash-install.json'))
a = p.parse_args()
if torch.__version__.split('+')[0] != '2.4.1' or not (torch.version.cuda or '').startswith('12.'):
    raise SystemExit('Use the existing HW10 PyTorch 2.4.1 CUDA 12 environment; do not replace Q1.1 dependencies.')
if not torch.cuda.is_available():
    raise SystemExit('Run inside the allocated GPU compute session.')
subprocess.run([sys.executable, '-m', 'pip', 'install', 'pytest>=8,<9', 'einops', 'packaging'], check=True)
req = urllib.request.Request('https://api.github.com/repos/Dao-AILab/flash-attention/releases/tags/v2.6.3', headers={'User-Agent':'llmA10-HW10'})
with urllib.request.urlopen(req, timeout=60) as response:
    release = json.load(response)
abi = 'cxx11abi' + str(torch._C._GLIBCXX_USE_CXX11_ABI).upper()
cp = f'cp{sys.version_info.major}{sys.version_info.minor}'
assets = [v for v in release['assets'] if '+cu12' in v['name'] and 'torch2.4' in v['name'] and abi in v['name'] and v['name'].endswith(f'-{cp}-{cp}-linux_x86_64.whl')]
if len(assets) != 1:
    raise SystemExit(f'Expected one official matching wheel, found {len(assets)}; no package installed.')
asset = assets[0]
print('Official wheel:', asset['name'], flush=True)
with tempfile.TemporaryDirectory(prefix='hw10-flash-') as folder:
    wheel = Path(folder)/asset['name']
    urllib.request.urlretrieve(asset['browser_download_url'], wheel)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-deps', str(wheel)], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'check'], check=True)
a.receipt.parent.mkdir(parents=True, exist_ok=True)
a.receipt.write_text(json.dumps({'flash_attention':'2.6.3','official_wheel':asset['name'],'download_url':asset['browser_download_url'],'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'gpu':torch.cuda.get_device_name(0)}, indent=2)+'\n')
