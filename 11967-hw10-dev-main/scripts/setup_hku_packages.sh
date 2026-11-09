#!/usr/bin/env bash
# Install Q1.1 dependencies in an already-created Python 3.11 environment.
set -euo pipefail
A10_PYTHON="${A10_PYTHON:-$HOME/hw10-env/bin/python}"
"$A10_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 11), "Use the dedicated Python 3.11 environment"'
"$A10_PYTHON" -m pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu124
"$A10_PYTHON" -m pip install transformers==4.45.0 accelerate==0.34.2 datasets==3.0.1 numpy==1.26.4 wandb 'huggingface-hub[cli]<1'
"$A10_PYTHON" -m pip check
# FlashAttention is not required for the Q1.1 eager baseline.
# Install/validate it separately before Q1.2, retaining these versions for comparison.
