#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
"${A10_PYTHON:-python}" scripts/download_initial_model.py
