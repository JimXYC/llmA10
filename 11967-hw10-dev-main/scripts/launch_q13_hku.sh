#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${A10_PYTHON:-$HOME/hw10-env/bin/python}" scripts/run_q13.py "$@"
