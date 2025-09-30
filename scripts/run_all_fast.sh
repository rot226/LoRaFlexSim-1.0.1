#!/usr/bin/env bash
set -euo pipefail

# Quick orchestrator executing representative scenarios in fast mode.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON:-python}

cd "$ROOT_DIR"

run() {
  echo "[FAST] $*"
  "${PYTHON_BIN}" -m loraflexsim.run "$@" --fast --sample-size 0.2 >/dev/null
}

run --nodes 50 --gateways 1 --channels 3 --mode random --interval 120 --steps 7200
run --nodes 20 --gateways 2 --channels 6 --mode periodic --interval 300 --steps 3600
run --nodes 80 --gateways 1 --channels 8 --mode random --interval 60 --steps 14400

echo "Fast regression suite completed."

