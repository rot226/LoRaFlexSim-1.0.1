#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python IWCMC/rl_mobile/scenarios/run_rl_mobile.py "$@"
python IWCMC/rl_mobile/plots/plot_rlm_figures.py
