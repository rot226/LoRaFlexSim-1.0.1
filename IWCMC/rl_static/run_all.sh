#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

python IWCMC/rl_static/scenarios/run_ucb1_vs_qos.py "$@"
python IWCMC/rl_static/plots/plot_rls_figures.py
