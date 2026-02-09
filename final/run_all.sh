#!/usr/bin/env bash
set -euo pipefail

# Script d'orchestration (Linux/macOS) :
# - Enchaîne les simulations QoS baselines, SNIR et UCB1.
# - Ajustez les paramètres ci-dessous selon vos besoins.
#
# Exemple :
#   ./final/run_all.sh
#   CELL_RADIUS=3000 RUNS=5 PERIOD=600 DURATION=43200 ./final/run_all.sh

DATA_DIR="${DATA_DIR:-final/data}"
CELL_RADIUS="${CELL_RADIUS:-2500}"
RUNS="${RUNS:-10}"
PERIOD="${PERIOD:-300}"
DURATION="${DURATION:-86400}"

# Chemin optionnel vers un INI (laissez vide si non utilisé).
CONFIG_PATH="${CONFIG_PATH:-}"

# Tailles de réseau (N ∈ {1000,...,15000}).
NODES_LIST=(1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 11000 12000 13000 14000 15000)

CONFIG_ARGS=()
if [[ -n "${CONFIG_PATH}" ]]; then
  CONFIG_ARGS=(--config "${CONFIG_PATH}")
fi

for NODES in "${NODES_LIST[@]}"; do
  python final/scenarios/run_qos_baselines.py \
    --cell-radius "${CELL_RADIUS}" \
    --nodes "${NODES}" \
    --runs "${RUNS}" \
    --period "${PERIOD}" \
    --duration "${DURATION}" \
    --output-dir "${DATA_DIR}/qos_baselines/N${NODES}" \
    "${CONFIG_ARGS[@]}"

  python final/scenarios/run_snir.py \
    --cell-radius "${CELL_RADIUS}" \
    --nodes "${NODES}" \
    --runs "${RUNS}" \
    --period "${PERIOD}" \
    --duration "${DURATION}" \
    --output-dir "${DATA_DIR}/snir/N${NODES}" \
    "${CONFIG_ARGS[@]}"

  python final/scenarios/run_ucb1.py \
    --cell-radius "${CELL_RADIUS}" \
    --nodes "${NODES}" \
    --runs "${RUNS}" \
    --period "${PERIOD}" \
    --duration "${DURATION}" \
    --output-dir "${DATA_DIR}/ucb1/N${NODES}" \
    "${CONFIG_ARGS[@]}"
done
