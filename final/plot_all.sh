#!/usr/bin/env bash
set -euo pipefail

# Script de tracé (Linux/macOS) :
# - Génère toutes les figures à partir des CSV déjà présents.
# - Ajustez DATA_DIR / FIGURES_DIR si besoin.
#
# Exemple :
#   ./final/plot_all.sh

DATA_DIR="${DATA_DIR:-final/data}"
FIGURES_DIR="${FIGURES_DIR:-final/figures}"

python final/plots/plot_der_vs_nodes.py --data-dir "${DATA_DIR}" --output-dir "${FIGURES_DIR}"
python final/plots/plot_throughput.py --data-dir "${DATA_DIR}" --output-dir "${FIGURES_DIR}"
python final/plots/plot_snir_distribution.py --data-dir "${DATA_DIR}" --output-dir "${FIGURES_DIR}"
