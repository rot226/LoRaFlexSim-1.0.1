#!/usr/bin/env bash
set -euo pipefail

# End-to-end helper executing the recommended CI workflow:
#  1. Run the Python test-suite.
#  2. Regenerate key Article A/B datasets and plots.
#  3. Execute targeted node-population sweeps (including ADR sensitivity runs).
#  4. Export summary tables in CSV/LaTeX format.
#  5. Cross-check the simulator against the FLoRa validation matrix.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN=${PYTHON:-python}
PROFILE=${MNE3SD_PROFILE:-ci}
NODE_SERIES=(20 50 100 150)
ADR_SERIES=(100 150)

log() {
  printf '\n[%s] %s\n' "$(date '+%H:%M:%S')" "$*"
}

run_density_sweep() {
  local nodes=$1
  log "Article A – class density sweep (${nodes} nodes, profile ${PROFILE})"
  "${PYTHON_BIN}" -m scripts.mne3sd.article_a.scenarios.run_class_density_sweep \
    --nodes-list "${nodes}" \
    --profile "${PROFILE}" \
    --seed $((40 + nodes))
  local output="${ROOT_DIR}/results/mne3sd/article_a/class_density_metrics.csv"
  local target="${ROOT_DIR}/results/mne3sd/article_a/class_density_metrics_nodes_${nodes}.csv"
  if [[ -f "${output}" ]]; then
    mv -f "${output}" "${target}"
  else
    log "Warning: expected density metrics at ${output} not found"
  fi
}

run_density_sweep_adr() {
  local nodes=$1
  log "Article A – ADR sensitivity (${nodes} nodes)"
  "${PYTHON_BIN}" -m scripts.mne3sd.article_a.scenarios.run_class_density_sweep \
    --nodes-list "${nodes}" \
    --profile "${PROFILE}" \
    --adr-node \
    --adr-server \
    --seed $((140 + nodes))
  local output="${ROOT_DIR}/results/mne3sd/article_a/class_density_metrics.csv"
  local target="${ROOT_DIR}/results/mne3sd/article_a/class_density_metrics_adr_nodes_${nodes}.csv"
  if [[ -f "${output}" ]]; then
    mv -f "${output}" "${target}"
  else
    log "Warning: expected ADR density metrics at ${output} not found"
  fi
}

run_range_sweep() {
  local nodes=$1
  log "Article B – mobility range sweep (${nodes} nodes, profile ${PROFILE})"
  "${PYTHON_BIN}" -m scripts.mne3sd.article_b.scenarios.run_mobility_range_sweep \
    --nodes "${nodes}" \
    --profile "${PROFILE}" \
    --seed $((80 + nodes))
  local output="${ROOT_DIR}/results/mne3sd/article_b/mobility_range_metrics.csv"
  local target="${ROOT_DIR}/results/mne3sd/article_b/mobility_range_metrics_nodes_${nodes}.csv"
  if [[ -f "${output}" ]]; then
    mv -f "${output}" "${target}"
  else
    log "Warning: expected mobility range metrics at ${output} not found"
  fi
}

run_range_sweep_adr() {
  local nodes=$1
  log "Article B – ADR sensitivity (${nodes} nodes)"
  "${PYTHON_BIN}" -m scripts.mne3sd.article_b.scenarios.run_mobility_range_sweep \
    --nodes "${nodes}" \
    --profile "${PROFILE}" \
    --adr-node \
    --adr-server \
    --seed $((180 + nodes))
  local output="${ROOT_DIR}/results/mne3sd/article_b/mobility_range_metrics.csv"
  local target="${ROOT_DIR}/results/mne3sd/article_b/mobility_range_metrics_adr_nodes_${nodes}.csv"
  if [[ -f "${output}" ]]; then
    mv -f "${output}" "${target}"
  else
    log "Warning: expected ADR mobility metrics at ${output} not found"
  fi
}

export_tables() {
  log "Exporting summary tables"
  "${PYTHON_BIN}" -m scripts.mne3sd.export_node_summaries \
    --inputs "results/mne3sd/article_a/class_density_metrics_nodes_*.csv" \
    --group-columns class \
    --output-csv "results/mne3sd/article_a/tables/class_density_summary.csv" \
    --output-tex "results/mne3sd/article_a/tables/class_density_summary.tex" \
    --tex-caption "Article A node population sweep (classes A/B/C)." \
    --tex-label "tab:article_a_node_summary"

  "${PYTHON_BIN}" -m scripts.mne3sd.export_node_summaries \
    --inputs "results/mne3sd/article_b/mobility_range_metrics_nodes_*.csv" \
    --group-columns model range_km \
    --output-csv "results/mne3sd/article_b/tables/mobility_range_summary.csv" \
    --output-tex "results/mne3sd/article_b/tables/mobility_range_summary.tex" \
    --tex-caption "Article B mobility range sweep across node populations." \
    --tex-label "tab:article_b_node_summary"
}

cd "${ROOT_DIR}"

log "Running unit tests"
pytest -q

log "Executing article pipelines (profile ${PROFILE})"
"${PYTHON_BIN}" -m scripts.mne3sd.run_all_article_outputs --article both --profile "${PROFILE}"

for nodes in "${NODE_SERIES[@]}"; do
  run_density_sweep "${nodes}"
  run_range_sweep "${nodes}"
done

for nodes in "${ADR_SERIES[@]}"; do
  run_density_sweep_adr "${nodes}"
  run_range_sweep_adr "${nodes}"
done

export_tables

log "Running FLoRa validation matrix"
"${PYTHON_BIN}" scripts/run_validation.py --output results/validation_matrix.csv

log "Workflow completed"
