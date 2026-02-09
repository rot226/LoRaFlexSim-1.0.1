#!/usr/bin/env bash
set -euo pipefail

IWCMC_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
REPO_DIR=$(cd "$IWCMC_DIR/.." && pwd)
ARCHIVE_DIR="$IWCMC_DIR/archive"

mkdir -p "$ARCHIVE_DIR"

stamp=$(date +"%Y%m%d_%H%M%S")
archive_path="$ARCHIVE_DIR/iwcmc_results_${stamp}.tar.gz"

targets=()
for rel in "IWCMC/snir_static/data" "IWCMC/snir_static/figures" "IWCMC/rl_static/figures" "IWCMC/rl_mobile/figures" "results/iwcmc"; do
  if [ -d "$REPO_DIR/$rel" ]; then
    targets+=("$rel")
  fi
done

if [ ${#targets[@]} -eq 0 ]; then
  echo "Aucun dossier de résultats à archiver." >&2
  exit 1
fi

tar -czf "$archive_path" -C "$REPO_DIR" "${targets[@]}"

echo "Archive créée : $archive_path"
