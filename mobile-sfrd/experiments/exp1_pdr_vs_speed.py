"""Expérience figure 1 : génération CSV + tracé."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.seeds import spawn_rng
from core.utils import ensure_output_dirs, load_yaml, path_join, save_csv
from loraflexsim.launcher.mobility_effects import generate_fig1_pdr_vs_speed
from plotting import plot_fig1


def run(base_dir: str) -> str:
    """Exécute le pipeline complet de la figure 1."""
    config = load_yaml(path_join(base_dir, "config", "fig1.yaml"))
    seed = int(config.get("seed", 202401))
    rng = spawn_rng(seed, "fig1")

    df = generate_fig1_pdr_vs_speed(config, rng)
    dirs = ensure_output_dirs(base_dir)
    csv_path = path_join(dirs["csv"], "fig1.csv")
    save_csv(df, csv_path)

    plot_fig1.run(base_dir)
    return csv_path


if __name__ == "__main__":
    run(str(Path(__file__).resolve().parents[1]))
