"""Expérience figure 4 : génération CSV + tracé."""

from __future__ import annotations

from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.seeds import spawn_rng
from core.utils import ensure_output_dirs, load_yaml, path_join, save_csv
from loraflexsim.launcher.mobility_effects import generate_fig4_der_vs_speed
from plotting import plot_fig4


def run(base_dir: str) -> str:
    """Exécute le pipeline complet de la figure 4."""
    config = load_yaml(path_join(base_dir, "config", "fig4.yaml"))
    seed = int(config.get("seed", 202404))
    rng = spawn_rng(seed, "fig4")

    df = generate_fig4_der_vs_speed(config, rng)
    dirs = ensure_output_dirs(base_dir)
    csv_path = path_join(dirs["csv"], "fig4.csv")
    save_csv(df, csv_path)

    plot_fig4.run(base_dir)
    return csv_path


if __name__ == "__main__":
    run(str(Path(__file__).resolve().parents[1]))
