"""Expérience figure 5 : génération CSV + tracé."""

from __future__ import annotations

from pathlib import Path

from core.generators import generate_fig5_changepoint
from core.seeds import spawn_rng
from core.utils import ensure_output_dirs, load_yaml, path_join, save_csv
from plotting import plot_fig5


def run(base_dir: str) -> str:
    """Exécute le pipeline complet de la figure 5."""
    config = load_yaml(path_join(base_dir, "config", "fig5.yaml"))
    seed = int(config.get("seed", 202405))
    rng = spawn_rng(seed, "fig5")

    df = generate_fig5_changepoint(config, rng)
    dirs = ensure_output_dirs(base_dir)
    csv_path = path_join(dirs["csv"], "fig5.csv")
    save_csv(df, csv_path)

    plot_fig5.run(base_dir)
    return csv_path


if __name__ == "__main__":
    run(str(Path(__file__).resolve().parents[1]))
