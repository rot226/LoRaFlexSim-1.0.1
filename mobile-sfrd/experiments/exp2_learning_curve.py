"""Expérience figure 2 : génération CSV + tracé."""

from __future__ import annotations

from pathlib import Path

from core.generators import generate_fig2_learning_curve
from core.seeds import spawn_rng
from core.utils import ensure_output_dirs, load_yaml, path_join, save_csv
from plotting import plot_fig2


def run(base_dir: str) -> str:
    """Exécute le pipeline complet de la figure 2."""
    config = load_yaml(path_join(base_dir, "config", "fig2.yaml"))
    seed = int(config.get("seed", 202402))
    rng = spawn_rng(seed, "fig2")

    df = generate_fig2_learning_curve(config, rng)
    dirs = ensure_output_dirs(base_dir)
    csv_path = path_join(dirs["csv"], "fig2.csv")
    save_csv(df, csv_path)

    plot_fig2.run(base_dir)
    return csv_path


if __name__ == "__main__":
    run(str(Path(__file__).resolve().parents[1]))
