"""Tracé de la figure 2 depuis outputs/csv/fig2.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from core.utils import ensure_output_dirs, path_join


def run(base_dir: str) -> str:
    """Produit outputs/figures/figure2.png à partir du CSV fig2."""
    csv_path = path_join(base_dir, "outputs", "csv", "fig2.csv")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["episode"], df["reward_v1"], label="v=1")
    ax.plot(df["episode"], df["reward_v5"], label="v=5")
    ax.plot(df["episode"], df["reward_v10"], label="v=10")
    ax.set_xlabel("Épisode")
    ax.set_ylabel("Reward")
    ax.set_title("Courbe d'apprentissage")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    dirs = ensure_output_dirs(base_dir)
    out_path = path_join(dirs["figures"], "figure2.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    run(str(Path(__file__).resolve().parents[1]))
