"""Tracé de la figure 1 depuis outputs/csv/fig1.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from core.utils import ensure_output_dirs, path_join


def run(base_dir: str) -> str:
    """Produit outputs/figures/figure1.png à partir du CSV fig1."""
    csv_path = path_join(base_dir, "outputs", "csv", "fig1.csv")
    df = pd.read_csv(csv_path)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["speed"], df["pdr_sm"], marker="o", linewidth=1.8, label="SM")
    ax.plot(df["speed"], df["pdr_rwp"], marker="s", linewidth=1.8, label="RWP")
    ax.set_xlim(0, 11)
    ax.set_xticks(range(0, 12))
    ax.set_ylim(0, 1)
    ax.set_xlabel("Vitesse")
    ax.set_ylabel("PDR")
    ax.set_title("PDR vs vitesse")
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    dirs = ensure_output_dirs(base_dir)
    out_path = path_join(dirs["figures"], "figure1.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    run(str(Path(__file__).resolve().parents[1]))
