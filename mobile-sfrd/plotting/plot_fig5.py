"""Tracé de la figure 5 depuis outputs/csv/fig5.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from core.utils import ensure_output_dirs, path_join


def run(base_dir: str) -> str:
    """Produit outputs/figures/figure5.png à partir du CSV fig5."""
    csv_path = path_join(base_dir, "outputs", "csv", "fig5.csv")
    df = pd.read_csv(csv_path)
    cp = int(df["changepoint_t"].iloc[0])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["t"], df["pdr"], color="tab:blue", linewidth=1.5)
    ax.axvline(cp, color="tab:red", linestyle="--", linewidth=1.2, label=f"Changepoint t={cp}")
    ax.set_xlabel("Temps")
    ax.set_ylabel("PDR")
    ax.set_title("Détection de changement")
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.25)
    ax.legend()

    fig.tight_layout()
    dirs = ensure_output_dirs(base_dir)
    out_path = path_join(dirs["figures"], "figure5.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    run(str(Path(__file__).resolve().parents[1]))
