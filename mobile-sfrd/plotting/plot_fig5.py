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

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["t"], df["pdr"], color="tab:blue", linewidth=1.6, label="PDR")
    ax.axvline(cp, color="tab:red", linestyle="--", linewidth=1.2, label=f"Change-point (t={cp})")
    ax.annotate(
        f"Change-point\nt={cp}",
        xy=(cp, df.loc[df["t"] == cp, "pdr"].iloc[0]),
        xytext=(cp + 12, min(0.98, df["pdr"].max() + 0.04)),
        arrowprops={"arrowstyle": "->", "color": "tab:red", "lw": 1.0},
        fontsize=9,
        color="tab:red",
    )
    ax.set_xlabel("Temps")
    ax.set_ylabel("PDR")
    ax.set_title("Détection de changement")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25)
    ax.legend()

    fig.tight_layout()
    dirs = ensure_output_dirs(base_dir)
    out_path = path_join(dirs["figures"], "figure5.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    run(str(Path(__file__).resolve().parents[1]))
