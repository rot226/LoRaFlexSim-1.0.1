"""Tracé de la figure 3 depuis outputs/csv/fig3.csv."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from core.utils import ensure_output_dirs, path_join


def run(base_dir: str) -> str:
    """Produit outputs/figures/figure3.png à partir du CSV fig3."""
    csv_path = path_join(base_dir, "outputs", "csv", "fig3.csv")
    df = pd.read_csv(csv_path)

    panels = [("SM", 1), ("SM", 10), ("RWP", 1), ("RWP", 10)]
    fig, axes = plt.subplots(2, 2, figsize=(7.6, 5), sharey=True)

    for ax, (mobility, speed) in zip(axes.flatten(), panels):
        sub = df[(df["mobility"] == mobility) & (df["speed"] == speed)]
        pivot = sub.pivot(index="sf", columns="window", values="nodes_count").fillna(0)
        x = pivot.index.to_numpy()
        ax.bar(x - 0.2, pivot.get("initial", 0), width=0.4, label="Initial")
        ax.bar(x + 0.2, pivot.get("final", 0), width=0.4, label="Final")
        ax.set_title(f"{mobility} - {speed} m/s")
        ax.grid(axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.supylabel("Nombre de nœuds")
    fig.supxlabel("SF")
    fig.suptitle("Distribution des SF")
    fig.tight_layout(rect=(0.04, 0.04, 1, 0.93))

    dirs = ensure_output_dirs(base_dir)
    out_path = path_join(dirs["figures"], "figure3.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    run(str(Path(__file__).resolve().parents[1]))
