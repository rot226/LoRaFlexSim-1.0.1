"""Trace le DER par cluster en fonction du nombre de nœuds.

Le script consomme le CSV généré par ``run_ucb1_density_sweep.py`` et ajoute
les lignes cibles de PDR/DER pour chaque cluster QoS.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "ucb1_density_metrics.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "plots" / "ucb1_der_by_cluster.png"
DEFAULT_TARGETS: Sequence[float] = (0.9, 0.8, 0.7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace le DER par cluster QoS.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT, help="Chemin du CSV de densité.")
    parser.add_argument(
        "--targets",
        type=float,
        nargs="*",
        default=list(DEFAULT_TARGETS),
        help="Objectifs DER/PDR par cluster (dans l'ordre des clusters).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Chemin du PNG à générer (répertoires créés si besoin).",
    )
    return parser.parse_args()


def _load_density_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Impossible de trouver le CSV '{path}'.")
    df = pd.read_csv(path)
    required_cols = {"num_nodes", "cluster", "der"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {', '.join(sorted(missing))}")
    return df


def _plot_der(df: pd.DataFrame, targets: Iterable[float], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    clusters = sorted(df["cluster"].unique())
    for cluster_id in clusters:
        subset = df[df["cluster"] == cluster_id].sort_values("num_nodes")
        ax.plot(
            subset["num_nodes"],
            subset["der"],
            marker="o",
            label=f"Cluster {cluster_id}",
        )

    for cluster_id, target in zip(clusters, targets):
        ax.axhline(target, linestyle="--", color="gray", linewidth=1, alpha=0.7)
        ax.text(
            df["num_nodes"].max(),
            target + 0.01,
            f"Cible {cluster_id}: {target:.0%}",
            ha="right",
            va="bottom",
            fontsize=9,
            color="gray",
        )

    ax.set_xlabel("Nombre de nœuds")
    ax.set_ylabel("Data Extraction Rate (DER)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    ax.set_title("DER par cluster lors du balayage de densité (UCB1)")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    print(f"Figure enregistrée dans {output}")


def main() -> None:
    args = parse_args()
    df = _load_density_csv(args.csv)
    _plot_der(df, args.targets, args.output)


if __name__ == "__main__":
    main()
