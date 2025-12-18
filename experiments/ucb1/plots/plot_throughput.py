"""Trace un débit relatif (taux de succès) par cluster et au global.

Le script lit un CSV de métriques (balayage de densité par défaut) et représente
un taux de réussite assimilé au débit livré. Le calcul global agrège les
clusters via une moyenne pondérée par leurs proportions.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "ucb1_density_metrics.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "plots" / "ucb1_throughput.png"
DEFAULT_PROPORTIONS: Sequence[float] = (0.1, 0.3, 0.6)
AVAILABLE_METRICS: Sequence[str] = ("success_rate", "der", "pdr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace un débit relatif par cluster et globalement.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT, help="Chemin du CSV de métriques.")
    parser.add_argument(
        "--metric",
        choices=AVAILABLE_METRICS,
        default="success_rate",
        help="Colonne utilisée comme indicateur de débit.",
    )
    parser.add_argument(
        "--proportions",
        type=float,
        nargs="*",
        default=list(DEFAULT_PROPORTIONS),
        help="Proportions des clusters pour la moyenne globale (même ordre que les clusters).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Chemin du PNG à générer (répertoires créés si besoin).",
    )
    return parser.parse_args()


def _load_csv(path: Path, metric: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Impossible de trouver le CSV '{path}'.")
    df = pd.read_csv(path)
    required_cols = {"num_nodes", "cluster", metric}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {', '.join(sorted(missing))}")
    return df


def _compute_global(df: pd.DataFrame, metric: str, proportions: Iterable[float]) -> pd.DataFrame:
    clusters = sorted(df["cluster"].unique())
    proportions_list = list(proportions)
    if len(proportions_list) < len(clusters):
        raise ValueError("Le nombre de proportions fourni est inférieur au nombre de clusters détectés.")

    global_rows: list[tuple[int, float]] = []
    for num_nodes, group in df.groupby("num_nodes"):
        weighted_values = []
        weights = []
        for cluster_id, weight in zip(clusters, proportions_list):
            row = group[group["cluster"] == cluster_id]
            if row.empty:
                continue
            weighted_values.append(float(row.iloc[0][metric]) * weight)
            weights.append(weight)
        if weights:
            global_rows.append((int(num_nodes), sum(weighted_values) / sum(weights)))

    return pd.DataFrame(global_rows, columns=["num_nodes", metric])


def _plot_throughput(df: pd.DataFrame, global_df: pd.DataFrame, metric: str, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    clusters = sorted(df["cluster"].unique())
    for cluster_id in clusters:
        subset = df[df["cluster"] == cluster_id].sort_values("num_nodes")
        ax.plot(
            subset["num_nodes"],
            subset[metric],
            marker="o",
            label=f"Cluster {cluster_id}",
        )

    if not global_df.empty:
        ax.plot(
            global_df["num_nodes"],
            global_df[metric],
            linestyle="--",
            color="black",
            marker="s",
            label="Global (pondéré)",
        )

    ax.set_xlabel("Nombre de nœuds")
    ax.set_ylabel("Taux de succès / débit relatif")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    ax.set_title(f"Débit relatif basé sur '{metric}' (UCB1)")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    print(f"Figure enregistrée dans {output}")


def main() -> None:
    args = parse_args()
    df = _load_csv(args.csv, args.metric)
    global_df = _compute_global(df, args.metric, args.proportions)
    _plot_throughput(df, global_df, args.metric, args.output)


if __name__ == "__main__":
    main()
