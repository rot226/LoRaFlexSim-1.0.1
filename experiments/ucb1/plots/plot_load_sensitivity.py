"""Visualise la sensibilité à la charge du DER pour UCB1.

Le script lit le CSV produit par ``run_ucb1_load_sweep.py`` et trace le DER par
cluster en fonction de l'intervalle de génération de paquets. Si la colonne
``packet_interval`` est absente (CSV généré avec le script actuel), les
intervalles sont déduits à partir des valeurs par défaut ou de l'argument
``--packet-intervals``.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import pandas as pd

DEFAULT_INPUT = Path(__file__).resolve().parents[1] / "ucb1_load_metrics.csv"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "plots" / "ucb1_load_sensitivity.png"
DEFAULT_INTERVALS: Sequence[float] = (300.0, 600.0, 900.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trace la sensibilité à la charge du DER.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_INPUT, help="Chemin du CSV de balayage de charge.")
    parser.add_argument(
        "--packet-intervals",
        type=float,
        nargs="*",
        default=list(DEFAULT_INTERVALS),
        help="Intervalles de paquets en secondes (utilisés si non présents dans le CSV).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Chemin du PNG à générer (répertoires créés si besoin).",
    )
    return parser.parse_args()


def _load_load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Impossible de trouver le CSV '{path}'.")
    df = pd.read_csv(path)
    required_cols = {"cluster", "der"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {', '.join(sorted(missing))}")
    return df


def _attach_intervals(df: pd.DataFrame, intervals: Iterable[float]) -> pd.DataFrame:
    if "packet_interval" in df.columns:
        return df

    clusters_per_sweep = df["cluster"].nunique()
    intervals = list(intervals)
    if len(intervals) * clusters_per_sweep != len(df):
        raise ValueError(
            "Impossible d'inférer les intervalles : le nombre de lignes du CSV ne correspond pas"
            " au produit (nombre d'intervalles) x (nombre de clusters)."
        )

    expanded: list[float] = []
    for interval in intervals:
        expanded.extend([interval] * clusters_per_sweep)
    df = df.copy()
    df["packet_interval"] = expanded
    return df


def _plot_load_sensitivity(df: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    clusters = sorted(df["cluster"].unique())
    for cluster_id in clusters:
        subset = df[df["cluster"] == cluster_id].sort_values("packet_interval")
        ax.plot(
            subset["packet_interval"] / 60.0,
            subset["der"],
            marker="o",
            label=f"Cluster {cluster_id}",
        )

    ax.set_xlabel("Intervalle entre paquets (minutes)")
    ax.set_ylabel("Data Extraction Rate (DER)")
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    ax.set_title("Sensibilité à la charge (UCB1)")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300)
    print(f"Figure enregistrée dans {output}")


def main() -> None:
    args = parse_args()
    df = _load_load_csv(args.csv)
    df = _attach_intervals(df, args.packet_intervals)
    _plot_load_sensitivity(df, args.output)


if __name__ == "__main__":
    main()
