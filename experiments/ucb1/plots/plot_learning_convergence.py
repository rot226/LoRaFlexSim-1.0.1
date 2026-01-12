"""Trace la convergence d'apprentissage (reward/PDR/throughput) par épisode."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from experiments.ucb1.plots.plot_style import apply_ieee_style, filter_top_groups

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_DECISION_CSV = Path(__file__).resolve().parents[1] / "ucb1_decision_log.csv"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "plots"
PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Courbes de convergence UCB1 par épisode.")
    parser.add_argument(
        "--decision-csv",
        type=Path,
        default=DEFAULT_DECISION_CSV,
        help="CSV des décisions (run_ucb1_load_sweep.py).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Répertoire de sortie pour les PNG.",
    )
    parser.add_argument(
        "--packet-interval",
        type=float,
        action="append",
        default=[],
        help="Filtre les intervalles de paquets (peut être répété).",
    )
    return parser.parse_args()


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {', '.join(missing)}")


def _save_plot(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def main() -> None:
    apply_ieee_style()
    args = parse_args()
    df = pd.read_csv(args.decision_csv)
    _ensure_columns(
        df,
        ["episode_idx", "reward", "pdr", "throughput", "cluster", "packet_interval_s"],
        args.decision_csv,
    )
    if args.packet_interval:
        df = df[df["packet_interval_s"].isin(args.packet_interval)]

    df = filter_top_groups(df, ["cluster"], max_groups=3)
    clusters = sorted(df["cluster"].dropna().unique())
    if not clusters:
        raise ValueError("Aucun cluster détecté dans le CSV de décisions.")

    grouped = (
        df.groupby(["cluster", "episode_idx"], as_index=False)[
            ["reward", "pdr", "throughput"]
        ]
        .mean()
        .sort_values("episode_idx")
    )

    cluster_colors = {cluster: PALETTE[index % len(PALETTE)] for index, cluster in enumerate(clusters)}
    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(7.5, 9.5), sharex=True)
    metrics = [
        ("reward", "Récompense moyenne"),
        ("pdr", "PDR cumulée"),
        ("throughput", "Débit instantané (bps)"),
    ]

    for idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[idx]
        for cluster in clusters:
            subset = grouped[grouped["cluster"] == cluster]
            if subset.empty:
                continue
            ax.plot(
                subset["episode_idx"],
                subset[metric],
                label=f"Cluster {int(cluster)}",
                color=cluster_colors[cluster],
                marker="o",
                markersize=3,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx == 0 and ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8, ncol=2)

    axes[-1].set_xlabel("Épisode (compte par nœud)")
    fig.suptitle("Convergence UCB1")
    _save_plot(fig, args.output_dir, "ucb1_learning_convergence.png")


if __name__ == "__main__":
    main()
