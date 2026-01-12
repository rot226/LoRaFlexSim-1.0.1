"""Variance temporelle des décisions SF/TX Power (rolling variance)."""
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
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stabilité des décisions (variance glissante).")
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
        "--rolling-window",
        type=int,
        default=50,
        help="Taille de la fenêtre glissante (en décisions).",
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


def _resolve_time_column(df: pd.DataFrame) -> str:
    if "time_s" in df.columns:
        return "time_s"
    return "decision_idx"


def main() -> None:
    apply_ieee_style()
    args = parse_args()
    df = pd.read_csv(args.decision_csv)
    _ensure_columns(df, ["sf", "tx_power", "cluster", "decision_idx"], args.decision_csv)
    time_col = _resolve_time_column(df)

    df = filter_top_groups(df, ["cluster"], max_groups=3)
    clusters = sorted(df["cluster"].dropna().unique())
    if not clusters:
        raise ValueError("Aucun cluster détecté dans le CSV de décisions.")

    cluster_colors = {cluster: PALETTE[index % len(PALETTE)] for index, cluster in enumerate(clusters)}
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.5, 8.0), sharex=True)
    metrics = [("sf", "Variance glissante SF"), ("tx_power", "Variance glissante TX Power")]

    for idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[idx]
        for cluster in clusters:
            subset = df[df["cluster"] == cluster].sort_values(time_col)
            if subset.empty:
                continue
            series = pd.to_numeric(subset[metric], errors="coerce")
            rolling_var = series.rolling(args.rolling_window, min_periods=2).var()
            ax.plot(
                subset[time_col],
                rolling_var,
                label=f"Cluster {int(cluster)}",
                color=cluster_colors[cluster],
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx == 0 and ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8, ncol=2)

    fig.suptitle("Stabilité des décisions")
    axes[-1].set_xlabel("Temps (s)" if time_col == "time_s" else "Indice de décision")
    _save_plot(fig, args.output_dir, "ucb1_decision_stability.png")


if __name__ == "__main__":
    main()
