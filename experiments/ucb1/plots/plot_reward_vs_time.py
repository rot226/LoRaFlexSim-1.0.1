"""Trace la récompense par temps/décision avec moyenne glissante."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import apply_figure_layout
from plot_defaults import resolve_ieee_figsize
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
    parser = argparse.ArgumentParser(description="Récompense vs temps (moyenne glissante).")
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
        help="Taille de la fenêtre pour la moyenne glissante (en décisions).",
    )
    return parser.parse_args()


def _ensure_columns(df: pd.DataFrame, required: Iterable[str], path: Path) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Colonnes manquantes dans {path}: {', '.join(missing)}")


def _save_plot(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    apply_figure_layout(fig, tight_layout=True)
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
    _ensure_columns(df, ["reward", "cluster", "decision_idx"], args.decision_csv)
    time_col = _resolve_time_column(df)

    df = filter_top_groups(df, ["cluster"], max_groups=3)
    clusters = sorted(df["cluster"].dropna().unique())
    if not clusters:
        raise ValueError("Aucun cluster détecté dans le CSV de décisions.")

    cluster_colors = {cluster: PALETTE[index % len(PALETTE)] for index, cluster in enumerate(clusters)}
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(max(1, len(clusters))))

    for cluster in clusters:
        subset = df[df["cluster"] == cluster].sort_values(time_col)
        if subset.empty:
            continue
        reward_series = pd.to_numeric(subset["reward"], errors="coerce")
        rolling_mean = reward_series.rolling(args.rolling_window, min_periods=1).mean()
        ax.plot(
            subset[time_col],
            rolling_mean,
            label=f"Cluster {int(cluster)}",
            color=cluster_colors[cluster],
            linewidth=2,
        )
        ax.scatter(
            subset[time_col],
            reward_series,
            color=cluster_colors[cluster],
            alpha=0.15,
            s=10,
        )

    ax.set_title("Récompense vs temps")
    ax.set_xlabel("Temps (s)" if time_col == "time_s" else "Indice de décision")
    ax.set_ylabel("Récompense")
    ax.grid(True, linestyle=":", alpha=0.5)
    if ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=8, ncol=2)

    _save_plot(fig, args.output_dir, "ucb1_reward_vs_time.png")


if __name__ == "__main__":
    main()
