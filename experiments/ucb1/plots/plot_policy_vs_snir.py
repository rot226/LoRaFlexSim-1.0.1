"""Analyse des choix SF/TX Power en fonction du SNIR (binned)."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import apply_figure_layout
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
    parser = argparse.ArgumentParser(description="Choix de SF/puissance vs SNIR.")
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
        "--bin-width",
        type=float,
        default=2.0,
        help="Largeur des bins SNIR (dB).",
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


def main() -> None:
    apply_ieee_style()
    args = parse_args()
    df = pd.read_csv(args.decision_csv)
    _ensure_columns(df, ["snir_db", "sf", "tx_power"], args.decision_csv)

    df = df.dropna(subset=["snir_db"])
    if df.empty:
        raise ValueError("Aucune mesure SNIR disponible dans le CSV.")

    min_snir = df["snir_db"].min()
    max_snir = df["snir_db"].max()
    bins = list(pd.interval_range(
        start=min_snir,
        end=max_snir + args.bin_width,
        freq=args.bin_width,
        closed="left",
    ))
    df["snir_bin"] = pd.cut(df["snir_db"], bins=bins)

    if "policy" in df.columns:
        df = filter_top_groups(df, ["policy"], max_groups=3)
        policies = sorted(df["policy"].dropna().unique())
    else:
        policies = ["all"]
    if "policy" not in df.columns:
        df["policy"] = "all"

    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7.5, 8.0), sharex=True)
    metrics = [("sf", "SF moyen"), ("tx_power", "Puissance d'émission (dBm)")]

    for idx, (metric, ylabel) in enumerate(metrics):
        ax = axes[idx]
        for policy_index, policy in enumerate(policies):
            subset = df[df["policy"] == policy]
            summary = (
                subset.groupby("snir_bin", observed=True)[metric]
                .mean()
                .reset_index()
                .dropna()
            )
            if summary.empty:
                continue
            centers = [interval.mid for interval in summary["snir_bin"]]
            ax.plot(
                centers,
                summary[metric],
                label=policy,
                color=PALETTE[policy_index % len(PALETTE)],
                marker="o",
                markersize=4,
            )
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle=":", alpha=0.5)
        if idx == 0 and ax.get_legend_handles_labels()[1]:
            ax.legend(fontsize=8, ncol=2)

    axes[-1].set_xlabel("SNIR (dB)")
    fig.suptitle("Politique vs SNIR")
    _save_plot(fig, args.output_dir, "ucb1_policy_vs_snir.png")


if __name__ == "__main__":
    main()
