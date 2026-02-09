"""Outils de tracé pour les figures SNIR statiques (S1–S8)."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def _validate_columns(df: pd.DataFrame, required: Iterable[str]) -> None:
    missing = [column for column in required if column not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Colonnes manquantes dans le CSV: {missing_str}")


def _prepare_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby(["cluster_id", "algorithm"], as_index=False)
        .agg(
            pdr_achieved=("pdr_achieved", "mean"),
            pdr_target=("pdr_target", "mean"),
        )
        .sort_values(["cluster_id", "algorithm"])
    )
    return grouped


def _plot_cluster_axes(
    axes: plt.Axes,
    data: pd.DataFrame,
    algorithms: list[str],
    colors: list[str],
) -> None:
    x_positions = list(range(len(algorithms)))
    for idx, algo in enumerate(algorithms):
        algo_data = data[data["algorithm"] == algo]
        if algo_data.empty:
            continue
        axes.bar(
            x_positions[idx],
            float(algo_data["pdr_achieved"].iloc[0]),
            color=colors[idx],
            label=algo,
        )
    if not data.empty:
        target = float(data["pdr_target"].iloc[0])
        axes.axhline(
            target,
            color="red",
            linestyle="--",
            linewidth=1.5,
            label="PDR cible",
        )
    axes.set_xticks(x_positions, algorithms, rotation=30, ha="right")
    axes.set_ylim(0.0, 1.0)
    axes.set_xlabel("Algorithme")
    axes.set_ylabel("PDR atteinte")


def plot_figure(
    *,
    figure_id: str,
    csv_path: Path,
    output_dir: Path,
) -> None:
    _ensure_output_dir(output_dir)
    df = pd.read_csv(csv_path)
    _validate_columns(
        df,
        ["figure", "cluster_id", "algorithm", "pdr_target", "pdr_achieved"],
    )
    df = df[df["figure"] == figure_id]
    if df.empty:
        raise ValueError(f"Aucune donnée pour la figure {figure_id} dans {csv_path}.")

    aggregates = _prepare_aggregates(df)
    cluster_ids = sorted(aggregates["cluster_id"].unique())
    algorithms = sorted(aggregates["algorithm"].unique())
    color_map = plt.get_cmap("tab10")
    colors = [color_map(idx) for idx in range(len(algorithms))]

    fig, axes = plt.subplots(
        1,
        len(cluster_ids),
        figsize=(4.5 * len(cluster_ids), 4.2),
        sharey=True,
        constrained_layout=True,
    )
    if len(cluster_ids) == 1:
        axes = [axes]

    for ax, cluster_id in zip(axes, cluster_ids):
        cluster_data = aggregates[aggregates["cluster_id"] == cluster_id]
        _plot_cluster_axes(ax, cluster_data, algorithms, colors)
        ax.set_title(f"Cluster {cluster_id}")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(len(labels), 4),
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(f"Figure {figure_id} – SNIR statique")

    png_path = output_dir / f"{figure_id}.png"
    pdf_path = output_dir / f"{figure_id}.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)


def build_argument_parser(figure_id: str) -> argparse.ArgumentParser:
    default_csv = _default_csv_path(figure_id)
    default_output = _default_output_dir()
    parser = argparse.ArgumentParser(
        description=(
            f"Génère la figure {figure_id} (SNIR statique) depuis un CSV de résultats."
        )
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=default_csv,
        help=f"Chemin du CSV (défaut: {default_csv}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Répertoire de sortie (défaut: {default_output}).",
    )
    return parser


def _default_output_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "figures"


def _default_csv_path(figure_id: str) -> Path:
    return Path(__file__).resolve().parents[1] / "data" / f"{figure_id}.csv"


def main_for_figure(figure_id: str) -> None:
    parser = build_argument_parser(figure_id)
    args = parser.parse_args()
    plot_figure(
        figure_id=figure_id,
        csv_path=args.csv,
        output_dir=args.output_dir,
    )

