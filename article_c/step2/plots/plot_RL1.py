"""Trace la figure RL1 (récompense moyenne vs densité)."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import pandas as pd

from article_c.common.plot_helpers import (
    apply_plot_style,
    ensure_network_size,
    filter_rows_by_network_sizes,
    filter_cluster,
    load_step2_aggregated,
    place_legend,
    save_figure,
)


def _normalized_network_sizes(network_sizes: list[int] | None) -> list[int] | None:
    if not network_sizes or len(network_sizes) < 2:
        return None
    return network_sizes


def _has_invalid_network_sizes(network_sizes: list[float]) -> bool:
    if any(float(size) == 0.0 for size in network_sizes):
        warnings.warn(
            "Erreur: taille de réseau invalide détectée (0.0). "
            "Aucune figure ne sera tracée.",
            stacklevel=2,
        )
        return True
    return False


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure | None:
    fig, ax = plt.subplots()
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    network_sizes = sorted(df["network_size"].unique())
    if _has_invalid_network_sizes(network_sizes):
        return None
    if len(network_sizes) < 2:
        warnings.warn(
            f"Moins de deux tailles de réseau disponibles: {network_sizes}.",
            stacklevel=2,
        )
    algorithms = sorted({row["algo"] for row in rows})
    for algo in algorithms:
        points = {
            int(row["network_size"]): row[metric_key]
            for row in rows
            if row["algo"] == algo
        }
        values = [points.get(size, float("nan")) for size in network_sizes]
        ax.plot(network_sizes, values, marker="o", label=algo)
    ax.set_xticks(network_sizes)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Step 2 - Mean Reward vs Network size (number of nodes)")
    place_legend(ax)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    args = parser.parse_args()
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = filter_cluster(load_step2_aggregated(results_path), "all")
    rows = [row for row in rows if row["snir_mode"] == "snir_on"]
    network_sizes_filter = _normalized_network_sizes(args.network_sizes)
    rows, _ = filter_rows_by_network_sizes(rows, network_sizes_filter)

    fig = _plot_metric(rows, "reward_mean")
    if fig is None:
        return
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL1", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
