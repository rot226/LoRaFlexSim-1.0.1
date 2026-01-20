"""Trace la figure S6 (PDR vs densité par cluster)."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import pandas as pd

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    apply_plot_style,
    ensure_network_size,
    filter_rows_by_network_sizes,
    load_step1_aggregated,
    place_legend,
    plot_metric_by_snir,
    save_figure,
)


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    network_sizes = sorted(df["network_size"].unique())
    if len(network_sizes) < 2:
        warnings.warn("Moins de deux tailles de réseau disponibles.", stacklevel=2)
    clusters = sorted(
        {row["cluster"] for row in rows if row.get("cluster") not in (None, "all")}
    )
    if not clusters:
        clusters = list(DEFAULT_CONFIG.qos.clusters)
    fig, axes = plt.subplots(1, len(clusters), figsize=(5 * len(clusters), 4), sharey=True)
    if len(clusters) == 1:
        axes = [axes]
    for ax, cluster in zip(axes, clusters, strict=False):
        cluster_rows = [row for row in rows if row.get("cluster") == cluster]
        plot_metric_by_snir(ax, cluster_rows, metric_key)
        ax.set_xlabel("Network size (number of nodes)")
        ax.set_title(f"Cluster {cluster}")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        ax.set_xticks(network_sizes)
    axes[0].set_ylabel("Packet Delivery Ratio")
    place_legend(axes[-1])
    fig.suptitle("Step 1 - Packet Delivery Ratio by Cluster (SNIR on/off)")
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
    rows = load_step1_aggregated(results_path)
    rows = [row for row in rows if row.get("cluster") != "all"]
    rows, _ = filter_rows_by_network_sizes(rows, args.network_sizes)

    fig = _plot_metric(rows, "pdr_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S6", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
