"""Trace la figure S3 (réceptions moyennes vs densité, SNIR on/off)."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import (
    apply_plot_style,
    ensure_network_size,
    filter_rows_by_network_sizes,
    filter_cluster,
    load_step1_aggregated,
    place_legend,
    plot_metric_by_snir,
    save_figure,
)


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    fig, ax = plt.subplots()
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    network_sizes = sorted(df["network_size"].unique())
    if len(network_sizes) < 2:
        warnings.warn("Moins de deux tailles de réseau disponibles.", stacklevel=2)
    plot_metric_by_snir(ax, rows, metric_key)
    ax.set_xticks(network_sizes)
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Received Frames (mean)")
    ax.set_title("Step 1 - Received Frames vs Network size (number of nodes) (SNIR on/off)")
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
    rows = filter_cluster(load_step1_aggregated(results_path), "all")
    rows, _ = filter_rows_by_network_sizes(rows, args.network_sizes)

    fig = _plot_metric(rows, "received_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S3", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
