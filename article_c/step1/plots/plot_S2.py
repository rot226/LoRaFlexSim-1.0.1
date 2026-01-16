"""Trace la figure S2 (PDR vs densité, SNIR on/off)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import (
    apply_plot_style,
    filter_cluster,
    load_step1_aggregated,
    place_legend,
    plot_metric_by_snir,
    save_figure,
)


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    fig, ax = plt.subplots()
    plot_metric_by_snir(ax, rows, metric_key)
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Packet Delivery Ratio")
    ax.set_title("Step 1 - Packet Delivery Ratio (SNIR on/off)")
    place_legend(ax)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = filter_cluster(load_step1_aggregated(results_path), "all")

    fig = _plot_metric(rows, "pdr_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S2", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
