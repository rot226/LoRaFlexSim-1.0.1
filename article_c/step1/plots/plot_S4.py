"""Trace la figure S4 (trames envoyées vs densité, SNIR on/off)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
    network_sizes = sorted({int(row["density"]) for row in rows})
    plot_metric_by_snir(ax, rows, metric_key)
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Sent Frames (mean)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.set_xticks(network_sizes)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.set_title("Step 1 - Sent Frames vs Network size (number of nodes) (SNIR on/off)")
    place_legend(ax)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = filter_cluster(load_step1_aggregated(results_path), "all")

    fig = _plot_metric(rows, "sent_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S4", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
