"""Trace la figure S2 (ToA moyen vs densité, SNIR on/off)."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from article_c.common.plot_helpers import (
    apply_plot_style,
    apply_figure_layout,
    ensure_network_size,
    filter_rows_by_network_sizes,
    filter_cluster,
    filter_mixra_opt_fallback,
    is_constant_metric,
    load_step1_aggregated,
    metric_values,
    place_legend,
    plot_metric_by_snir,
    render_constant_metric,
    save_figure,
)
from plot_defaults import DEFAULT_FIGSIZE_MULTI


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=DEFAULT_FIGSIZE_MULTI)
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    network_sizes = sorted(df["network_size"].unique())
    if len(network_sizes) < 2:
        warnings.warn("Moins de deux tailles de réseau disponibles.", stacklevel=2)
    if is_constant_metric(metric_values(rows, metric_key)):
        render_constant_metric(fig, ax, legend_handles=None)
        ax.set_title("Step 1 - Mean Time on Air vs Network Size (SNIR on/off)")
        return fig
    plot_metric_by_snir(ax, rows, metric_key)
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Mean Time on Air (s)")
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xticks(network_sizes)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.set_title("Step 1 - Mean Time on Air vs Network Size (SNIR on/off)")
    place_legend(ax, legend_loc="above")
    legend = ax.get_legend()
    if legend is not None:
        legend.set_title("Mean Time on Air (s)")
    metric_values = pd.to_numeric(df[metric_key], errors="coerce").dropna()
    if not metric_values.empty:
        y_min = metric_values.min()
        y_max = metric_values.max()
        padding = max((y_max - y_min) * 0.05, 0.01)
        y_min = 0.0 if y_min >= 0 else y_min - padding
        y_max = y_max + padding
        ax.set_ylim(y_min, y_max)
    apply_figure_layout(fig, margins={"left": 0.16})
    return fig


def main(argv: list[str] | None = None, allow_sample: bool = True) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    args = parser.parse_args(argv)
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = load_step1_aggregated(results_path, allow_sample=allow_sample)
    if not rows:
        warnings.warn("CSV Step1 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows = filter_cluster(rows, "all")
    rows, _ = filter_rows_by_network_sizes(rows, args.network_sizes)
    rows = filter_mixra_opt_fallback(rows)

    fig = _plot_metric(rows, "mean_toa_s")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S2", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
