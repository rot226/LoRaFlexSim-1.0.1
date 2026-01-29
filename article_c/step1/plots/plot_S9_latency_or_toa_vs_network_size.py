"""Trace la figure S9 (latence/ToA vs taille du réseau, SNIR on/off)."""

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
    plot_metric_by_snir,
    render_constant_metric,
    save_figure,
)
from article_c.step1.plots.plot_utils import configure_figure
from plot_defaults import resolve_ieee_figsize


METRIC_KEY = "mean_toa_s"
METRIC_LABEL = "Mean ToA (s)"


def _plot_metric(rows: list[dict[str, object]], metric_key: str, y_label: str) -> plt.Figure:
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    series_count = (
        df[["algo", "snir_mode"]].dropna().drop_duplicates().shape[0]
        if {"algo", "snir_mode"}.issubset(df.columns)
        else len(df.dropna().drop_duplicates())
    )
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(series_count))
    apply_figure_layout(fig, figsize=tuple(fig.get_size_inches()))
    network_sizes = sorted(df["network_size"].unique())
    if len(network_sizes) < 2:
        warnings.warn("Moins de deux tailles de réseau disponibles.", stacklevel=2)
    if is_constant_metric(metric_values(rows, metric_key)):
        render_constant_metric(fig, ax, legend_handles=None)
        configure_figure(
            fig,
            ax,
            "Step 1 - ToA/Latency vs Network Size (SNIR on/off)",
            legend_loc="above",
        )
        return fig
    plot_metric_by_snir(ax, rows, metric_key)
    ax.set_xticks(network_sizes)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel(y_label)
    configure_figure(
        fig,
        ax,
        "Step 1 - ToA/Latency vs Network Size (SNIR on/off)",
        legend_loc="above",
    )
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

    if not any(METRIC_KEY in row for row in rows):
        raise ValueError("La métrique mean_toa_s est absente des résultats.")
    fig = _plot_metric(rows, METRIC_KEY, METRIC_LABEL)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S9", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
