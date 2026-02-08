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
    place_adaptive_legend,
    assert_legend_present,
    MetricStatus,
    ensure_network_size,
    filter_mixra_opt_fallback,
    filter_rows_by_network_sizes,
    is_constant_metric,
    legend_handles_for_algos_snir,
    load_step1_aggregated,
    metric_values,
    plot_metric_by_snir,
    render_metric_status,
    select_received_metric_key,
    save_figure,
    suptitle_y_from_top,
    warn_if_insufficient_network_sizes,
)
from plot_defaults import resolve_ieee_figsize


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    network_sizes = sorted(df["network_size"].unique())
    warn_if_insufficient_network_sizes(network_sizes)
    metric_key = select_received_metric_key(rows, metric_key)
    clusters = sorted(
        {row["cluster"] for row in rows if row.get("cluster") not in (None, "all")}
    )
    if not clusters:
        clusters = list(DEFAULT_CONFIG.qos.clusters)
    fig, axes = plt.subplots(
        1,
        len(clusters),
        sharey=True,
        figsize=resolve_ieee_figsize(len(clusters)),
    )
    if len(clusters) == 1:
        axes = [axes]
    metric_state = is_constant_metric(metric_values(rows, metric_key))
    if metric_state is not MetricStatus.OK:
        render_metric_status(
            fig,
            axes,
            metric_state,
            legend_handles=legend_handles_for_algos_snir(),
        )
        return fig
    for ax, cluster in zip(axes, clusters, strict=False):
        cluster_rows = [row for row in rows if row.get("cluster") == cluster]
        plot_metric_by_snir(ax, cluster_rows, metric_key)
        ax.set_xlabel("Network size (nodes)")
        ax.set_ylabel("PDR (prob.)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        ax.set_xticks(network_sizes)
    handles, labels = legend_handles_for_algos_snir()
    place_adaptive_legend(
        fig,
        axes[0],
        preferred_loc="right",
        handles=handles if handles else None,
        labels=labels if handles else None,
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
    rows = [row for row in rows if row.get("cluster") != "all"]
    rows, _ = filter_rows_by_network_sizes(rows, args.network_sizes)
    rows = filter_mixra_opt_fallback(rows)

    fig = _plot_metric(rows, "pdr_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S6", use_tight=False)
    assert_legend_present(fig, "plot_S6")
    plt.close(fig)


if __name__ == "__main__":
    main()
