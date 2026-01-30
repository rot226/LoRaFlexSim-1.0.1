"""Trace la figure S7 (probabilité d'outage vs densité par cluster)."""

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
    apply_figure_layout,
    add_global_legend,
    MetricStatus,
    ensure_network_size,
    filter_mixra_opt_fallback,
    filter_rows_by_network_sizes,
    is_constant_metric,
    legend_margins,
    load_step1_aggregated,
    metric_values,
    plot_metric_by_snir,
    render_metric_status,
    save_figure,
)
from article_c.step1.plots.plot_utils import configure_figure

LAYOUT_MARGINS = legend_margins("above")
LAYOUT_RECT = (0, 0, 1, 0.78)


def _cluster_labels(clusters: list[str]) -> dict[str, str]:
    return {cluster: f"C{idx + 1}" for idx, cluster in enumerate(clusters)}


def _outage_probability(row: dict[str, object]) -> float:
    sent = float(row.get("sent_mean") or 0.0)
    received = float(row.get("received_mean") or 0.0)
    if sent > 0:
        pdr = received / sent
    else:
        pdr = float(row.get("pdr_mean") or 0.0)
    return max(0.0, min(1.0, 1.0 - pdr))


def _with_outage(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        enriched_row = dict(row)
        enriched_row["outage_prob"] = _outage_probability(row)
        enriched.append(enriched_row)
    return enriched


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    network_sizes = sorted(df["network_size"].unique())
    if len(network_sizes) < 2:
        warnings.warn("Moins de deux tailles de réseau disponibles.", stacklevel=2)
    available_clusters = {
        row["cluster"] for row in rows if row.get("cluster") not in (None, "all")
    }
    clusters = [
        cluster
        for cluster in DEFAULT_CONFIG.qos.clusters
        if cluster in available_clusters
    ]
    if not clusters:
        clusters = sorted(available_clusters)
    cluster_labels = _cluster_labels(clusters)

    fig, axes = plt.subplots(1, len(clusters), sharey=True)
    apply_figure_layout(fig, figsize=(5 * len(clusters), 8.5))
    if len(clusters) == 1:
        axes = [axes]

    metric_state = is_constant_metric(metric_values(rows, metric_key))
    if metric_state is not MetricStatus.OK:
        render_metric_status(fig, axes, metric_state, legend_handles=None)
        configure_figure(
            fig,
            axes,
            "Step 1 - Outage probability by Cluster (SNIR on/off)",
            legend_loc="above",
        )
        apply_figure_layout(
            fig,
            tight_layout={"rect": LAYOUT_RECT},
            margins=LAYOUT_MARGINS,
        )
        return fig

    for ax, cluster in zip(axes, clusters, strict=False):
        cluster_rows = [row for row in rows if row.get("cluster") == cluster]
        plot_metric_by_snir(ax, cluster_rows, metric_key)
        ax.set_xlabel("Network size (number of nodes)")
        ax.set_title(f"Cluster {cluster_labels.get(cluster, cluster)}")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        ax.set_xticks(network_sizes)
    axes[0].set_ylabel("Outage probability")
    handles, labels = axes[0].get_legend_handles_labels()
    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if handles:
        add_global_legend(
            fig,
            axes[0],
            legend_loc="above",
            handles=handles,
            labels=labels,
        )
    configure_figure(
        fig,
        axes,
        "Step 1 - Outage probability by Cluster (SNIR on/off)",
        legend_loc="above",
    )
    apply_figure_layout(
        fig,
        tight_layout={"rect": LAYOUT_RECT},
        margins=LAYOUT_MARGINS,
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
    rows = _with_outage(rows)

    fig = _plot_metric(rows, "outage_prob")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S7_cluster_outage_vs_density", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
