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
    add_figure_legend,
    assert_legend_present,
    clear_axis_legends,
    MetricStatus,
    ensure_network_size,
    filter_mixra_opt_fallback,
    filter_rows_by_network_sizes,
    fallback_legend_handles,
    is_constant_metric,
    legend_handles_for_algos_snir,
    load_step1_aggregated,
    metric_values,
    plot_metric_by_snir,
    render_metric_status,
    save_figure,
    suptitle_y_from_top,
    warn_if_insufficient_network_sizes,
)
from plot_defaults import resolve_ieee_figsize

MAX_ALGOS_PER_FIG = 3


def _algo_key(row: dict[str, object]) -> tuple[str, bool]:
    algo_value = str(row.get("algo", ""))
    fallback = bool(row.get("mixra_opt_fallback")) if algo_value == "mixra_opt" else False
    return algo_value, fallback


def _chunk_algorithms(
    algorithms: list[tuple[str, bool]],
    max_per_fig: int,
) -> list[list[tuple[str, bool]]]:
    if not algorithms:
        return []
    max_per_fig = max(1, max_per_fig)
    return [
        algorithms[start : start + max_per_fig]
        for start in range(0, len(algorithms), max_per_fig)
    ]


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


def _plot_metric_page(
    rows: list[dict[str, object]],
    metric_key: str,
    *,
    clusters: list[str],
    cluster_labels: dict[str, str],
    network_sizes: list[int],
    metric_state: MetricStatus,
    title_suffix: str = "",
) -> plt.Figure:
    fig, axes = plt.subplots(1, len(clusters), sharey=True)
    apply_figure_layout(fig, figsize=resolve_ieee_figsize(len(clusters)))
    if len(clusters) == 1:
        axes = [axes]

    title = "Step 1 - Outage probability by Cluster (SNIR on/off)"
    if title_suffix:
        title = f"{title} ({title_suffix})"

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
        ax.set_ylabel("Outage (prob.)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
        ax.set_xticks(network_sizes)
    handles, labels = legend_handles_for_algos_snir()
    clear_axis_legends(axes)
    if not handles:
        handles, labels = fallback_legend_handles()
    add_figure_legend(fig, handles, labels, legend_loc="right")
    return fig


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> list[plt.Figure]:
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    network_sizes = sorted(df["network_size"].unique())
    warn_if_insufficient_network_sizes(network_sizes)
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

    metric_state = is_constant_metric(metric_values(rows, metric_key))
    algorithms = sorted({_algo_key(row) for row in rows})
    algo_chunks = _chunk_algorithms(algorithms, MAX_ALGOS_PER_FIG)
    total_pages = max(1, len(algo_chunks))
    figures: list[plt.Figure] = []
    for page_index, algo_chunk in enumerate(algo_chunks or [[]], start=1):
        title_suffix = f"page {page_index}/{total_pages}" if total_pages > 1 else ""
        if algo_chunk:
            filtered_rows = [
                row
                for row in rows
                if _algo_key(row) in set(algo_chunk)
            ]
        else:
            filtered_rows = rows
        fig = _plot_metric_page(
            filtered_rows,
            metric_key,
            clusters=clusters,
            cluster_labels=cluster_labels,
            network_sizes=network_sizes,
            metric_state=metric_state,
            title_suffix=title_suffix,
        )
        figures.append(fig)
    return figures


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

    figures = _plot_metric(rows, "outage_prob")
    output_dir = step_dir / "plots" / "output"
    for index, fig in enumerate(figures, start=1):
        stem = "plot_S7_cluster_outage_vs_density"
        if len(figures) > 1:
            stem = f"{stem}_page{index}"
        save_figure(fig, output_dir, stem, use_tight=False)
        assert_legend_present(fig, stem)
        plt.close(fig)


if __name__ == "__main__":
    main()
