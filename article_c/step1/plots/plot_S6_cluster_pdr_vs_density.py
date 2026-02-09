"""Trace la figure S6 (PDR vs densité par cluster, algorithmes séparés)."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import pandas as pd

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    SNIR_LABELS,
    SNIR_LINESTYLES,
    SNIR_MODES,
    MetricStatus,
    algo_label,
    apply_plot_style,
    place_adaptive_legend,
    assert_legend_present,
    ensure_network_size,
    filter_mixra_opt_fallback,
    filter_rows_by_network_sizes,
    is_constant_metric,
    legend_handles_for_algos_snir,
    load_step1_aggregated,
    metric_values,
    render_metric_status,
    save_figure,
    suptitle_y_from_top,
    warn_if_insufficient_network_sizes,
)
from plot_defaults import resolve_ieee_figsize

MAX_ALGOS_PER_FIG = 3


def _cluster_labels(clusters: list[str]) -> dict[str, str]:
    cluster_label_map = {
        "gold": "C1",
        "silver": "C2",
        "bronze": "C3",
    }
    return {cluster: cluster_label_map.get(cluster, cluster) for cluster in clusters}


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


def _plot_metric_page(
    rows: list[dict[str, object]],
    metric_key: str,
    *,
    clusters: list[str],
    cluster_labels: dict[str, str],
    network_sizes: list[int],
    algorithms: list[tuple[str, bool]],
    metric_state: MetricStatus,
    title_suffix: str = "",
) -> plt.Figure:
    base_width, base_height = resolve_ieee_figsize(len(clusters))
    figsize = (base_width, base_height * max(1, len(algorithms)))
    fig, axes = plt.subplots(
        max(1, len(algorithms)),
        len(clusters),
        figsize=figsize,
        sharex=True,
        sharey=True,
    )
    if len(algorithms) == 1 and len(clusters) == 1:
        axes = [[axes]]
    elif len(algorithms) == 1:
        axes = [axes]
    elif len(clusters) == 1:
        axes = [[ax] for ax in axes]

    title = "Step 1 - PDR by Cluster (SNIR on/off, per algorithm)"
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

    for algo_idx, (algo, fallback) in enumerate(algorithms):
        algo_rows = [row for row in rows if _algo_key(row) == (algo, fallback)]
        for cluster_idx, cluster in enumerate(clusters):
            ax = axes[algo_idx][cluster_idx]
            cluster_rows = [
                row for row in algo_rows if row.get("cluster") == cluster
            ]
            for snir_mode in SNIR_MODES:
                points = {
                    int(row["network_size"]): row[metric_key]
                    for row in cluster_rows
                    if row["snir_mode"] == snir_mode
                }
                if not points:
                    continue
                values = [points.get(size, float("nan")) for size in network_sizes]
                label = (
                    f"{algo_label(algo, fallback)} / "
                    f"Cluster {cluster_labels.get(cluster, cluster)} "
                    f"({SNIR_LABELS[snir_mode]})"
                )
                ax.plot(
                    network_sizes,
                    values,
                    marker="o",
                    linestyle=SNIR_LINESTYLES[snir_mode],
                    label=label,
                )
            ax.set_xlabel("Network size (nodes)")
            if cluster_idx == 0:
                ax.set_ylabel(f"{algo_label(algo, fallback)}\nPDR (prob.)")
            else:
                ax.set_ylabel("PDR (prob.)")
            ax.set_xticks(network_sizes)
            ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))

    handles: list[plt.Line2D] = []
    labels: list[str] = []
    for row_axes in axes:
        for ax in row_axes:
            series_handles, series_labels = ax.get_legend_handles_labels()
            handles.extend(series_handles)
            labels.extend(series_labels)
    place_adaptive_legend(
        fig,
        axes[0][0],
        preferred_loc="right",
        handles=handles if handles else None,
        labels=labels if handles else None,
    )
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
        fig = _plot_metric_page(
            rows,
            metric_key,
            clusters=clusters,
            cluster_labels=cluster_labels,
            network_sizes=network_sizes,
            algorithms=algo_chunk,
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

    figures = _plot_metric(rows, "pdr_mean")
    output_dir = step_dir / "plots" / "output"
    for index, fig in enumerate(figures, start=1):
        stem = "plot_S6_cluster_pdr_vs_density"
        if len(figures) > 1:
            stem = f"{stem}_page{index}"
        save_figure(fig, output_dir, stem, use_tight=False)
        assert_legend_present(fig, stem)
        plt.close(fig)


if __name__ == "__main__":
    main()
