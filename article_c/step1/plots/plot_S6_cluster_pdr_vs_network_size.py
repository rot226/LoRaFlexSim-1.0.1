"""Trace la figure S6 (PDR vs taille du réseau par cluster)."""

from __future__ import annotations

import math
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
    apply_plot_style,
    apply_figure_layout,
    add_figure_legend,
    assert_legend_present,
    fallback_legend_handles,
    filter_mixra_opt_fallback,
    is_constant_metric,
    legend_margins,
    legend_handles_for_algos_snir,
    load_step1_aggregated,
    metric_values,
    filter_rows_by_network_sizes,
    render_metric_status,
    save_figure,
    suptitle_y_from_top,
    warn_if_insufficient_network_sizes,
)
from plot_defaults import resolve_ieee_figsize

PDR_TARGETS = (0.90, 0.80, 0.70)


def _cluster_labels(clusters: list[str]) -> dict[str, str]:
    return {cluster: f"C{idx + 1}" for idx, cluster in enumerate(clusters)}


def _cluster_targets(clusters: list[str]) -> dict[str, float | None]:
    targets: dict[str, float | None] = {}
    for idx, cluster in enumerate(clusters):
        target = PDR_TARGETS[idx] if idx < len(PDR_TARGETS) else None
        targets[cluster] = target
    return targets


def _density_to_nodes(density: float) -> int:
    area_km2 = math.pi * (DEFAULT_CONFIG.scenario.radius_m / 1000.0) ** 2
    return max(1, int(round(density * area_km2)))


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    for row in rows:
        if "network_size" not in row and "density" in row:
            row["network_size"] = _density_to_nodes(float(row["density"]))
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
    cluster_targets = _cluster_targets(clusters)

    fig, axes = plt.subplots(1, len(SNIR_MODES), sharey=True)
    apply_figure_layout(fig, figsize=resolve_ieee_figsize(len(SNIR_MODES)))
    if len(SNIR_MODES) == 1:
        axes = [axes]

    metric_state = is_constant_metric(metric_values(rows, metric_key))
    if metric_state is not MetricStatus.OK:
        render_metric_status(
            fig,
            axes,
            metric_state,
            legend_handles=legend_handles_for_algos_snir(),
        )
        fig.suptitle(
            "Step 1 - PDR by Cluster (network size)",
            y=suptitle_y_from_top(fig),
        )
        return fig

    cluster_handles: list[plt.Line2D] = []
    legend_labels: list[str] = []
    seen_labels: set[str] = set()
    for ax, snir_mode in zip(axes, SNIR_MODES, strict=False):
        snir_rows = [row for row in rows if row["snir_mode"] == snir_mode]
        for cluster in clusters:
            points = {
                int(row["network_size"]): row[metric_key]
                for row in snir_rows
                if row.get("cluster") == cluster
            }
            if not points:
                continue
            values = [points.get(size, float("nan")) for size in network_sizes]
            target = cluster_targets.get(cluster)
            target_label = f" (target {target:.2f})" if target is not None else ""
            label = (
                f"Cluster {cluster_labels.get(cluster, cluster)}"
                f"{target_label} ({SNIR_LABELS[snir_mode]})"
            )
            (line,) = ax.plot(
                network_sizes,
                values,
                marker="o",
                linestyle=SNIR_LINESTYLES[snir_mode],
                label=label,
            )
            if label not in seen_labels:
                cluster_handles.append(line)
                legend_labels.append(label)
                seen_labels.add(label)
        ax.set_title(SNIR_LABELS[snir_mode])
        ax.set_xlabel("Network size (number of nodes)")
        ax.set_ylabel("Packet Delivery Ratio")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xticks(network_sizes)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))

    for ax in axes:
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
    if not cluster_handles:
        cluster_handles, legend_labels = fallback_legend_handles()
    legend_rows = add_figure_legend(
        fig,
        cluster_handles,
        legend_labels,
        legend_loc="right",
    )
    if fig.legends:
        fig.legends[-1].set_title("Clusters")
    layout_margins = legend_margins("above", legend_rows=max(1, legend_rows), fig=fig)
    apply_figure_layout(
        fig,
        margins={
            **layout_margins,
            "top": max(0.7, layout_margins.get("top", 0.0)),
        },
        legend_rows=max(1, legend_rows),
    )
    fig.suptitle(
        "Step 1 - PDR by Cluster (network size)",
        y=suptitle_y_from_top(fig),
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
    save_figure(fig, output_dir, "plot_S6_cluster_pdr_vs_network_size", use_tight=False)
    assert_legend_present(fig, "plot_S6_cluster_pdr_vs_network_size")
    plt.close(fig)


if __name__ == "__main__":
    main()
