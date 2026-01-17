"""Trace la figure S6 (PDR vs densité par cluster, algorithmes séparés)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    SNIR_LABELS,
    SNIR_LINESTYLES,
    SNIR_MODES,
    algo_label,
    apply_plot_style,
    load_step1_aggregated,
    place_legend,
    save_figure,
)


def _cluster_labels(clusters: list[str]) -> dict[str, str]:
    return {cluster: f"C{idx + 1}" for idx, cluster in enumerate(clusters)}


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
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
    algorithms = sorted({row["algo"] for row in rows})

    fig, axes = plt.subplots(
        len(algorithms),
        len(clusters),
        figsize=(4.2 * len(clusters), 3.4 * len(algorithms)),
        sharex=True,
        sharey=True,
    )
    if len(algorithms) == 1 and len(clusters) == 1:
        axes = [[axes]]
    elif len(algorithms) == 1:
        axes = [axes]
    elif len(clusters) == 1:
        axes = [[ax] for ax in axes]

    for algo_idx, algo in enumerate(algorithms):
        algo_rows = [row for row in rows if row["algo"] == algo]
        for cluster_idx, cluster in enumerate(clusters):
            ax = axes[algo_idx][cluster_idx]
            cluster_rows = [
                row for row in algo_rows if row.get("cluster") == cluster
            ]
            densities = sorted({int(row["density"]) for row in cluster_rows})
            for snir_mode in SNIR_MODES:
                points = {
                    int(row["density"]): row[metric_key]
                    for row in cluster_rows
                    if row["snir_mode"] == snir_mode
                }
                if not points:
                    continue
                values = [points.get(density, float("nan")) for density in densities]
                ax.plot(
                    densities,
                    values,
                    marker="o",
                    linestyle=SNIR_LINESTYLES[snir_mode],
                    label=SNIR_LABELS[snir_mode],
                )
            if algo_idx == 0:
                ax.set_title(f"Cluster {cluster_labels.get(cluster, cluster)}")
            if cluster_idx == 0:
                ax.set_ylabel(f"{algo_label(algo)}\nPacket Delivery Ratio")
            if algo_idx == len(algorithms) - 1:
                ax.set_xlabel("Network size (number of nodes)")
            ax.set_xticks(densities)
            ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))

    place_legend(axes[-1][-1])
    fig.suptitle("Step 1 - PDR by Cluster (SNIR on/off, per algorithm)")
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = load_step1_aggregated(results_path)
    rows = [row for row in rows if row.get("cluster") != "all"]

    fig = _plot_metric(rows, "pdr_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S6_cluster_pdr_vs_density", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
