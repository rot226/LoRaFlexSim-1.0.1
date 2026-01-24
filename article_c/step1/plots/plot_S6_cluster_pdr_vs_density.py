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
    algo_label,
    apply_plot_style,
    apply_figure_layout,
    ensure_network_size,
    filter_mixra_opt_fallback,
    filter_rows_by_network_sizes,
    load_step1_aggregated,
    save_figure,
)
from article_c.step1.plots.plot_utils import configure_figure


def _cluster_labels(clusters: list[str]) -> dict[str, str]:
    cluster_label_map = {
        "gold": "C1",
        "silver": "C2",
        "bronze": "C3",
    }
    return {cluster: cluster_label_map.get(cluster, cluster) for cluster in clusters}


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

    def _algo_key(row: dict[str, object]) -> tuple[str, bool]:
        algo_value = str(row.get("algo", ""))
        fallback = bool(row.get("mixra_opt_fallback")) if algo_value == "mixra_opt" else False
        return algo_value, fallback

    algorithms = sorted({_algo_key(row) for row in rows})

    fig, axes = plt.subplots(
        len(algorithms),
        len(clusters),
        sharex=True,
        sharey=True,
    )
    apply_figure_layout(
        fig,
        figsize=(4.2 * len(clusters), 3.4 * len(algorithms) + 2),
    )
    if len(algorithms) == 1 and len(clusters) == 1:
        axes = [[axes]]
    elif len(algorithms) == 1:
        axes = [axes]
    elif len(clusters) == 1:
        axes = [[ax] for ax in axes]

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
                ax.plot(
                    network_sizes,
                    values,
                    marker="o",
                    linestyle=SNIR_LINESTYLES[snir_mode],
                    label=SNIR_LABELS[snir_mode],
                )
            if algo_idx == 0:
                ax.set_title(f"Cluster {cluster_labels.get(cluster, cluster)}")
            if cluster_idx == 0:
                ax.set_ylabel(f"{algo_label(algo, fallback)}\nPacket Delivery Ratio")
            if algo_idx == len(algorithms) - 1:
                ax.set_xlabel("Network size (number of nodes)")
            ax.set_xticks(network_sizes)
            ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=3,
            frameon=False,
        )
        apply_figure_layout(fig, bbox_to_anchor=(0.5, 1.02))
    configure_figure(
        fig,
        axes,
        "Step 1 - PDR by Cluster (SNIR on/off, per algorithm)",
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
    rows = [row for row in rows if row.get("cluster") != "all"]
    rows, _ = filter_rows_by_network_sizes(rows, args.network_sizes)
    rows = filter_mixra_opt_fallback(rows)

    fig = _plot_metric(rows, "pdr_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S6_cluster_pdr_vs_density", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
