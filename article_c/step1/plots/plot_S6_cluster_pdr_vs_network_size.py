"""Trace la figure S6 (PDR vs taille du réseau par cluster)."""

from __future__ import annotations

import math
import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
from matplotlib.lines import Line2D
import pandas as pd

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    SNIR_LABELS,
    SNIR_LINESTYLES,
    SNIR_MODES,
    LEGEND_ABOVE_TIGHT_LAYOUT_TOP,
    apply_plot_style,
    apply_figure_layout,
    filter_mixra_opt_fallback,
    is_constant_metric,
    legend_margins,
    load_step1_aggregated,
    metric_values,
    filter_rows_by_network_sizes,
    render_constant_metric,
    save_figure,
)
from article_c.common.plotting_style import LEGEND_STYLE
from article_c.step1.plots.plot_utils import configure_figure

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
    cluster_targets = _cluster_targets(clusters)

    fig, axes = plt.subplots(1, len(SNIR_MODES), sharey=True)
    apply_figure_layout(fig, figsize=(10, 6))
    if len(SNIR_MODES) == 1:
        axes = [axes]

    if is_constant_metric(metric_values(rows, metric_key)):
        render_constant_metric(fig, axes, legend_handles=None)
        configure_figure(
            fig,
            axes,
            "Step 1 - PDR by Cluster (network size)",
            legend_loc="above",
        )
        return fig

    cluster_handles: list[plt.Line2D] = []
    legend_labels: list[str] = []
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
            label = f"Cluster {cluster_labels.get(cluster, cluster)}{target_label}"
            show_label = snir_mode == SNIR_MODES[0]
            (line,) = ax.plot(
                network_sizes,
                values,
                marker="o",
                linestyle=SNIR_LINESTYLES[snir_mode],
                label=label if show_label else "_nolegend_",
            )
            if show_label:
                cluster_handles.append(line)
                legend_labels.append(label)
        ax.set_title(SNIR_LABELS[snir_mode])
        ax.set_xlabel("Network size (number of nodes)")
        ax.set_ylabel("Packet Delivery Ratio")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xticks(network_sizes)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))

    snir_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=SNIR_LINESTYLES[snir_mode],
            marker=None,
            label=SNIR_LABELS[snir_mode],
        )
        for snir_mode in SNIR_MODES
    ]
    cluster_legend = fig.legend(
        cluster_handles,
        legend_labels,
        title="Clusters",
        **{
            **LEGEND_STYLE,
            "bbox_to_anchor": (0.5, 1.14),
            "ncol": 3,
        },
    )
    fig.add_artist(cluster_legend)
    snir_labels = [handle.get_label() for handle in snir_handles]
    fig.legend(
        snir_handles,
        snir_labels,
        title="SNIR",
        **{
            **LEGEND_STYLE,
            "bbox_to_anchor": (0.5, 1.06),
            "ncol": min(len(snir_labels), 3),
        },
    )
    configure_figure(
        fig,
        axes,
        "Step 1 - PDR by Cluster (network size)",
        legend_loc="above",
    )
    layout_margins = legend_margins("above")
    apply_figure_layout(
        fig,
        margins={
            **layout_margins,
            "top": max(0.7, layout_margins.get("top", 0.0)),
        },
        tight_layout={
            "rect": (0, 0, 1, max(0.8, LEGEND_ABOVE_TIGHT_LAYOUT_TOP)),
        },
        legend_rows=2,
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
    plt.close(fig)


if __name__ == "__main__":
    main()
