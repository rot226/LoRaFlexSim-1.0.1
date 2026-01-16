"""Trace la figure S6 (PDR vs taille du réseau par cluster)."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    SNIR_LABELS,
    SNIR_LINESTYLES,
    SNIR_MODES,
    apply_plot_style,
    load_step1_aggregated,
    place_legend,
    save_figure,
)

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

    fig, axes = plt.subplots(1, len(SNIR_MODES), figsize=(10, 4), sharey=True)
    if len(SNIR_MODES) == 1:
        axes = [axes]

    densities = sorted({row["density"] for row in rows})
    network_sizes = [_density_to_nodes(density) for density in densities]

    for ax, snir_mode in zip(axes, SNIR_MODES, strict=False):
        snir_rows = [row for row in rows if row["snir_mode"] == snir_mode]
        for cluster in clusters:
            points = {
                row["density"]: row[metric_key]
                for row in snir_rows
                if row.get("cluster") == cluster
            }
            if not points:
                continue
            values = [points.get(density, float("nan")) for density in densities]
            target = cluster_targets.get(cluster)
            target_label = f" (target {target:.2f})" if target is not None else ""
            label = f"Cluster {cluster_labels.get(cluster, cluster)}{target_label}"
            ax.plot(
                network_sizes,
                values,
                marker="o",
                linestyle=SNIR_LINESTYLES[snir_mode],
                label=label,
            )
        ax.set_title(SNIR_LABELS[snir_mode])
        ax.set_xlabel("Network size (number of nodes)")
        ax.set_ylabel("Packet Delivery Ratio")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, linestyle=":", alpha=0.4)

    place_legend(axes[-1])
    fig.suptitle("Step 1 - PDR by Cluster (network size)")
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = load_step1_aggregated(results_path)
    rows = [row for row in rows if row.get("cluster") != "all"]

    fig = _plot_metric(rows, "pdr_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S6_cluster_pdr_vs_network_size", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
