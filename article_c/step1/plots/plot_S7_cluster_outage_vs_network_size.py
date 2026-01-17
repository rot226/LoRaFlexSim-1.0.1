"""Trace la figure S7 (probabilité d'outage vs taille du réseau par cluster)."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

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

    raw_densities = sorted({row["density"] for row in rows})
    densities = [_density_to_nodes(density) for density in raw_densities]

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
            values = [points.get(density, float("nan")) for density in raw_densities]
            target = cluster_targets.get(cluster)
            target_label = f" (target {target:.2f})" if target is not None else ""
            label = f"Cluster {cluster_labels.get(cluster, cluster)}{target_label}"
            ax.plot(
                densities,
                values,
                marker="o",
                linestyle=SNIR_LINESTYLES[snir_mode],
                label=label,
            )
        ax.set_title(SNIR_LABELS[snir_mode])
        ax.set_xlabel("Network size (number of nodes)")
        ax.set_ylabel("Outage probability")
        ax.set_ylim(0.0, 1.0)
        ax.grid(True, linestyle=":", alpha=0.4)
        ax.set_xticks(densities)
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))

    place_legend(axes[-1])
    fig.suptitle("Step 1 - Outage probability by Cluster (network size)")
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = load_step1_aggregated(results_path)
    rows = [row for row in rows if row.get("cluster") != "all"]
    rows = _with_outage(rows)

    fig = _plot_metric(rows, "outage_prob")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S7_cluster_outage_vs_network_size", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
