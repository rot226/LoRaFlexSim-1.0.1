"""Trace la figure S7 (probabilité d'outage vs densité par cluster)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    apply_plot_style,
    load_step1_aggregated,
    place_legend,
    plot_metric_by_snir,
    save_figure,
)


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

    fig, axes = plt.subplots(1, len(clusters), figsize=(5 * len(clusters), 4), sharey=True)
    if len(clusters) == 1:
        axes = [axes]

    for ax, cluster in zip(axes, clusters, strict=False):
        cluster_rows = [row for row in rows if row.get("cluster") == cluster]
        plot_metric_by_snir(ax, cluster_rows, metric_key)
        ax.set_xlabel("Network size (number of nodes)")
        ax.set_title(f"Cluster {cluster_labels.get(cluster, cluster)}")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    axes[0].set_ylabel("Outage probability")
    place_legend(axes[-1])
    fig.suptitle("Step 1 - Outage probability by Cluster (SNIR on/off)")
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
    save_figure(fig, output_dir, "plot_S7_cluster_outage_vs_density", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
