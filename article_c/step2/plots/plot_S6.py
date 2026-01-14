"""Trace la figure S6 (récompense moyenne vs densité par cluster)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    apply_plot_style,
    load_step2_aggregated,
    place_legend,
    plot_metric_by_snir,
    save_figure,
)


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    clusters = sorted(
        {row["cluster"] for row in rows if row.get("cluster") not in (None, "all")}
    )
    if not clusters:
        clusters = list(DEFAULT_CONFIG.qos.clusters)
    fig, axes = plt.subplots(1, len(clusters), figsize=(5 * len(clusters), 4), sharey=True)
    if len(clusters) == 1:
        axes = [axes]
    for ax, cluster in zip(axes, clusters, strict=False):
        cluster_rows = [row for row in rows if row.get("cluster") == cluster]
        plot_metric_by_snir(ax, cluster_rows, metric_key)
        ax.set_xlabel("Density")
        ax.set_title(f"Cluster {cluster}")
    axes[0].set_ylabel("Mean Reward")
    place_legend(axes[-1])
    fig.suptitle("Step 2 - Mean Reward by Cluster (SNIR on/off)")
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = load_step2_aggregated(results_path)
    rows = [row for row in rows if row.get("cluster") != "all"]

    fig = _plot_metric(rows, "reward_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S6")
    plt.close(fig)


if __name__ == "__main__":
    main()
