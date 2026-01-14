"""Trace la figure RL6 (outage par cluster vs densité)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.config import DEFAULT_CONFIG
from article_c.common.plot_helpers import (
    algo_label,
    apply_plot_style,
    load_step2_aggregated,
    place_legend,
    save_figure,
)

ALGO_ALIASES = {
    "adr": "adr",
    "ADR": "adr",
    "mixra_h": "mixra_h",
    "MixRA-H": "mixra_h",
    "mixra_opt": "mixra_opt",
    "MixRA-Opt": "mixra_opt",
    "ucb1_sf": "ucb1_sf",
    "UCB1-SF": "ucb1_sf",
}
TARGET_ALGOS = {"adr", "mixra_h", "mixra_opt", "ucb1_sf"}


def _cluster_labels(clusters: list[str]) -> dict[str, str]:
    return {cluster: f"C{idx + 1}" for idx, cluster in enumerate(clusters)}


def _canonical_algo(algo: str) -> str | None:
    return ALGO_ALIASES.get(algo)


def _label_for_algo(algo: str) -> str:
    canonical = _canonical_algo(algo)
    if canonical is None:
        return algo
    return algo_label(canonical)


def _outage_probability(row: dict[str, object]) -> float:
    success_rate = float(row.get("success_rate_mean") or 0.0)
    outage = 1.0 - success_rate
    return max(0.0, min(1.0, outage))


def _with_outage(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    enriched: list[dict[str, object]] = []
    for row in rows:
        enriched_row = dict(row)
        enriched_row["outage_prob"] = _outage_probability(row)
        enriched.append(enriched_row)
    return enriched


def _filter_algorithms(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    filtered = [
        row for row in rows if _canonical_algo(str(row.get("algo", ""))) in TARGET_ALGOS
    ]
    return filtered or rows


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

    algorithms = sorted({row["algo"] for row in rows})
    for ax, cluster in zip(axes, clusters, strict=False):
        cluster_rows = [row for row in rows if row.get("cluster") == cluster]
        for algo in algorithms:
            points = {
                row["density"]: row[metric_key]
                for row in cluster_rows
                if row.get("algo") == algo
            }
            if not points:
                continue
            densities = sorted(points)
            values = [points[density] for density in densities]
            ax.plot(densities, values, marker="o", label=_label_for_algo(str(algo)))
        ax.set_xlabel("Density")
        ax.set_title(f"Cluster {cluster_labels.get(cluster, cluster)}")
    axes[0].set_ylabel("Outage Probability")
    place_legend(axes[-1])
    fig.suptitle("Step 2 - Outage Probability by Cluster (SNIR on)")
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = load_step2_aggregated(results_path)
    rows = [row for row in rows if row.get("cluster") != "all"]
    rows = [row for row in rows if row.get("snir_mode") == "snir_on"]
    rows = _filter_algorithms(rows)
    rows = _with_outage(rows)

    fig = _plot_metric(rows, "outage_prob")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL6_cluster_outage_vs_density")
    plt.close(fig)


if __name__ == "__main__":
    main()
