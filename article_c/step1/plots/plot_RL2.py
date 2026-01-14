"""Trace la figure RL2 (taux de succès moyen vs densité)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import (
    apply_plot_style,
    load_step2_aggregated,
    place_legend,
    save_figure,
)


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    fig, ax = plt.subplots()
    algorithms = sorted({row["algo"] for row in rows})
    for algo in algorithms:
        points = {
            row["density"]: row[metric_key]
            for row in rows
            if row["algo"] == algo
        }
        densities = sorted(points)
        values = [points[density] for density in densities]
        ax.plot(densities, values, marker="o", label=algo)
    ax.set_xlabel("Density")
    ax.set_ylabel("Mean Success Rate")
    ax.set_title("Step 2 - Success Rate vs Density")
    place_legend(ax)
    return fig


def main() -> None:
    apply_plot_style()
    article_dir = Path(__file__).resolve().parents[2]
    results_path = article_dir / "step2" / "results" / "aggregated_results.csv"
    rows = load_step2_aggregated(results_path)
    rows = [row for row in rows if row["snir_mode"] == "snir_on"]

    fig = _plot_metric(rows, "success_rate_mean")
    output_dir = article_dir / "step1" / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL2")
    plt.close(fig)


if __name__ == "__main__":
    main()
