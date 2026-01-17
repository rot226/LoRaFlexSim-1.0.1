"""Trace la figure RL4 (énergie normalisée moyenne vs densité)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker

from article_c.common.plot_helpers import (
    apply_plot_style,
    filter_cluster,
    load_step2_aggregated,
    place_legend,
    save_figure,
)


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    fig, ax = plt.subplots()
    all_densities = sorted({int(row["density"]) for row in rows})
    algorithms = sorted({row["algo"] for row in rows})
    for algo in algorithms:
        points = {
            int(row["density"]): row[metric_key]
            for row in rows
            if row["algo"] == algo
        }
        densities = sorted(points)
        values = [points[density] for density in densities]
        ax.plot(densities, values, marker="o", label=algo)
    ax.set_xticks(all_densities)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Energy per successful bit (J/bit)")
    ax.set_title("Step 2 - Normalized Energy vs Network size (number of nodes)")
    place_legend(ax)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = filter_cluster(load_step2_aggregated(results_path), "all")
    rows = [row for row in rows if row["snir_mode"] == "snir_on"]

    fig = _plot_metric(rows, "energy_norm_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL4", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
