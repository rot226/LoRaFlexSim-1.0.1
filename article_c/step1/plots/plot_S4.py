"""Trace la figure S4 (trames envoyées vs densité)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import (
    ALGO_LABELS,
    apply_plot_style,
    load_step1_aggregated,
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
        ax.plot(densities, values, marker="o", label=ALGO_LABELS.get(algo, algo))
    ax.set_xlabel("Density")
    ax.set_ylabel("Sent Frames (mean)")
    ax.set_title("Step 1 - Sent Frames vs Density (SNIR on)")
    place_legend(ax)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = load_step1_aggregated(results_path)
    rows = [row for row in rows if row["snir_mode"] == "snir_on"]

    fig = _plot_metric(rows, "sent_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S4")
    plt.close(fig)


if __name__ == "__main__":
    main()
