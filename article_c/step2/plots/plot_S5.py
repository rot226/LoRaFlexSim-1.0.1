"""Trace la figure S5 (PDR vs densité pour SNIR on/off)."""

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

SNIR_LABELS = {
    "snir_on": "SNIR on",
    "snir_off": "SNIR off",
}


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    fig, ax = plt.subplots()
    algorithms = sorted({row["algo"] for row in rows})
    for snir_mode in ("snir_on", "snir_off"):
        for algo in algorithms:
            points = {
                row["density"]: row[metric_key]
                for row in rows
                if row["algo"] == algo and row["snir_mode"] == snir_mode
            }
            if not points:
                continue
            densities = sorted(points)
            values = [points[density] for density in densities]
            label = f"{ALGO_LABELS.get(algo, algo)} ({SNIR_LABELS[snir_mode]})"
            ax.plot(densities, values, marker="o", label=label)
    ax.set_xlabel("Density")
    ax.set_ylabel("Packet Delivery Ratio")
    ax.set_title("Step 1 - Packet Delivery Ratio by SNIR Mode")
    place_legend(ax)
    return fig


def main() -> None:
    apply_plot_style()
    article_dir = Path(__file__).resolve().parents[2]
    results_path = article_dir / "step1" / "results" / "aggregated_results.csv"
    rows = load_step1_aggregated(results_path)

    fig = _plot_metric(rows, "pdr_mean")
    output_dir = article_dir / "step2" / "plots" / "output"
    save_figure(fig, output_dir, "plot_S5")
    plt.close(fig)


if __name__ == "__main__":
    main()
