"""Trace la figure RL1 (récompense moyenne vs densité)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import (
    apply_plot_style,
    load_step2_aggregated,
    place_legend,
    plot_metric_by_snir,
    save_figure,
)


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    fig, ax = plt.subplots()
    plot_metric_by_snir(ax, rows, metric_key)
    ax.set_xlabel("Density")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Step 2 - Mean Reward vs Density (SNIR on/off)")
    place_legend(ax)
    return fig


def main() -> None:
    apply_plot_style()
    article_dir = Path(__file__).resolve().parents[2]
    results_path = article_dir / "step2" / "results" / "aggregated_results.csv"
    rows = load_step2_aggregated(results_path)

    fig = _plot_metric(rows, "reward_mean")
    output_dir = article_dir / "step1" / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL1")
    plt.close(fig)


if __name__ == "__main__":
    main()
