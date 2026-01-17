"""Trace la figure RL8 (distribution des récompenses par algorithme)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import (
    algo_label,
    apply_plot_style,
    filter_cluster,
    load_step2_aggregated,
    save_figure,
)


def _plot_distribution(rows: list[dict[str, object]]) -> plt.Figure:
    fig, ax = plt.subplots()
    algorithms = sorted({row["algo"] for row in rows})
    rewards_by_algo = [
        [row["reward_mean"] for row in rows if row["algo"] == algo]
        for algo in algorithms
    ]
    positions = list(range(1, len(algorithms) + 1))
    ax.violinplot(rewards_by_algo, positions=positions, showmedians=True)
    ax.boxplot(
        rewards_by_algo,
        positions=positions,
        widths=0.2,
        patch_artist=True,
        boxprops={"facecolor": "white", "alpha": 0.6},
        medianprops={"color": "black"},
    )
    ax.set_xticks(positions)
    ax.set_xticklabels([algo_label(str(algo)) for algo in algorithms])
    ax.set_xlabel("Algorithm")
    ax.set_ylabel("Mean Reward")
    ax.set_title("Step 2 - Reward Distribution by Algorithm")
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = filter_cluster(load_step2_aggregated(results_path), "all")
    rows = [row for row in rows if row.get("snir_mode") == "snir_on"]

    fig = _plot_distribution(rows)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL8_reward_distribution", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
