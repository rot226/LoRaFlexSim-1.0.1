"""Trace la figure RL7 (récompense moyenne fenêtre vs densité)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import (
    algo_label,
    apply_plot_style,
    filter_cluster,
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


def _canonical_algo(algo: str) -> str | None:
    return ALGO_ALIASES.get(algo)


def _label_for_algo(algo: str) -> str:
    canonical = _canonical_algo(algo)
    if canonical is None:
        return algo
    return algo_label(canonical)


def _filter_algorithms(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    filtered = [
        row for row in rows if _canonical_algo(str(row.get("algo", ""))) in TARGET_ALGOS
    ]
    return filtered or rows


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    fig, ax = plt.subplots()
    algorithms = sorted({row["algo"] for row in rows})
    for algo in algorithms:
        points = {
            row["density"]: row[metric_key]
            for row in rows
            if row.get("algo") == algo
        }
        densities = sorted(points)
        values = [points[density] for density in densities]
        ax.plot(densities, values, marker="o", label=_label_for_algo(str(algo)))
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Mean Window Reward")
    ax.set_title("Step 2 - Mean Window Reward vs Network size (number of nodes)")
    place_legend(ax)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = filter_cluster(load_step2_aggregated(results_path), "all")
    rows = [row for row in rows if row.get("snir_mode") == "snir_on"]
    rows = _filter_algorithms(rows)

    fig = _plot_metric(rows, "reward_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL7_reward_vs_density")
    plt.close(fig)


if __name__ == "__main__":
    main()
