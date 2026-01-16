"""Trace la figure RL5 (probabilité de sélection par SF)."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from article_c.common.plot_helpers import (
    apply_plot_style,
    load_step2_selection_probs,
    place_legend,
    save_figure,
)


def _plot_selection(rows: list[dict[str, object]]) -> plt.Figure:
    fig, ax = plt.subplots()
    sfs = sorted({row["sf"] for row in rows})
    for sf in sfs:
        points = {
            row["round"]: row["selection_prob"]
            for row in rows
            if row["sf"] == sf
        }
        rounds = sorted(points)
        values = [points[round_id] for round_id in rounds]
        ax.plot(rounds, values, marker="o", label=f"SF {sf}")
    ax.set_xlabel("Round")
    ax.set_ylabel("Selection Probability")
    ax.set_title("Step 2 - UCB1-SF Selection Probability")
    place_legend(ax)
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "rl5_selection_prob.csv"
    rows = load_step2_selection_probs(results_path)

    fig = _plot_selection(rows)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL5", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
