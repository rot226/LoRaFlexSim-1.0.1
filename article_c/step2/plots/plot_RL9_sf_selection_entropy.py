"""Trace la figure RL9 (entropie de sélection des SF vs rounds)."""

from __future__ import annotations

from math import log2
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import (
    apply_plot_style,
    ensure_network_size,
    load_step2_aggregated,
    load_step2_selection_probs,
    save_figure,
)


def _entropy(probabilities: list[float]) -> float:
    return -sum(p * log2(p) for p in probabilities if p > 0.0)


def _plot_entropy(rows: list[dict[str, object]]) -> plt.Figure:
    fig, ax = plt.subplots()
    rounds = sorted({row["round"] for row in rows})
    entropy_values: list[float] = []
    for round_id in rounds:
        probs = [
            float(row["selection_prob"])
            for row in rows
            if row["round"] == round_id
        ]
        entropy_values.append(_entropy(probs))
    ax.plot(rounds, entropy_values, marker="o")
    ax.set_xlabel("Round")
    ax.set_ylabel("Selection Entropy (bits)")
    ax.set_title("Step 2 - SF Selection Entropy vs Round")
    return fig


def main() -> None:
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "rl5_selection_prob.csv"
    rows = load_step2_selection_probs(results_path)
    aggregated_results_path = step_dir / "results" / "aggregated_results.csv"
    size_rows = load_step2_aggregated(aggregated_results_path)
    ensure_network_size(size_rows)
    df = pd.DataFrame(size_rows)
    network_sizes = sorted(df["network_size"].unique())
    if len(network_sizes) < 2:
        warnings.warn("Moins de deux tailles de réseau disponibles.", stacklevel=2)

    fig = _plot_entropy(rows)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL9_sf_selection_entropy", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
