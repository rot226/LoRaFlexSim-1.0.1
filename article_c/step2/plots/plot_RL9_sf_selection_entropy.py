"""Trace la figure RL9 (entropie de sélection des SF vs rounds)."""

from __future__ import annotations

from math import log2
import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import (
    apply_plot_style,
    ensure_network_size,
    filter_rows_by_network_sizes,
    load_step2_aggregated,
    load_step2_selection_probs,
    save_figure,
)


def _normalized_network_sizes(network_sizes: list[int] | None) -> list[int] | None:
    if not network_sizes or len(network_sizes) < 2:
        return None
    return network_sizes


def _has_invalid_network_sizes(network_sizes: list[float]) -> bool:
    if any(float(size) == 0.0 for size in network_sizes):
        print(
            "ERREUR: taille de réseau invalide détectée (0.0). "
            "Aucune figure ne sera tracée."
        )
        return True
    return False


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


def main(network_sizes: list[int] | None = None, argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    args = parser.parse_args(argv)
    if network_sizes is None:
        network_sizes = args.network_sizes
    if network_sizes is not None and _has_invalid_network_sizes(network_sizes):
        return
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "rl5_selection_prob.csv"
    rows = load_step2_selection_probs(results_path)
    aggregated_results_path = step_dir / "results" / "aggregated_results.csv"
    size_rows = load_step2_aggregated(aggregated_results_path)
    ensure_network_size(size_rows)
    network_sizes_filter = _normalized_network_sizes(network_sizes)
    size_rows, _ = filter_rows_by_network_sizes(size_rows, network_sizes_filter)
    if network_sizes_filter is None:
        df = pd.DataFrame(size_rows)
        network_sizes = sorted(df["network_size"].unique())
    else:
        network_sizes = network_sizes_filter
    if _has_invalid_network_sizes(network_sizes):
        return
    if len(network_sizes) < 2:
        warnings.warn(
            f"Moins de deux tailles de réseau disponibles: {network_sizes}.",
            stacklevel=2,
        )

    fig = _plot_entropy(rows)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL9_sf_selection_entropy", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
