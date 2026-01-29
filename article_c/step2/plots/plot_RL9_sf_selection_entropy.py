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
    apply_figure_layout,
    filter_rows_by_network_sizes,
    is_constant_metric,
    load_step2_aggregated,
    load_step2_selection_probs,
    legend_margins,
    normalize_network_size_rows,
    place_legend,
    render_constant_metric,
    save_figure,
)
from plot_defaults import resolve_ieee_figsize


def _normalized_network_sizes(network_sizes: list[int] | None) -> list[int] | None:
    if not network_sizes:
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


def _title_suffix(network_sizes: list[int]) -> str:
    if len(network_sizes) == 1:
        return " (taille unique)"
    return ""


def _entropy(probabilities: list[float]) -> float:
    return -sum(p * log2(p) for p in probabilities if p > 0.0)


def _normalized_probs(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0.0:
        return values
    return [value / total for value in values]


def _plot_entropy(
    rows: list[dict[str, object]],
    network_sizes: list[int],
) -> plt.Figure:
    network_sizes = sorted(network_sizes)
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(len(network_sizes)))
    width, height = fig.get_size_inches()
    apply_figure_layout(
        fig, figsize=(width, height + 2), margins=legend_margins("top")
    )
    all_entropy_values: list[float] = []
    for network_size in network_sizes:
        size_rows = [row for row in rows if row["network_size"] == network_size]
        rounds = sorted({row["round"] for row in size_rows})
        entropy_values: list[float] = []
        for round_id in rounds:
            probs = [
                float(row["selection_prob"])
                for row in size_rows
                if row["round"] == round_id
            ]
            entropy_values.append(_entropy(_normalized_probs(probs)))
        all_entropy_values.extend(entropy_values)
        ax.plot(rounds, entropy_values, marker="o", label=f"N={network_size}")
    if is_constant_metric(all_entropy_values):
        render_constant_metric(fig, ax, legend_handles=None)
        ax.set_title(
            "Step 2 - SF Selection Entropy vs Round"
            f"{_title_suffix(network_sizes)}"
        )
        return fig
    ax.set_xlabel("Round")
    ax.set_ylabel("Selection Entropy (bits)")
    ax.set_title(
        "Step 2 - SF Selection Entropy vs Round"
        f"{_title_suffix(network_sizes)}"
    )
    if network_sizes:
        place_legend(ax, legend_loc="above")
    return fig


def main(
    network_sizes: list[int] | None = None,
    argv: list[str] | None = None,
    allow_sample: bool = True,
) -> None:
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
    if not allow_sample and not results_path.exists():
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows = load_step2_selection_probs(results_path)
    aggregated_results_path = step_dir / "results" / "aggregated_results.csv"
    size_rows = load_step2_aggregated(
        aggregated_results_path,
        allow_sample=allow_sample,
    )
    if not size_rows:
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    normalize_network_size_rows(size_rows)
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

    rows, _ = filter_rows_by_network_sizes(rows, network_sizes)
    fig = _plot_entropy(rows, network_sizes)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL9_sf_selection_entropy", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
