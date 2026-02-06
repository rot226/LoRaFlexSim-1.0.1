"""Trace la figure RL5 (probabilité de sélection par SF)."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import pandas as pd

from article_c.common.plot_helpers import (
    apply_plot_style,
    apply_figure_layout,
    assert_legend_present,
    MetricStatus,
    fallback_legend_handles,
    filter_rows_by_network_sizes,
    is_constant_metric,
    load_step2_aggregated,
    load_step2_selection_probs,
    normalize_network_size_rows,
    create_right_legend_layout,
    render_metric_status,
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


def _plot_selection(
    rows: list[dict[str, object]],
    network_sizes: list[int],
) -> plt.Figure:
    network_sizes = sorted(network_sizes)
    sfs = sorted({row["sf"] for row in rows})
    series_count = len(sfs) * len(network_sizes) if sfs and network_sizes else None
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(series_count))
    apply_figure_layout(fig)
    selection_values = [
        float(row.get("selection_prob"))
        for row in rows
        if isinstance(row.get("selection_prob"), (int, float))
    ]
    metric_state = is_constant_metric(selection_values)
    if metric_state is not MetricStatus.OK:
        render_metric_status(
            fig,
            ax,
            metric_state,
            legend_loc="right",
            show_fallback_legend=True,
            legend_handles=fallback_legend_handles(),
        )
        return fig
    for network_size in network_sizes:
        size_rows = [row for row in rows if row["network_size"] == network_size]
        for sf in sfs:
            points = {
                row["round"]: row["selection_prob"]
                for row in size_rows
                if row["sf"] == sf
            }
            rounds = sorted(points)
            values = [points[round_id] for round_id in rounds]
            label = f"SF {sf} (N={network_size})"
            ax.plot(rounds, values, marker="o", label=label)
    ax.set_xlabel("Round (index)")
    ax.set_ylabel("Selection prob. (prob.)")
    create_right_legend_layout(fig, ax)
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
    fig = _plot_selection(rows, network_sizes)
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL5", use_tight=False)
    assert_legend_present(fig, "plot_RL5")
    plt.close(fig)


if __name__ == "__main__":
    main()
