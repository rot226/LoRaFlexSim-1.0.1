"""Trace la figure RL1 (récompense médiane par cluster vs densité)."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
from matplotlib import ticker as mticker
import pandas as pd

from article_c.common.plot_helpers import (
    apply_plot_style,
    apply_figure_layout,
    assert_legend_present,
    MetricStatus,
    ensure_network_size,
    filter_rows_by_network_sizes,
    filter_cluster,
    is_constant_metric,
    load_step2_aggregated,
    metric_values,
    normalize_network_size_rows,
    place_adaptive_legend,
    legend_handles_for_algos_snir,
    plot_metric_by_algo,
    render_metric_status,
    save_figure,
)
from plot_defaults import resolve_ieee_figsize

TARGET_CLUSTER = "gold"


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


def _plot_metric(
    rows: list[dict[str, object]],
    metric_key: str,
    network_sizes: list[int] | None,
    cluster_label: str,
) -> plt.Figure | None:
    df = pd.DataFrame(rows)
    if "algo" in df.columns:
        algo_col = "algo"
    elif "algorithm" in df.columns:
        algo_col = "algorithm"
    else:
        algo_col = None
    series_count = len(df[algo_col].dropna().unique()) if algo_col else None
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(series_count))
    width, height = fig.get_size_inches()
    apply_figure_layout(fig, figsize=(width, height))
    ensure_network_size(rows)
    if network_sizes is None:
        network_sizes = sorted(df["network_size"].unique())
    if _has_invalid_network_sizes(network_sizes):
        return None
    if len(network_sizes) < 2:
        warnings.warn(
            f"Moins de deux tailles de réseau disponibles: {network_sizes}.",
            stacklevel=2,
        )
    metric_state = is_constant_metric(metric_values(rows, metric_key))
    if metric_state is not MetricStatus.OK:
        render_metric_status(
            fig,
            ax,
            metric_state,
            show_fallback_legend=True,
            legend_handles=legend_handles_for_algos_snir(["snir_on"]),
        )
        return fig
    plot_metric_by_algo(ax, rows, metric_key, network_sizes)
    ax.set_xticks(network_sizes)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.set_xlabel("Network size (nodes)")
    ax.set_ylabel(f"Reward {cluster_label} (a.u., median, p10–p90)")
    place_adaptive_legend(fig, ax, preferred_loc="right")
    return fig


def _select_cluster_rows(
    rows: list[dict[str, object]],
    cluster: str,
) -> tuple[list[dict[str, object]], str]:
    available_clusters = {row.get("cluster") for row in rows}
    if cluster in available_clusters:
        return filter_cluster(rows, cluster), cluster
    if "all" in available_clusters:
        return filter_cluster(rows, "all"), "all"
    return rows, "all"


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
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = load_step2_aggregated(results_path, allow_sample=allow_sample)
    if not rows:
        warnings.warn("CSV Step2 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows, cluster_label = _select_cluster_rows(rows, TARGET_CLUSTER)
    rows = [row for row in rows if row["snir_mode"] == "snir_on"]
    normalize_network_size_rows(rows)
    network_sizes_filter = _normalized_network_sizes(network_sizes)
    rows, _ = filter_rows_by_network_sizes(rows, network_sizes_filter)

    fig = _plot_metric(rows, "reward_mean", network_sizes_filter, cluster_label)
    if fig is None:
        return
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_RL1", use_tight=False)
    assert_legend_present(fig, "plot_RL1")
    plt.close(fig)


if __name__ == "__main__":
    main()
