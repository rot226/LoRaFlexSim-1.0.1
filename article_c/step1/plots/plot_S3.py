"""Trace la figure S3 (réceptions médianes vs densité, SNIR on/off)."""

from __future__ import annotations

import argparse
from pathlib import Path
from statistics import mean
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

from article_c.common.plot_helpers import (
    apply_plot_style,
    apply_figure_layout,
    assert_legend_present,
    MetricStatus,
    ensure_network_size,
    filter_rows_by_network_sizes,
    filter_cluster,
    filter_mixra_opt_fallback,
    is_constant_metric,
    load_step1_aggregated,
    metric_values,
    plot_metric_by_snir,
    render_metric_status,
    select_received_metric_key,
    save_figure,
)
from article_c.step1.plots.plot_utils import configure_figure
from plot_defaults import resolve_ieee_figsize

_ALGO_SPECIFIC_TOL = 1e-6


def _warn_if_low_algo_variance(
    rows: list[dict[str, object]],
    metric_key: str,
    tolerance: float = _ALGO_SPECIFIC_TOL,
) -> None:
    grouped: dict[tuple[float, str], list[float]] = {}
    for row in rows:
        if metric_key not in row:
            continue
        network_size = float(row.get("network_size", 0.0))
        snir_mode = str(row.get("snir_mode", ""))
        value = row.get(metric_key)
        if not isinstance(value, (int, float)):
            continue
        grouped.setdefault((network_size, snir_mode), []).append(float(value))
    low_variance_groups = []
    for (network_size, snir_mode), values in grouped.items():
        if len(values) < 2:
            continue
        value_range = max(values) - min(values)
        scale = max(1.0, abs(mean(values)))
        if value_range <= tolerance * scale:
            low_variance_groups.append((network_size, snir_mode))
    if low_variance_groups:
        details = ", ".join(
            f"N={size:g} ({mode})" for size, mode in low_variance_groups
        )
        warnings.warn(
            "Variance inter-algo ≈ 0 pour received_mean. "
            f"Groupes concernés: {details}.",
            stacklevel=2,
        )


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    series_count = (
        df[["algo", "snir_mode"]].dropna().drop_duplicates().shape[0]
        if {"algo", "snir_mode"}.issubset(df.columns)
        else len(df.dropna().drop_duplicates())
    )
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize(series_count))
    apply_figure_layout(fig, figsize=tuple(fig.get_size_inches()))
    network_sizes = sorted(df["network_size"].unique())
    if len(network_sizes) < 2:
        warnings.warn("Moins de deux tailles de réseau disponibles.", stacklevel=2)
    metric_key = select_received_metric_key(rows, metric_key)
    metric_state = is_constant_metric(metric_values(rows, metric_key))
    if metric_state is not MetricStatus.OK:
        render_metric_status(fig, ax, metric_state, legend_handles=None)
        configure_figure(
            fig,
            ax,
            "Step 1 - Received Frames vs Network size (number of nodes) (SNIR on/off)",
            legend_loc="above",
        )
        return fig
    _warn_if_low_algo_variance(rows, metric_key)
    plot_metric_by_snir(ax, rows, metric_key)
    ax.set_xticks(network_sizes)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Received Frames (median, p10-p90)")
    configure_figure(
        fig,
        ax,
        "Step 1 - Received Frames vs Network size (number of nodes) (SNIR on/off)",
        legend_loc="above",
    )
    return fig


def main(argv: list[str] | None = None, allow_sample: bool = True) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    args = parser.parse_args(argv)
    apply_plot_style()
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = load_step1_aggregated(results_path, allow_sample=allow_sample)
    if not rows:
        warnings.warn("CSV Step1 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows = filter_cluster(rows, "all")
    rows, _ = filter_rows_by_network_sizes(rows, args.network_sizes)
    rows = filter_mixra_opt_fallback(rows)

    fig = _plot_metric(rows, "received_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S3", use_tight=False)
    assert_legend_present(fig, "plot_S3")
    plt.close(fig)


if __name__ == "__main__":
    main()
