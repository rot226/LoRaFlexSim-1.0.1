"""Trace la figure S4 (trames envoyées saturées vs densité, SNIR on/off)."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from matplotlib.lines import Line2D

from article_c.common.plot_helpers import (
    ALGO_COLORS,
    ALGO_LABELS,
    ALGO_MARKERS,
    MetricStatus,
    SNIR_LABELS,
    SNIR_LINESTYLES,
    SNIR_MODES,
    apply_plot_style,
    apply_figure_layout,
    algo_label,
    assert_legend_present,
    clear_axis_legends,
    ensure_network_size,
    filter_rows_by_network_sizes,
    filter_cluster,
    filter_mixra_opt_fallback,
    is_constant_metric,
    fallback_legend_handles,
    legend_margins,
    load_step1_aggregated,
    metric_values,
    place_adaptive_legend,
    plot_metric_by_snir,
    render_metric_status,
    select_received_metric_key,
    resolve_percentile_keys,
    save_figure,
    warn_if_insufficient_network_sizes,
)
from plot_defaults import (
    WIDE_SERIES_THRESHOLD,
    WIDE_SERIES_WSPACE,
    resolve_ieee_figsize_for_series,
)


def _algo_sort_key(algo: object) -> int:
    normalized = str(algo).strip().lower().replace("-", "_").replace(" ", "_")
    order = list(ALGO_LABELS.keys())
    return order.index(normalized) if normalized in order else len(order)


def _normalize_algo(algo: object) -> str:
    return str(algo).strip().lower().replace("-", "_").replace(" ", "_")


def _add_summary_plot(
    ax: plt.Axes,
    rows: list[dict[str, object]],
    metric_key: str,
) -> tuple[list[Line2D], list[str]]:
    df = pd.DataFrame(rows)
    if df.empty:
        return [], []
    median_key, _, _ = resolve_percentile_keys(rows, metric_key)
    algos = sorted(df["algo"].dropna().unique(), key=_algo_sort_key)
    if not algos:
        return [], []
    offsets = {"snir_on": -0.15, "snir_off": 0.15}
    for snir_mode in SNIR_MODES:
        for index, algo in enumerate(algos):
            subset = df[(df["algo"] == algo) & (df["snir_mode"] == snir_mode)]
            if subset.empty or median_key not in subset:
                continue
            values = subset[median_key].dropna()
            if values.empty:
                continue
            median = float(values.median())
            vmin = float(values.min())
            vmax = float(values.max())
            normalized_algo = _normalize_algo(algo)
            color = ALGO_COLORS.get(normalized_algo, "#4c4c4c")
            marker = ALGO_MARKERS.get(normalized_algo, "o")
            label = f"{algo_label(str(algo))} ({SNIR_LABELS[snir_mode]})"
            ax.errorbar(
                index + offsets[snir_mode],
                median,
                yerr=[[median - vmin], [vmax - median]],
                fmt=marker,
                color=color,
                ecolor=color,
                linestyle=SNIR_LINESTYLES.get(snir_mode, "solid"),
                capsize=3,
                markersize=5,
                label=label,
            )
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([algo_label(str(algo)) for algo in algos])
    ax.set_ylabel("Tx frames (count)\n(median ± min/max)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    summary_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            color="#333333",
            markersize=5,
            label="médiane",
        ),
        Line2D(
            [],
            [],
            marker="_",
            linestyle="None",
            color="#333333",
            markersize=10,
            label="min",
        ),
        Line2D(
            [],
            [],
            marker="_",
            linestyle="None",
            color="#333333",
            markersize=10,
            label="max",
        ),
    ]
    return summary_handles, [handle.get_label() for handle in summary_handles]


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    series_count = (
        df[["algo", "snir_mode"]].dropna().drop_duplicates().shape[0]
        if {"algo", "snir_mode"}.issubset(df.columns)
        else len(df.dropna().drop_duplicates())
    )
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize_for_series(series_count))
    wide_series = series_count >= WIDE_SERIES_THRESHOLD
    network_sizes = sorted(df["network_size"].unique())
    warn_if_insufficient_network_sizes(network_sizes)
    metric_key = select_received_metric_key(rows, metric_key)
    metric_state = is_constant_metric(metric_values(rows, metric_key))
    if metric_state is not MetricStatus.OK:
        render_metric_status(fig, ax, metric_state, legend_handles=None)
        placement = place_adaptive_legend(fig, ax, preferred_loc="right")
        margins = {
            **legend_margins(
                placement.legend_loc,
                legend_rows=placement.legend_rows,
                fig=fig,
            ),
            "bottom": 0.16,
        }
        if wide_series:
            margins["wspace"] = WIDE_SERIES_WSPACE
        apply_figure_layout(
            fig,
            margins=margins,
            legend_rows=placement.legend_rows,
            legend_loc=placement.legend_loc,
        )
        return fig
    plot_metric_by_snir(
        ax,
        rows,
        metric_key,
        use_algo_styles=True,
        line_width=2.4,
        marker_size=6.5,
        percentile_line_width=1.4,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        clear_axis_legends(ax)
    else:
        handles, labels = fallback_legend_handles()
    ax.set_xlabel("Network size (nodes)")
    ax.set_ylabel("Tx frames (count, median, p10–p90)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.set_xticks(network_sizes)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    placement = place_adaptive_legend(
        fig,
        ax,
        preferred_loc="right",
        handles=handles,
        labels=labels,
    )
    margins = {
        **legend_margins(
            placement.legend_loc,
            legend_rows=placement.legend_rows,
            fig=fig,
        ),
        "bottom": 0.16,
    }
    if wide_series:
        margins["wspace"] = WIDE_SERIES_WSPACE
    apply_figure_layout(
        fig,
        margins=margins,
        legend_rows=placement.legend_rows,
        legend_loc=placement.legend_loc,
    )
    return fig


def _plot_summary_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    series_count = (
        df[["algo", "snir_mode"]].dropna().drop_duplicates().shape[0]
        if {"algo", "snir_mode"}.issubset(df.columns)
        else len(df.dropna().drop_duplicates())
    )
    fig, ax = plt.subplots(figsize=resolve_ieee_figsize_for_series(series_count))
    wide_series = series_count >= WIDE_SERIES_THRESHOLD
    metric_key = select_received_metric_key(rows, metric_key)
    metric_state = is_constant_metric(metric_values(rows, metric_key))
    if metric_state is not MetricStatus.OK:
        render_metric_status(fig, ax, metric_state, legend_handles=None)
        placement = place_adaptive_legend(fig, ax, preferred_loc="right")
        margins = {
            **legend_margins(
                placement.legend_loc,
                legend_rows=placement.legend_rows,
                fig=fig,
            ),
            "bottom": 0.16,
        }
        if wide_series:
            margins["wspace"] = WIDE_SERIES_WSPACE
        apply_figure_layout(
            fig,
            margins=margins,
            legend_rows=placement.legend_rows,
            legend_loc=placement.legend_loc,
        )
        return fig
    summary_handles, summary_labels = _add_summary_plot(ax, rows, metric_key)
    handles, labels = ax.get_legend_handles_labels()
    if summary_handles:
        handles = [*handles, *summary_handles]
        labels = [*labels, *summary_labels]
    if not handles:
        handles, labels = fallback_legend_handles()
    placement = place_adaptive_legend(
        fig,
        ax,
        preferred_loc="right",
        handles=handles,
        labels=labels,
    )
    margins = {
        **legend_margins(
            placement.legend_loc,
            legend_rows=placement.legend_rows,
            fig=fig,
        ),
        "bottom": 0.16,
    }
    if wide_series:
        margins["wspace"] = WIDE_SERIES_WSPACE
    apply_figure_layout(
        fig,
        margins=margins,
        legend_rows=placement.legend_rows,
        legend_loc=placement.legend_loc,
    )
    return fig


def main(argv: list[str] | None = None, allow_sample: bool = True) -> None:
    apply_plot_style()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--network-sizes",
        type=int,
        nargs="+",
        help="Filtrer les tailles de réseau (ex: --network-sizes 100 200 300).",
    )
    args = parser.parse_args(argv)
    step_dir = Path(__file__).resolve().parents[1]
    results_path = step_dir / "results" / "aggregated_results.csv"
    rows = load_step1_aggregated(results_path, allow_sample=allow_sample)
    if not rows:
        warnings.warn("CSV Step1 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows = filter_cluster(rows, "all")
    rows, _ = filter_rows_by_network_sizes(rows, args.network_sizes)
    rows = filter_mixra_opt_fallback(rows)

    fig = _plot_metric(rows, "sent_mean")
    fig_summary = _plot_summary_metric(rows, "sent_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S4", use_tight=False)
    save_figure(fig_summary, output_dir, "plot_S4_summary", use_tight=False)
    assert_legend_present(fig, "plot_S4")
    assert_legend_present(fig_summary, "plot_S4_summary")
    plt.close(fig)
    plt.close(fig_summary)


if __name__ == "__main__":
    main()
