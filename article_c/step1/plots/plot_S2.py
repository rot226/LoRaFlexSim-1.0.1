"""Trace la figure S2 (ToA moyen vs densité, SNIR on/off)."""

from __future__ import annotations

LAST_EFFECTIVE_SOURCE = "aggregates"
import argparse
from pathlib import Path
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
    fallback_legend_handles,
    legend_margins,
    load_step1_aggregated,
    metric_values as get_metric_values,
    place_adaptive_legend,
    plot_metric_by_snir,
    render_metric_status,
    save_figure,
    warn_metric_checks_by_group,
    warn_if_insufficient_network_sizes,
    auto_figsize_for_traces,
)
from article_c.common.plot_data_source import load_aggregated_rows_for_source
from plot_defaults import (
    WIDE_SERIES_WSPACE,
)


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    series_count = (
        df[["algo", "snir_mode"]].dropna().drop_duplicates().shape[0]
        if {"algo", "snir_mode"}.issubset(df.columns)
        else len(df.dropna().drop_duplicates())
    )
    fig, ax = plt.subplots(figsize=auto_figsize_for_traces(series_count))
    wide_series = series_count >= 3
    network_sizes = sorted(df["network_size"].unique())
    warn_if_insufficient_network_sizes(network_sizes)
    warn_metric_checks_by_group(
        rows,
        metric_key,
        x_key="network_size",
        label="ToA moyen",
        min_value=0.0,
        expected_monotonic="nondecreasing",
        group_keys=("algo", "snir_mode"),
    )
    metric_state = is_constant_metric(get_metric_values(rows, metric_key))
    if metric_state is not MetricStatus.OK:
        render_metric_status(fig, ax, metric_state, legend_handles=None)
        placement = place_adaptive_legend(fig, ax, preferred_loc="right")
        margins = legend_margins(
            placement.legend_loc,
            legend_rows=placement.legend_rows,
            fig=fig,
        )
        if wide_series:
            margins = {**margins, "wspace": WIDE_SERIES_WSPACE}
        apply_figure_layout(
            fig,
            margins=margins,
            legend_rows=placement.legend_rows,
            legend_loc=placement.legend_loc,
        )
        return fig
    plot_metric_by_snir(ax, rows, metric_key)
    ax.set_xlabel("Network size (nodes)")
    ax.set_ylabel("Mean ToA (s)")
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_xticks(network_sizes)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        handles, labels = fallback_legend_handles()
    placement = place_adaptive_legend(
        fig,
        ax,
        preferred_loc="right",
        handles=handles,
        labels=labels,
    )
    metric_series = pd.to_numeric(df[metric_key], errors="coerce").dropna()
    if not metric_series.empty:
        y_min = metric_series.min()
        y_max = metric_series.max()
        padding = max((y_max - y_min) * 0.05, 0.01)
        y_min = 0.0 if y_min >= 0 else y_min - padding
        y_max = y_max + padding
        ax.set_ylim(y_min, y_max)
    margins = {
        **legend_margins(
            placement.legend_loc,
            legend_rows=placement.legend_rows,
            fig=fig,
        ),
        "left": 0.16,
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


def main(argv: list[str] | None = None, allow_sample: bool = True, source: str = "aggregates") -> None:
    global LAST_EFFECTIVE_SOURCE
    LAST_EFFECTIVE_SOURCE = str(source).strip().lower()
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
    rows = load_aggregated_rows_for_source(
        step_dir=step_dir,
        source=LAST_EFFECTIVE_SOURCE,
        step_label="Step1",
        loader=load_step1_aggregated,
        allow_sample=allow_sample,
    )
    if not rows:
        warnings.warn("CSV Step1 manquant ou vide, figure ignorée.", stacklevel=2)
        return
    rows = filter_cluster(rows, "all")
    rows, _ = filter_rows_by_network_sizes(rows, args.network_sizes)
    rows = filter_mixra_opt_fallback(rows)

    fig = _plot_metric(rows, "mean_toa_s")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S2", use_tight=False)
    assert_legend_present(fig, "plot_S2")
    plt.close(fig)


if __name__ == "__main__":
    main()
