"""Trace la figure S4 (trames envoyées saturées vs densité, SNIR on/off)."""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import pandas as pd

from article_c.common.plot_helpers import (
    ALGO_COLORS,
    ALGO_LABELS,
    ALGO_MARKERS,
    SNIR_LABELS,
    SNIR_LINESTYLES,
    SNIR_MODES,
    apply_plot_style,
    apply_figure_layout,
    algo_label,
    ensure_network_size,
    filter_rows_by_network_sizes,
    filter_cluster,
    filter_mixra_opt_fallback,
    is_constant_metric,
    legend_margins,
    load_step1_aggregated,
    metric_values,
    plot_metric_by_snir,
    render_constant_metric,
    select_received_metric_key,
    resolve_percentile_keys,
    save_figure,
)
from article_c.step1.plots.plot_utils import configure_figure
from plot_defaults import resolve_ieee_figsize


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
) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    median_key, _, _ = resolve_percentile_keys(rows, metric_key)
    algos = sorted(df["algo"].dropna().unique(), key=_algo_sort_key)
    if not algos:
        return
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
            )
    ax.set_xticks(range(len(algos)))
    ax.set_xticklabels([algo_label(str(algo)) for algo in algos])
    ax.set_ylabel("Trames\n(médiane ± min/max)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.set_title("Synthèse min/médiane/max")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color="#222222",
            linestyle=SNIR_LINESTYLES.get(snir_mode, "solid"),
            label=SNIR_LABELS[snir_mode],
        )
        for snir_mode in SNIR_MODES
    ]
    ax.legend(
        handles=legend_handles,
        title="SNIR",
        ncol=2,
        frameon=False,
        loc="upper center",
        fontsize=8,
        title_fontsize=8,
    )


def _plot_metric(rows: list[dict[str, object]], metric_key: str) -> plt.Figure:
    ensure_network_size(rows)
    df = pd.DataFrame(rows)
    series_count = (
        df[["algo", "snir_mode"]].dropna().drop_duplicates().shape[0]
        if {"algo", "snir_mode"}.issubset(df.columns)
        else len(df.dropna().drop_duplicates())
    )
    fig, (ax, ax_summary) = plt.subplots(
        2,
        1,
        figsize=resolve_ieee_figsize(series_count),
        gridspec_kw={"height_ratios": [4.0, 1.4]},
    )
    network_sizes = sorted(df["network_size"].unique())
    if len(network_sizes) < 2:
        warnings.warn("Moins de deux tailles de réseau disponibles.", stacklevel=2)
    metric_key = select_received_metric_key(rows, metric_key)
    if is_constant_metric(metric_values(rows, metric_key)):
        render_constant_metric(fig, (ax, ax_summary), legend_handles=None)
        configure_figure(
            fig,
            (ax, ax_summary),
            "Step 1 - Sent Frames (budget saturant) vs Network size (number of nodes) "
            "(SNIR on/off)",
            legend_loc="above",
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
        existing_legend = ax.get_legend()
        if existing_legend is not None:
            existing_legend.remove()
    ax.set_xlabel("Network size (number of nodes)")
    ax.set_ylabel("Sent Frames (budget saturant, median, p10-p90)")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.yaxis.set_label_coords(-0.08, 0.5)
    ax.set_xticks(network_sizes)
    ax.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}"))
    configure_figure(
        fig,
        (ax, ax_summary),
        "Step 1 - Sent Frames (budget saturant) vs Network size (number of nodes) "
        "(SNIR on/off)",
        legend_loc="above",
        legend_handles=handles if handles else None,
        legend_labels=labels if handles else None,
    )
    _add_summary_plot(ax_summary, rows, metric_key)
    apply_figure_layout(
        fig,
        margins={
            **legend_margins("above"),
            "hspace": 0.55,
            "bottom": 0.16,
        },
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

    fig = _plot_metric(rows, "sent_mean")
    output_dir = step_dir / "plots" / "output"
    save_figure(fig, output_dir, "plot_S4", use_tight=False)
    plt.close(fig)


if __name__ == "__main__":
    main()
